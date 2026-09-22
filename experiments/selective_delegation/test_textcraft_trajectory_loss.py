import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch


def component():
    assert importlib.util.find_spec("textcraft_trajectory_loss") is not None
    import textcraft_trajectory_loss

    return textcraft_trajectory_loss


def complete_batch():
    episodes, calls = [], {}
    for g in range(8):
        for r in range(4):
            eid, cid = f"g{g}-r{r}", f"g{g}-r{r}-c0"
            episodes.append(
                dict(
                    episode_id=eid,
                    task_id=f"g{g}",
                    repeat=r,
                    observed=True,
                    native_score=int(g == 1 or (g == 0 and r == 0)),
                    call_ids=[cid],
                    errors={"invalid_schema": 1},
                )
            )
            calls[cid] = dict(
                call_id=cid,
                episode_id=eid,
                available=True,
                input_token_ids=[0, 1],
                output_token_ids=[2],
                request=dict(
                    input_token_ids=[0, 1],
                    task_id=f"g{g}",
                    adapter_sha256="policy",
                    adapter_commit_sha256="commit",
                    model_manifest_sha256="base",
                    sampling=dict(temperature=0.5, top_p=1.0, top_k=0),
                ),
            )
    return episodes, calls


def test_complete_batch_rloo_retains_invalid_actions_and_flat_groups():
    c = component()
    episodes, calls = complete_batch()
    credits = c.batch_credits(episodes, calls)
    assert len(credits) == 32
    assert [v.advantage for v in credits[:4]] == [1.0, -1 / 3, -1 / 3, -1 / 3]
    assert all(v.advantage == 0 for v in credits[4:])
    for changed_episodes, changed_calls in [
        (episodes[:-1], calls),
        ([dict(e, observed=False) if i == 2 else e for i, e in enumerate(episodes)], calls),
        (episodes, {k: v for k, v in calls.items() if k != "g0-r0-c0"}),
    ]:
        with pytest.raises(ValueError):
            c.batch_credits(changed_episodes, changed_calls)
    mixed = copy.deepcopy(calls)
    mixed["g0-r1-c0"]["request"]["adapter_sha256"] = "otherpolicy"
    with pytest.raises(ValueError, match="policy"):
        c.batch_credits(episodes, mixed)


def test_loss_sums_tokens_and_turns_with_exact_temperature_and_gradient_direction():
    c = component()
    logits_a = torch.zeros(1, 2, 3, requires_grad=True)
    logits_b = torch.zeros(1, 1, 3, requires_grad=True)
    loss = c.action_loss(logits_a, [1, 2], 1.0) + c.action_loss(logits_b, [0], 1.0)
    assert float(loss.detach()) == pytest.approx(3 * math.log(3) / 32)
    loss.backward()
    assert float(logits_a.grad[0, 0, 1]) == pytest.approx(-1 / 24)
    assert float(logits_a.grad[0, 1, 2]) == pytest.approx(-1 / 24)
    assert float(logits_b.grad[0, 0, 0]) == pytest.approx(-1 / 24)
    negative = torch.zeros(1, 1, 3, requires_grad=True)
    c.action_loss(negative, [0], -1 / 3).backward()
    assert float(negative.grad[0, 0, 0]) == pytest.approx(1 / 72)
    flat = torch.zeros(1, 1, 3, requires_grad=True)
    c.action_loss(flat, [0], 0.0).backward()
    assert torch.count_nonzero(flat.grad) == 0


def test_causal_alignment_never_retargets_prior_answers_or_feedback_and_adds_no_eos():
    c = component()

    class Positions(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.zeros(8, 6))
            self.device = torch.device("cpu")

        def forward(self, input_ids, use_cache, logits_to_keep):
            assert use_cache is False
            from types import SimpleNamespace

            return SimpleNamespace(logits=self.table[: input_ids.shape[1]][-logits_to_keep:][None])

    # Token3 is an actually emitted EOS in turn1. Tokens2,3 recur only in turn2's prefix.
    first = dict(input_token_ids=[0, 1], output_token_ids=[2, 3])
    second = dict(input_token_ids=[0, 1, 2, 3, 5], output_token_ids=[4])
    assert c.causal_inputs(first) == ([0, 1, 2], [2, 3])
    assert c.causal_inputs(second) == ([0, 1, 2, 3, 5], [4])
    model = Positions()
    c.replay_action_loss(model, second, 1.0).backward()
    assert torch.count_nonzero(model.table.grad[:4]) == 0
    assert model.table.grad[4, 4] < 0
    assert torch.count_nonzero(model.table.grad[5:]) == 0
    with pytest.raises(ValueError, match="shape"):
        c.action_loss(torch.zeros(1, 5, 6), [4], 1.0)


def test_actual_saved057_native_layout_uses_only_emitted_suffix_without_weights():
    c = component()
    path = Path(
        "/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/"
        "textcraft-public-discovery-readout-001/calls/t00-r0-flat-original-c000.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "94b7fa4d1f08959dc3d8c556b75beff5e723ffc786fdaf2544cdcfbe1f9e3bb3"
    )
    record = json.loads(path.read_text())
    ids, targets = c.causal_inputs(record)
    assert len(record["input_token_ids"]) == 445 and len(targets) == 17
    assert ids == record["request"]["input_token_ids"] + targets[:-1]
    assert targets[-1] == 151645 and len(ids) == 461
    assert record["request"]["sampling"]["temperature"] == 0.5
    # No tokenizer reconstruction or synthetic terminal marker changes the native token evidence.
    truncated = dict(record, output_token_ids=record["output_token_ids"][:-1])
    assert len(c.causal_inputs(truncated)[1]) == 16
