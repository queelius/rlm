import pytest
import textcraft_positive_replay as run
import torch


def test_positive_credit_preserves_weights_and_fixed_denominator():
    cls = run.rl.loss_math.ActionCredit
    credits = [cls("t", "e", str(i), a) for i, a in enumerate((-1.0, 0.0, 1 / 3, 2 / 3))]
    kept = run.positive_credits(credits)
    assert kept == credits[2:] and kept[0] is credits[2]
    logits = torch.zeros((1, 2, 3), requires_grad=True)
    loss = sum(run.rl.loss_math.action_loss(logits, [0, 1], c.advantage) for c in kept)
    assert loss.item() == pytest.approx(2 * torch.log(torch.tensor(3.0)).item() / 32)
    loss.backward()
    assert logits.grad[0, 0, 0] < 0


def test_actual_saved_batch_positive_inventory_and_native_replay():
    from transformers import AutoTokenizer

    plan = run.read(run.SOURCE / "PLAN.json")
    original = run.read(run.Path(plan["first_batch"]["output"]) / "PLAN.json")
    tasks = {t["id"]: t for t in run.rl.readiness.runtime_tasks(original)}
    tokenizer = AutoTokenizer.from_pretrained(
        run.c.BASE, local_files_only=True, trust_remote_code=False
    )
    episodes, calls, credits, audits = run.rl.native_batch(
        run.DATA, run.read(run.DATA / "PLAN.json"), tokenizer, tasks, run.c.bridge.load_world()
    )
    kept = run.positive_credits(credits)
    assert len(episodes) == len(audits) == 32 and len(calls) == 909
    assert len({c.episode_id for c in kept}) == 12
    assert len(kept) == 399
    assert sum(len(calls[c.call_id]["output_token_ids"]) for c in kept) == 12927
    assert sum(c.advantage != 0 for c in credits) == 819


def test_endpoint_rejects_uncommitted_or_failed_step():
    good = dict(complete=True, failure=None, stopped=False, actual_optimizer_steps=2)
    run.valid_terminal(good)
    for changed in (
        dict(actual_optimizer_steps=1),
        dict(failure="nonfinite"),
        dict(complete=False),
    ):
        with pytest.raises(ValueError):
            run.valid_terminal({**good, **changed})
