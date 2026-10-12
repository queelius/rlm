"""One-off GPU diagnostic: restore native SPO state and exercise rejected PPO steps.

Never saves modified weights. Synthetic old logprobs trigger the observed ratio
guard on a real saved trajectory. This is not a scientific training endpoint.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from bf16_zero_norm import install
from datasets import load_from_disk


def exact_tensor(actual, saved):
    """Compare every element without allocating a second full GPU state."""
    assert actual.shape == saved.shape
    assert actual.dtype == saved.dtype
    actual = actual.detach().reshape(-1)
    saved = saved.detach().reshape(-1)
    for start in range(0, actual.numel(), 1048576):
        left = actual[start : start + 1048576].cpu()
        right = saved[start : start + 1048576]
        assert torch.isfinite(left).all()
        assert torch.equal(left, right), f"Restore mismatch at element {start}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path("/project/alex_phd/research-cache/repos/SPO")
    sys.path.insert(0, str(source / "src"))
    from treetune.main import EntryPoint
    from treetune.trainers.data_collator import PPODataCollator
    from treetune.trainers.policy_trainer import Checkpoint

    start = time.time()
    receipt = {"started": start, "adapter": install(), "diagnostic_only": True}
    runtime = EntryPoint(args.configs)._runtime
    trainer = runtime._construct_trainer(init_model_only=False)
    trainer.load_checkpoint(Checkpoint(path=args.checkpoint, iteration=490))
    actor = trainer._init_actor_model()
    trainer._load_checkpoint_to_ds_engines(args.checkpoint, actor)
    assert trainer.state.global_step == actor.global_steps == 7840
    assert actor.lr_scheduler.last_batch_iteration == 7839
    assert actor.gradient_accumulation_steps() == 4
    assert actor.train_batch_size() == 64
    assert actor.optimizer.__class__.__name__ == "BF16_Optimizer"
    assert actor.optimizer.optimizer.__class__.__name__ == "FusedAdam"
    receipt.update(
        restored_step=actor.global_steps,
        restored_scheduler=actor.lr_scheduler.last_batch_iteration,
        trainer_state=trainer.state.state_dict(),
        optimizer_class=actor.optimizer.optimizer.__class__.__name__,
    )
    print("RESTORED_NATIVE_COUNTERS " + json.dumps(receipt), flush=True)

    state_path = (
        args.checkpoint / "actor/global_step7840/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    )
    state = torch.load(str(state_path), map_location="cpu", mmap=True)["optimizer_state_dict"]
    current = actor.optimizer.state_dict()
    compared = 0
    for actual, saved in zip(
        current["single_partition_of_fp32_groups"],
        state["single_partition_of_fp32_groups"],
        strict=True,
    ):
        exact_tensor(actual, saved)
        compared += actual.numel()
    current_base = current["base_optimizer_state"]
    saved_base = state["base_optimizer_state"]
    assert current_base["param_groups"] == saved_base["param_groups"]
    assert current_base["state"].keys() == saved_base["state"].keys()
    for key, values in saved_base["state"].items():
        assert current_base["state"][key].keys() == values.keys()
        for field, saved in values.items():
            actual = current_base["state"][key][field]
            if isinstance(saved, torch.Tensor):
                exact_tensor(actual, saved)
                compared += actual.numel()
            else:
                assert actual == saved
    receipt["restored_finite_tensor_elements_exactly_compared"] = compared
    print("RESTORE_TENSORS_MATCH " + str(compared), flush=True)
    del state, current, current_base, saved_base

    episodes = load_from_disk(str(args.episodes))
    row = episodes[0]  # Prospectively fixed first row, not selected by reward.
    receipt["saved_response"] = {
        "row": 0,
        "query": row["query_text"],
        "response": row["response_text"],
        "native_reward": row["reward"],
    }
    collator = PPODataCollator()

    def batch():
        # Native forward may modify its local labels view; always collate afresh.
        return {k: v.to(actor.device) for k, v in collator([row]).items()}

    actor.eval()
    with torch.no_grad():
        logps = trainer._forward_pass_actor(actor, batch(), return_all_logps=True)["all_logps"]
    assert torch.isfinite(logps).all()
    receipt["saved_response_logprobs_finite"] = True
    print("NATIVE_SAVED_RESPONSE_FORWARD_OK " + str(logps.shape), flush=True)
    actor.train()
    norms = []
    observed_ratios = []
    for ratio in (95.34, 36.98, 153.45, 10.01):
        inputs = batch()
        action_mask = inputs["labels"][:, 1:] != -100
        loss, skipped, metrics, _ = trainer._compute_actor_loss(
            actor,
            model_inputs=inputs,
            shifted_labels_mask=action_mask,
            old_logprobs=logps.detach().clone() - math.log(ratio),
            ref_logprobs=logps.detach().clone(),
            advantages=inputs["advantages"][:, 1:],
        )
        assert skipped and torch.isfinite(loss) and float(loss.detach()) == 0.0
        observed_ratios.append(float(metrics["actor/ratio"]))
        actor.backward(loss)
        actor.step()
        del inputs, loss, metrics
    norms.append(float(actor.optimizer._global_grad_norm))
    assert norms == [0.0]
    assert actor.global_steps == 7841
    assert actor.lr_scheduler.last_batch_iteration == 7840
    assert all(int(values["step"]) == 7841 for values in actor.optimizer.optimizer.state.values())
    receipt.update(
        complete=True,
        synthetic_ratio_guard_values=observed_ratios,
        diagnostic_global_step=actor.global_steps,
        diagnostic_scheduler=actor.lr_scheduler.last_batch_iteration,
        diagnostic_gradient_norm=norms[0],
        modified_weights_saved=False,
        ended=time.time(),
    )
    receipt["sources"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path(__file__).with_name("bf16_zero_norm.py")]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("DIAGNOSTIC_COMPLETE " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
