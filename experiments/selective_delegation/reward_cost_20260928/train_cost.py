"""One changed-reward update from the same warm actor; no new rollouts or native relabeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cost_reward as r


class FlatCompositeReward(ValueError):
    def __init__(self, admission):
        super().__init__("No within-task composite reward variation")
        self.admission = admission


def configure_trainer(output: Path):
    trainer = r.f.load_legacy("train.py")
    original_load = trainer.load_batch

    def load_batch(directory, plan):
        calls, _, captured, admission = original_load(directory, plan)
        collected = r.f.read(directory / "PLAN.json")
        native = r.f.read(directory / "NATIVE-AUDIT.json")
        episodes = [
            r.f.read(directory / "episodes" / (j["episode_id"] + ".json"))
            for j in collected["jobs"]
        ]
        for episode in episodes:
            audited = native["audits"][episode["episode_id"]]
            if (
                not audited["replayed"]
                or audited["native_score"] != episode["native_score"]
                or audited["errors"] != episode["errors"]
                or audited["calls"] != episode["global_calls"]
            ):
                raise ValueError("error costs must match the independent native replay")
        credits, rewards = r.derive_credits(episodes, calls)
        rewards.update(
            collection=str(directory.resolve()),
            collection_plan_sha256=plan["collection_plan_sha256"],
            native_audit_sha256=admission["collection_audit_sha256"],
        )
        r.f.persist(output / "REWARD-TABLE.json", rewards)
        admission.update(
            reward_table_sha256=r.f.sha(output / "REWARD-TABLE.json"),
            mixed_native_groups=rewards["mixed_native_groups"],
            mixed_composite_groups=rewards["mixed_composite_groups"],
            all_failure_cost_varying_groups=rewards["all_failure_cost_varying_groups"],
            all_success_cost_varying_groups=rewards["all_success_cost_varying_groups"],
            probability_change_sign="Positive/negative refer to composite advantage, "
            "not native success/failure; native labels remain in REWARD-TABLE.json",
        )
        if not any(c.advantage for c in credits):
            raise FlatCompositeReward(admission)
        return calls, credits, captured, admission

    trainer.prepare = r.prepare_training
    trainer.load_batch = load_batch
    return trainer


def run(args):
    if not args.prepare_only and (
        list(args.output.glob("OWNER-*.json")) or (args.output / "CONDITIONAL-SKIP.json").exists()
    ):
        raise ValueError("existing attempted cost branch; no silent retry")
    if not args.prepare_only and not r.collection_ready(args.collection):
        r.prepare_training(args)
        r.skip(args.output, "First fresh collection absent/incomplete; no partial-batch update")
        print(json.dumps(dict(skipped=True, GPU_loaded=False, output=str(args.output))))
        return
    try:
        configure_trainer(args.output).run(args)
    except FlatCompositeReward as exc:
        r.skip(args.output, str(exc), admission=exc.admission)
        print(json.dumps(dict(skipped=True, GPU_loaded=False, output=str(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
