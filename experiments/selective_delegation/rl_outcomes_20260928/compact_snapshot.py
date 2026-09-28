"""One-shot contrast of completed compact/native audits, no polling or new rollouts."""

import json
from collections import Counter
from pathlib import Path

import analyze as a


def snapshot():
    worlds, pins = {}, {}
    for world in (42, 50):
        cells, plans = {}, {}
        for mode, directory, audit_name in (
            (
                "compact",
                a.ROOT / f"textcraft-compact-p00-w{world}-20260928-001",
                "COMPACT-AUDIT.json",
            ),
            (
                "full_binder",
                a.ROOT / f"textcraft-breadth-p00-w{world}-soriginal-binder-001",
                "NATIVE-AUDIT.json",
            ),
        ):
            audit_path, plan_path = directory / audit_name, directory / "PLAN.json"
            if not audit_path.exists():
                cells[mode] = dict(available=False, status="native audit not complete at snapshot")
                continue
            audit = a.read(audit_path)
            plan = a.read(plan_path, audit["sha256"][str(plan_path)])
            assert not audit["terminal"]["failure"]
            plans[mode] = plan
            pins.update({str(audit_path): a.sha(audit_path), str(plan_path): a.sha(plan_path)})
            values, errors = {}, Counter()
            for job in plan["jobs"]:
                episode_path = directory / "episodes" / (job["episode_id"] + ".json")
                episode = a.read(episode_path, audit["sha256"][str(episode_path)])
                receipt = audit["audits"][job["episode_id"]]
                assert receipt["replayed"] and receipt["native_score"] == episode["native_score"]
                values[job["episode_id"]] = receipt["native_score"]
                errors.update(receipt["errors"])
            cells[mode] = dict(
                available=True,
                native_scores=values,
                successes=sum(values.values()),
                episodes=len(values),
                errors=dict(errors),
                physical_cost=audit["physical_cost"],
            )
        paired = None
        if all(cell["available"] for cell in cells.values()):
            compact, full = plans["compact"], plans["full_binder"]
            assert [(j["task_id"], j["repeat"], j["seed"]) for j in compact["jobs"]] == [
                (j["task_id"], j["repeat"], j["seed"]) for j in full["jobs"]
            ]
            for field in (
                "tasks_sha256",
                "world_sha256",
                "model_manifest_sha256",
                "sampling",
                "max_global_calls",
                "max_global_output_tokens",
                "max_new_tokens",
                "input_plus_output_limit",
            ):
                assert compact[field] == full[field]
            first, second = cells["compact"], cells["full_binder"]
            deltas = [
                value - second["native_scores"][key]
                for key, value in first["native_scores"].items()
            ]
            paired = dict(
                wins=sum(v > 0 for v in deltas),
                losses=sum(v < 0 for v in deltas),
                ties=sum(v == 0 for v in deltas),
                cost_change_percent={
                    key: 100 * (first["physical_cost"][key] / value - 1)
                    for key, value in second["physical_cost"].items()
                    if value
                    and key
                    in ("calls", "prompt_tokens", "completion_tokens", "native_service_seconds")
                },
            )
        worlds[str(world)] = dict(cells=cells, compact_minus_full_binder=paired)
    return dict(
        worlds=worlds,
        inputs_sha256=pins,
        source_sha256=a.sha(Path(__file__)),
        analysis_helper_sha256=a.sha(Path(a.__file__)),
        scope="Snapshot of actual completed native audits; no generic flat/recursive summary "
        "as compact comparison, no polling or new rollout. Panel00 eight VAL goals repeated across "
        "two worlds, not32 independent tasks. Different interface/warm SFT token dose and "
        "unobserved recipe behavior; equal23 updates not equal compute. Not an RL contrast.",
    )


if __name__ == "__main__":
    result = snapshot()
    a.save(a.OUTPUT / "COMPACT-SNAPSHOT.json", result)
    print(json.dumps(result["worlds"], indent=2))
