"""Exactly 16 paired binder episodes from the authenticated masked endpoint."""

import argparse
import json
from types import SimpleNamespace

import payload as m


def prepare(args, _entrypoint):
    if args.hours != 0.75 or args.output != m.READOUT:
        raise ValueError("fixed 45-minute readout required")
    endpoint = m.endpoint(m.TRAIN)
    if endpoint is None or endpoint["path"] != str(args.checkpoint):
        raise ValueError("real usable masked endpoint required")
    warm_path = m.ORIGINAL / "readout-warm/PLAN.json"
    plan = dict(m.p.read(warm_path))
    if plan["planned_episodes"] != 16 or plan["phase"] != "readout":
        raise ValueError("fixed original 16-slot diagnostic required")
    plan.update(
        schema="textcraft-payload-mask-binder-readout-20260928-v1",
        adapter={k: endpoint[k] for k in ("path", "sha256", "commit_sha256")},
        budget_seconds=2700,
        source_sha256=m.source_pins(),
        masked_training_plan_sha256=endpoint["training_plan_sha256"],
        paired_warm_plan_sha256=m.p.sha(warm_path),
        intervention="Payload loss mask only during one update; unchanged binder inference",
    )
    m.persist(m.READOUT / "PLAN.json", plan)
    m.persist(m.READOUT / "ENDPOINT.json", endpoint)
    return plan, m.p.tasks()


def run(args):
    endpoint = m.endpoint(m.TRAIN)
    audit_path = m.TRAIN / "ENDPOINT-AUDIT.json"
    audit = m.p.read(audit_path) if audit_path.exists() else {}
    if endpoint is None or not audit.get("passed"):
        m.p.c.save(
            m.READOUT / "CONDITIONAL-SKIP.json",
            dict(
                reason="Masked endpoint absent, unusable or unaudited; outcomes unknown",
                GPU_loaded=False,
            ),
        )
        return
    if audit["binding"] != endpoint:
        raise ValueError("CPU-audited endpoint binding changed")
    args.output, args.checkpoint = m.READOUT, m.Path(endpoint["path"])
    args.mode, args.phase, args.update, args.prepare_only = "binder", "readout", 1, False
    collector = m.private(m.LEGACY / "collect.py")
    collector.p = SimpleNamespace(**vars(m.p))
    collector.p.prepare_collection = prepare
    collector.__file__ = str(m.HERE / "readout.py")
    collector.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=0.75)
    run(parser.parse_args())
    print(json.dumps(dict(output=str(m.READOUT))))
