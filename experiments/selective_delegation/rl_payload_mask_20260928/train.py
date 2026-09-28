"""One independently warm-started masked update; parent alone assigns the GPU."""

import argparse
import json

import payload as m


def run(args):
    record = m.masks_record()
    masks = {cid: row["mask"] for cid, row in record["records"].items()}
    if m.binding_receipt() != m.p.read(m.STUDY / "PRIVATE-ADAPTER.json"):
        raise ValueError("reviewed private source transformation changed")
    if not args.prepare_only:
        baseline = m.endpoint(m.ORIGINAL / "train-0001")
        if baseline is None:
            m.p.c.save(
                m.TRAIN / "CONDITIONAL-SKIP.json",
                dict(
                    reason="No actual usable full-token baseline; no masked GPU update",
                    GPU_loaded=False,
                    endpoint=None,
                ),
            )
            return
        if (
            baseline["training_plan_sha256"]
            != m.p.read(m.TRAIN / "PLAN.json")["baseline_full_plan_sha256"]
        ):
            raise ValueError("accepted baseline PLAN changed")
        m.persist(m.TRAIN / "BASELINE-ENDPOINT.json", baseline)
    m.bound_trainer(m.TRAIN, masks).run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.collection, args.output = m.COLLECTION, m.TRAIN
    run(args)
    if args.prepare_only:
        print(json.dumps(dict(plan=str(m.TRAIN / "PLAN.json"), GPU_loaded=False)))
