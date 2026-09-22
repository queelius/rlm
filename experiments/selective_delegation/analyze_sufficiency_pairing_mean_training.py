"""Analysis003: JSON-canonical pairing credit plus saved tensor/Adam verification."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

BASE = Path(__file__).with_name("pairing_mean_audit_base.py")
spec = importlib.util.spec_from_file_location("pairing_audit_base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def coefficient_map(groups):
    """Match source058's formula, but match JSON receipt arrays rather than Python tuples."""
    if len(groups) != 16 or any(len(g) != 4 or any(p is None for p in g) for g in groups):
        raise ValueError("complete observed16x4 product groups required")
    advantages = [
        [
            list(pair)
            for pair in base.marginal_credit(
                [candidate["positive_success"] for candidate in group],
                [candidate["negative_success"] for candidate in group],
            )
        ]
        for group in groups
    ]
    return {
        f"p{i:02d}-k{k}-v{v}": coefficient
        for i, group in enumerate(advantages)
        for k, pair in enumerate(group)
        for v, coefficient in enumerate(pair)
    }, advantages


def adapter_delta(before, after):
    from safetensors import safe_open

    total = 0.0
    with (
        safe_open(before, framework="pt", device="cpu") as left,
        safe_open(after, framework="pt", device="cpu") as right,
    ):
        keys = list(left.keys())
        if not keys or keys != list(right.keys()) or any("lora_" not in key for key in keys):
            raise ValueError("adapter tensor inventory differs from LoRA-only contract")
        for key in keys:
            a, b = left.get_tensor(key), right.get_tensor(key)
            if a.shape != b.shape:
                raise ValueError("adapter tensor shape differs")
            total += float((b.double() - a.double()).square().sum())
    return math.sqrt(total)


def tensor_checks(output, report):
    import torch

    # The base audit deliberately keeps its public report compact.  Re-read the
    # authenticated committed inventory instead of assuming it serializes boundaries.
    boundaries = base.shared.training.boundary_inventory(output)
    result = []
    for batch in report["batches"]:
        if not batch["committed"]:
            continue
        cursor = batch["sample_cursor"]
        before, after = boundaries[cursor - 1], boundaries[cursor]
        delta = adapter_delta(
            Path(before["checkpoint"]) / "adapter_model.safetensors",
            Path(after["checkpoint"]) / "adapter_model.safetensors",
        )
        update = json.loads(
            (output / "batches" / f"sample-{cursor:04d}" / "UPDATE.json").read_text()
        )
        state = after["state"]
        optimizer = torch.load(
            Path(after["checkpoint"]) / "optimizer.pt", map_location="cpu", weights_only=True
        )
        steps = {int(value["step"]) for value in optimizer["state"].values()}
        before_optimizer = json.loads((Path(before["checkpoint"]) / "COMMIT.json").read_text())[
            "files"
        ]["optimizer.pt"]
        after_optimizer = json.loads((Path(after["checkpoint"]) / "COMMIT.json").read_text())[
            "files"
        ]["optimizer.pt"]
        updated = batch["updated"]
        if updated:
            if not (
                math.isfinite(update["gradient_norm"])
                and update["gradient_norm"] > 0
                and math.isfinite(delta)
                and delta > 0
                and steps == {state["step"]}
                and before_optimizer != after_optimizer
            ):
                raise ValueError("actual Adam/tensor evidence is non-finite or absent")
        elif (
            update.get("optimizer_called") is not False
            or delta != 0
            or before_optimizer != after_optimizer
        ):
            raise ValueError("zero-credit block changed Adam or adapter")
        result.append(
            {
                "cursor": cursor,
                "updated": updated,
                "adapter_delta_l2": delta,
                "adam_steps": sorted(steps),
            }
        )
    return result


def main(args):
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    from transformers import AutoTokenizer

    base.coefficient_map = coefficient_map
    tokenizer = AutoTokenizer.from_pretrained(
        base.shared.training.BASE, local_files_only=True, trust_remote_code=False
    )
    report = base.analyze(args.output, tokenizer)
    report["schema"] = "sufficiency-pairing-mean-training-audit-v3"
    report["tensor_optimizer_checks"] = tensor_checks(args.output.resolve(), report)
    report["analyzer_sha256"] = {
        str(Path(__file__).resolve()): base.shared.paired.baseline.panel.sha256(Path(__file__)),
        str(BASE): base.shared.paired.baseline.panel.sha256(BASE),
    }
    base.shared.paired.baseline.native.save(args.report, report)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Pairing-mean TRAIN audit\n\n"
            "Terminal-only TRAIN receipt: product reward and pairing-mean response credit are "
            "reported separately; no held claim.\n"
        )
    print(json.dumps({"report": str(args.report)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    main(parser.parse_args())
