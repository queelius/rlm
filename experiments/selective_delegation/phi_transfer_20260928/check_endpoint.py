"""CPU-only authentication of complete Phi checkpoint23 and actual training dose."""

import argparse
import json
import math
from pathlib import Path

import acquire
import evaluate
import qualify


def check(teacher):
    from safetensors import safe_open

    binding = evaluate.endpoint(teacher)
    adapter = Path(binding["path"])
    config = acquire.read(adapter / "adapter_config.json")
    if set(config["target_modules"]) != set(qualify.TARGET_MODULES):
        raise ValueError("actual Phi adapter target modules differ")
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as file:
        tensor_names = file.keys()
        trainable = sum(math.prod(file.get_slice(key).get_shape()) for key in tensor_names)
    qualification = acquire.read(acquire.OUTPUT / "QUALIFICATION.json")
    if trainable != qualification["lora_trainable_parameters"]:
        raise ValueError("actual Phi adapter parameter count differs")
    steps = [acquire.read(path) for path in sorted((adapter.parent / "steps").glob("*.json"))]
    tokens = sum(step["target_tokens"] for step in steps)
    if tokens != qualification["teaching_doses"][teacher]["supervised_tokens"]:
        raise ValueError("actual Phi supervised token dose differs from native template")
    return dict(
        schema="phi-complete-endpoint-cpu-audit-20260928-v1",
        passed=True,
        teacher=teacher,
        binding=binding,
        trainable_parameters=trainable,
        actual_supervised_tokens=tokens,
        source_sha256={str(Path(__file__).resolve()): acquire.sha(Path(__file__))},
        gpu_used=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("discovery", "known"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.teacher)
    acquire.save(args.output, report)
    print(json.dumps(report, indent=2))
