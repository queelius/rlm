"""Process-local compatibility for legitimate zero-gradient PPO steps.

Only the pinned DeepSpeed 0.14.1 BF16 step assertion changes: >0 becomes >=0.
The native norm helper maps NaN/Inf to -1, which remains rejected. In particular,
zero-gradient steps still execute Adam: its moments, weight decay and step count
must not be skipped. This is a documented runtime deviation, not a PPO change.
"""

import hashlib
import inspect
import textwrap
from pathlib import Path

NATIVE_SHA256 = "dfb712446fb4dd0634b58f3a015eea184998274b74c27d6122eb6a0a6a70d4b4"


def install() -> dict:
    """Patch this Python process only; refuse any unverified installed source."""
    from deepspeed.runtime import bf16_optimizer

    optimizer = bf16_optimizer.BF16_Optimizer
    if hasattr(optimizer, "_spo_zero_norm_receipt"):
        return dict(optimizer._spo_zero_norm_receipt)
    source_path = Path(bf16_optimizer.__file__).resolve()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != NATIVE_SHA256:
        raise RuntimeError(f"Unverified DeepSpeed BF16 implementation: {digest}")
    source = textwrap.dedent(inspect.getsource(optimizer.step))
    old = "assert all_groups_norm > 0."
    if source.count(old) != 1:
        raise RuntimeError("Expected exactly one strict-positive gradient assertion")
    patched = source.replace(old, "assert all_groups_norm >= 0.")
    # Compile only our hash-pinned installed method, retaining its native globals.
    namespace = dict(vars(bf16_optimizer))
    exec(compile(patched, str(Path(__file__).resolve()) + ":patched-step", "exec"), namespace)
    optimizer.step = namespace["step"]
    receipt = {
        "adapter": "spo-bf16-zero-norm-v1",
        "native_file": str(source_path),
        "native_sha256": digest,
        "original_method_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_method_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "change": "strict-positive norm assertion accepts finite zero; no other change",
    }
    optimizer._spo_zero_norm_receipt = receipt
    return dict(receipt)
