"""Opt-in private Oat copy: bound diagnostic entropy memory, preserve the RL objective.

Run with the pinned R1 Python and --output-root NEW_DIRECTORY, then prepend that
directory to PYTHONPATH for the entire training owner and all Launchpad children.
The shared package is never modified. This command prepares files, never trains.
"""

import argparse
import hashlib
import inspect
import json
import sysconfig
from pathlib import Path

PPO_SHA256 = "e43237dd1f015c54f0a3b417ad94d7d042dd56e23f706116d84033245ce3d132"
OPS_SHA256 = "8983141f9afa6c02764994d85e9ca013874579f07846c037b3383cab32555601"


def chunked_entropy_from_logits(logits, token_chunk_size: int = 128):
    """Use the original per-token formula, bounding full-vocabulary temporaries."""
    import torch

    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if logits.shape[-2] == 0:
        return logits.sum(dim=-1)
    values = []
    for chunk in logits.split(token_chunk_size, dim=-2):
        pd = torch.nn.functional.softmax(chunk, dim=-1)
        values.append(torch.logsumexp(chunk, dim=-1) - torch.sum(pd * chunk, dim=-1))
    return torch.cat(values, dim=-1)


def patch_ppo(original: bytes) -> bytes:
    """Accept exactly the audited installed source; change only two memory lifetimes."""
    if hashlib.sha256(original).hexdigest() != PPO_SHA256:
        raise ValueError("Original oat/algorithms/ppo.py hash differs from audited source")
    source = original.decode()
    assignment = " " * 16 + "logps[mini_batch_inds, : mb_last_valid_token_pos - 1] = batch_logps"
    entropy = "                    entropy = entropy_from_logits(logits[:, :-1])"
    for old, new in (
        (assignment, assignment + "\n                del batch_logits, batch_logps"),
        (entropy, entropy.replace("entropy_from_logits", "chunked_entropy_from_logits")),
    ):
        if source.count(old) != 1:
            raise ValueError("Audited patch anchor is not unique")
        source = source.replace(old, new, 1)
    source += "\n\n# Explicit, attempt-private diagnostic-memory adapter.\n"
    source += inspect.getsource(chunked_entropy_from_logits)
    compile(source, "<private-oat-ppo>", "exec")
    return source.encode()


def prepare_copy(package: Path, output_root: Path) -> dict:
    """Copy the small package without caches, recording immutable file-level provenance."""
    package = package.resolve()
    original = (package / "algorithms/ppo.py").read_bytes()
    patched = patch_ppo(original)
    if hashlib.sha256((package / "utils/ops.py").read_bytes()).hexdigest() != OPS_SHA256:
        raise ValueError("Original oat/utils/ops.py hash differs from audited source")
    output_root.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        source_hash = hashlib.sha256(data).hexdigest()
        if relative.as_posix() == "algorithms/ppo.py":
            if data != original:
                raise ValueError("Original PPO source changed during preparation")
            data = patched
        destination = output_root / "oat" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        files[relative.as_posix()] = {
            "source_sha256": source_hash,
            "copy_sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    manifest = {
        "source_package": str(package),
        "pythonpath_root": str(output_root.resolve()),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "changed_files": [
            name for name, item in files.items() if item["source_sha256"] != item["copy_sha256"]
        ],
        "file_count": len(files),
        "entropy_token_chunk_size": 128,
        "scope": "Diagnostic entropy chunking and release of unused no-grad old-policy logits only",
        "files": files,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-package", type=Path, default=Path(sysconfig.get_paths()["purelib"]) / "oat"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_copy(args.source_package, args.output_root)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
