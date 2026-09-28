"""Pinned official Phi assets; small assets first, weights only after CPU qualification."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

REPO = "microsoft/Phi-4-mini-instruct"
REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
MODEL = Path("/project/alex_phd/research-cache/models") / (
    "microsoft--Phi-4-mini-instruct--" + REVISION
)
ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "textcraft-phi-transfer-20260928-001"
SMALL = (
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "config.json",
    "generation_config.json",
    "added_tokens.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors.index.json",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def retrieve(name, info):
    metadata = next(item for item in info["siblings"] if item["rfilename"] == name)
    path = MODEL / name
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    started = time.monotonic()
    if path.exists():
        raise FileExistsError("never overwrite cached asset: " + str(path))
    partial = path.with_name(path.name + ".partial")
    digest, count = hashlib.sha256(), 0
    with urlopen(url, timeout=60) as response, partial.open("xb") as stream:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            count += len(chunk)
    checksum = digest.hexdigest()
    if count != metadata["size"] or ("lfs" in metadata and checksum != metadata["lfs"]["sha256"]):
        raise ValueError("official size/LFS hash differs; partial retained: " + name)
    if "lfs" not in metadata:
        blob = hashlib.sha1(f"blob {count}\0".encode() + partial.read_bytes()).hexdigest()
        if blob != metadata["blobId"]:
            raise ValueError("official Git blob differs; partial retained: " + name)
    partial.rename(path)
    return dict(
        name=name,
        url=url,
        bytes=count,
        sha256=checksum,
        official=metadata,
        elapsed_seconds=time.monotonic() - started,
    )


def acquire(weights=False):
    MODEL.mkdir(parents=True, exist_ok=True)
    info_path = MODEL / "official-api-revision.json"
    if info_path.exists():
        info = read(info_path)
    else:
        url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
        with urlopen(url, timeout=30) as response:
            info = json.load(response)
        if info["sha"] != REVISION or info["id"] != REPO or info["private"] or info["gated"]:
            raise ValueError("official pinned public model identity differs")
        save(info_path, info)
    if weights:
        qualification = read(OUTPUT / "QUALIFICATION.json")
        if not qualification["ready_for_weights"] or qualification["model_revision"] != REVISION:
            raise ValueError("CPU native architecture/tokenizer qualification required first")
        names = sorted(set(read(MODEL / "model.safetensors.index.json")["weight_map"].values()))
        if any(not name.endswith(".safetensors") for name in names):
            raise ValueError("only pinned safetensors weights allowed")
    else:
        names = list(SMALL)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda name: retrieve(name, info), names))
    manifest = dict(
        schema="phi-official-assets-20260928-v1",
        repo=REPO,
        revision=REVISION,
        source_url=f"https://huggingface.co/{REPO}/tree/{REVISION}",
        upstream_last_modified=info["lastModified"],
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        license="MIT",
        license_sha256=sha(MODEL / "LICENSE"),
        metadata_sha256=sha(info_path),
        assets=receipts,
        weights=weights,
        custom_python_downloaded_or_executed=False,
        acquirer_sha256=sha(Path(__file__)),
    )
    path = MODEL / ("WEIGHT-ACQUISITION.json" if weights else "SMALL-ASSET-ACQUISITION.json")
    save(path, manifest)
    if weights:
        combined = read(MODEL / "SMALL-ASSET-ACQUISITION.json")["assets"] + receipts
        save(
            MODEL / "local-research-manifest.json",
            dict(
                repo_id=REPO,
                revision=REVISION,
                license="MIT",
                retrieved_utc=manifest["retrieved_utc"],
                files={row["name"]: row["sha256"] for row in combined},
                total_bytes=sum(row["bytes"] for row in combined),
                small_assets_sha256=sha(MODEL / "SMALL-ASSET-ACQUISITION.json"),
                weights_sha256=sha(path),
                qualification_sha256=sha(OUTPUT / "QUALIFICATION.json"),
                trust_remote_code=False,
            ),
        )
    return dict(path=str(path), sha256=sha(path), bytes=sum(row["bytes"] for row in receipts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquire(args.weights), indent=2))
