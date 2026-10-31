"""Paths and immutable artifact helpers for the research extension only."""
import csv
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results/research_extension/paired_calibration_v1"
SOURCE_ROOT = Path(__file__).resolve().parent
GLOBAL_T = 2.124186626456894
GLOBAL_TAU = 0.386413580715649
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
MODEL_REVISION = "7a91500a7086b805eeb868719ea5542a3e0bdeb3"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refuse to replace artifact: {path}")
        return
    with path.open("x") as handle:
        handle.write(content)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    import io
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != content.encode():
            raise FileExistsError(f"Refuse to replace artifact: {path}")
        return
    with path.open("x", newline="") as handle:
        handle.write(content)


def load_protocol(out_dir=OUTPUT_ROOT):
    out_dir = Path(out_dir)
    protocol_path = out_dir / "protocol.json"
    protocol = read_json(protocol_path)
    record = read_json(out_dir / "protocol_freeze.json")
    digest = sha256(protocol_path)
    if record["protocol_sha256"] != digest:
        raise RuntimeError("Frozen protocol hash mismatch")
    if Path(protocol["output_directory"]).resolve() != out_dir.resolve():
        raise RuntimeError("Protocol output directory mismatch")
    for path, expected in protocol["source_hashes"].items():
        if sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"Source changed after preregistration: {path}")
    for path, expected in protocol["artifact_hashes"].items():
        if sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    for name, expected in protocol["manifest_hashes"].items():
        if sha256(out_dir / name) != expected:
            raise RuntimeError(f"Pair manifest changed: {name}")
    return protocol, digest
