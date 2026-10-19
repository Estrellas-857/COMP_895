"""Verify historical source hashes after the documented stage-folder migration.

Experiment records remain byte-for-byte unchanged. Only explicitly recorded
old source paths/hashes can match the corresponding reviewed new source hash.
Data, configuration, and unknown source files always require an exact match.
"""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_PATH = PROJECT_ROOT / "stage1/source_layout.json"


def _relative(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return None


def _source_record(relative):
    if relative is None:
        return None, None
    records = json.loads(LAYOUT_PATH.read_text())["files"]
    if relative in records:
        return relative, records[relative]
    for original, record in records.items():
        if record["path"] == relative:
            return original, record
    return None, None


def manifest_key(path, manifest):
    """Find a current or historical source path in an existing manifest."""
    relative = _relative(path)
    if relative in manifest:
        return relative
    original, _ = _source_record(relative)
    if original in manifest:
        return original
    raise KeyError(f"No protected-file entry for {path}")


def matches_frozen_hash(path, expected):
    """Check exact bytes, or an explicitly recorded source-only migration."""
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = PROJECT_ROOT / absolute
    if absolute.is_file():
        with absolute.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() == expected:
                return True
    relative = _relative(path)
    # Never translate data/configuration hashes or arbitrary same-name files.
    if relative is None or not relative.endswith(".py"):
        return False
    _, record = _source_record(relative)
    if record is None or expected != record["original_sha256"]:
        return False
    current = PROJECT_ROOT / record["path"]
    if not current.is_file():
        return False
    with current.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"]
