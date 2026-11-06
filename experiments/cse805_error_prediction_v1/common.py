"""Paths, atomic artifacts and an append-only log for this isolated extension."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('New artifacts must remain inside the CSE805 extension')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def event(phase, **details):
    path = ROOT / 'logs' / 'experiment.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(utc=datetime.now(timezone.utc).isoformat(), phase=phase, **details)
    with path.open('a') as handle:
        handle.write(json.dumps(json_safe(record), allow_nan=False) + '\n')


def preservation_snapshot():
    roots = [PROJECT / f'stage{i}' for i in range(1, 6)]
    roots += [PROJECT / 'results', PROJECT / 'experiments/paired_calibration_v1']
    result = {}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                result[str(path.relative_to(PROJECT))] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return result
