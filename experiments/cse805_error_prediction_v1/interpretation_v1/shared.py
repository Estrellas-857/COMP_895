"""Read-only access to the frozen experiment; new outputs stay beside this file."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from ..common import sha256, save_json
from ..protocol import read_csv

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROWS = read_csv(BASE / 'data/metadata_train.csv')
CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'train', 'motorcycle', 'bicycle')


def read(path):
    return json.loads(Path(path).read_text())


def group_rows(group):
    rows = [r for r in ROWS if r['recording_group'] == group]
    if not rows:
        raise ValueError('Unknown recording proxy: ' + group)
    return rows


def models(group):
    folder = BASE / 'data/models' / group
    selection = read(folder / 'selection.json')
    fitted = {}
    for method in ('lr3', 'lr3_class'):
        path = folder / (method + '.joblib')
        if sha256(path) != selection['model_hashes'][method]:
            raise RuntimeError('Frozen model hash mismatch')
        fitted[method] = joblib.load(path)
    return fitted, selection


def samples(group, kind='qa'):
    if kind not in ('qa', 'train'):
        raise ValueError('Only saved samples may be read')
    pieces = {k: [] for k in ('X', 'errors', 'weights', 'tie_keys')}
    for row in group_rows(group):
        path = BASE / 'data/features' / (row['image_id'] + '.npz')
        record = read(path.with_suffix('.json'))
        if sha256(path) != record['sha256']:
            raise RuntimeError('Frozen sample hash mismatch')
        with np.load(path, allow_pickle=False) as saved:
            for k in pieces:
                pieces[k].append(saved[kind + '_' + k])
    return {k: np.concatenate(v) for k, v in pieces.items()}


def table(name, rows):
    if not rows or Path(name).name != name:
        raise ValueError('A nonempty table and plain filename are required')
    path = OUT / 'tables' / name
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def dump(name, value):
    path = (OUT / name).resolve()
    if not path.is_relative_to(OUT):
        raise ValueError('Interpretation artifacts must stay isolated')
    save_json(path, value)


def verify_execution():
    complete = read(BASE / 'results/evaluation_complete.json')
    if complete['status'] != 'PASS' or complete['num_groups'] != 17 or complete['num_images'] != 1600:
        raise RuntimeError('Expected completed frozen experiment')
    for name, digest in complete['source_hashes'].items():
        if sha256(BASE / name) != digest:
            raise RuntimeError('Frozen execution source changed: ' + name)
    for name, digest in read(BASE / 'logs/completion.json')['artifact_hashes'].items():
        if sha256(BASE / name) != digest:
            raise RuntimeError('Frozen completion artifact changed: ' + name)
    return complete
