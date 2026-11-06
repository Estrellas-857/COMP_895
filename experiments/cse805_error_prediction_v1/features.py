"""Frozen-backbone features; GT is read only by supervision/evaluation helpers."""
import io
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from experiments.paired_calibration_v1.metrics import sample_indices
from stage4.reliability_pipeline import MODEL_NAME, MODEL_REVISION
from .common import ROOT, sha256, save_json, event

NATIVE_SHAPE = (1, 19, 128, 256)
IMAGE_SHAPE = (1080, 1920)


def configure_torch(device):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; do not silently change numerical backend')
    torch.backends.cudnn.benchmark = False


def load_target(row):
    data = Path(row['gt_path']).read_bytes()
    import hashlib
    if hashlib.sha256(data).hexdigest() != row['gt_sha256']:
        raise RuntimeError('GT bytes changed: ' + row['image_id'])
    with Image.open(io.BytesIO(data)) as source:
        target = np.array(source)
    if target.shape != IMAGE_SHAPE or not np.isin(target, list(range(19)) + [255]).all():
        raise ValueError('GT resolution/trainId mismatch: ' + row['image_id'])
    if not np.any(target < 19):
        raise ValueError('Image has no valid semantic labels')
    return target


class FrozenBackbone:
    def __init__(self, device):
        self.device = device
        kwargs = dict(revision=MODEL_REVISION, local_files_only=True)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME, **kwargs)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, **kwargs)
        if self.model.config._commit_hash != MODEL_REVISION:
            raise RuntimeError('Unpinned model revision')
        self.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.inference_mode()
    def native(self, row):
        if sha256(row['rgb_path']) != row['rgb_sha256']:
            raise RuntimeError('RGB bytes changed')
        with Image.open(row['rgb_path']) as source:
            image = source.convert('RGB')
        if image.size != IMAGE_SHAPE[::-1]:
            raise ValueError('RGB resolution mismatch')
        inputs = self.processor(images=image, return_tensors='pt', size={'height': 512, 'width': 1024})
        logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
        return logits.cpu().numpy()


def validate_native(native):
    if native.shape != NATIVE_SHAPE or native.dtype != np.float32 or not np.isfinite(native).all():
        raise ValueError('Native cache must be finite float32 1x19x128x256')


def native_cache(row, device, backbone=None):
    """Load the original cache or create only its missing frozen forward pass."""
    original = row.get('native_cache_path', '')
    if original:
        path = Path(original)
        digest = sha256(path)
        if digest != row['native_cache_sha256']:
            raise RuntimeError('Original native cache changed')
        native = np.load(path, allow_pickle=False)
        validate_native(native)
        return native, dict(path=str(path), sha256=digest, origin='reused'), backbone
    path = ROOT / 'data/native_logits' / (row['image_id'] + '.npy')
    record_path = path.with_suffix('.json')
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        expected = dict(rgb_sha256=row['rgb_sha256'], model=MODEL_NAME,
                        revision=MODEL_REVISION, device=device, dtype='float32', deterministic=True)
        if any(record.get(k) != v for k, v in expected.items()) or sha256(path) != record['sha256']:
            raise RuntimeError('New native cache provenance mismatch')
        native = np.load(path, allow_pickle=False)
        validate_native(native)
        return native, record, backbone
    if path.exists() or record_path.exists():
        raise RuntimeError('Incomplete native cache requires review: ' + str(path))
    if backbone is None:
        backbone = FrozenBackbone(device)
    started = time.perf_counter()
    native = backbone.native(row)
    validate_native(native)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.save(handle, native, allow_pickle=False)
    record = dict(path=str(path), sha256=sha256(path), origin='new_frozen_forward',
                  rgb_sha256=row['rgb_sha256'], model=MODEL_NAME, revision=MODEL_REVISION,
                  device=device, dtype='float32', deterministic=True,
                  elapsed_seconds=time.perf_counter() - started)
    save_json(record_path, record)
    return native, record, backbone


@torch.inference_mode()
def feature_block(logits):
    """No GT argument: raw logit features at their already aligned resolution."""
    if logits.ndim != 4 or logits.shape[:2] != (1, 19) or logits.dtype != torch.float32:
        raise ValueError('Expected float32 aligned logits')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits')
    probabilities = logits.softmax(dim=1)
    normalization_error = float((probabilities.sum(dim=1) - 1).abs().max().item())
    if normalization_error > 2e-6:
        raise RuntimeError('Softmax normalization failed')
    top = probabilities.topk(2, dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1) / math.log(19)
    prediction = logits.argmax(dim=1)
    values = torch.stack((top[:, 0], entropy.clamp(0, 1), top[:, 0] - top[:, 1], prediction.float()), dim=-1)
    return values.reshape(-1, 4).cpu().numpy(), normalization_error


@torch.inference_mode()
def iter_feature_blocks(native, device, chunk_rows=128):
    validate_native(native)
    logits = F.interpolate(torch.from_numpy(native).to(device), size=IMAGE_SHAPE,
                           mode='bilinear', align_corners=False)
    for first in range(0, IMAGE_SHAPE[0], chunk_rows):
        last = min(first + chunk_rows, IMAGE_SHAPE[0])
        X, normalization_error = feature_block(logits[:, :, first:last])
        yield first * IMAGE_SHAPE[1], last * IMAGE_SHAPE[1], X, normalization_error


def sample_image(row, native, device, seed=805, training_pixels=512, qa_pixels=8192):
    target = load_target(row)
    selections = {
        'train': sample_indices(target, row['image_id'], seed=seed, n=training_pixels),
        'qa': sample_indices(target, row['image_id'], seed=seed + 1, n=qa_pixels),
    }
    for sample in selections.values():
        sample['X'] = np.empty((len(sample['indices']), 4), dtype=np.float32)
    max_normalization_error, num_errors, historical_changes = 0., 0, 0
    old_prediction = None
    if row.get('prediction_cache_path'):
        path = row['prediction_cache_path']
        if sha256(path) != row['prediction_cache_sha256']:
            raise RuntimeError('Historical prediction cache changed')
        old_prediction = np.load(path, mmap_mode='r', allow_pickle=False).ravel()
    for first, last, X, error in iter_feature_blocks(native, device):
        max_normalization_error = max(max_normalization_error, error)
        gt = target.ravel()[first:last]
        valid = gt < 19
        num_errors += int(np.count_nonzero((X[:, 3] != gt) & valid))
        if old_prediction is not None:
            historical_changes += int(np.count_nonzero(X[:, 3] != old_prediction[first:last]))
        for sample in selections.values():
            lo, hi = np.searchsorted(sample['indices'], [first, last])
            sample['X'][lo:hi] = X[sample['indices'][lo:hi] - first]
    for sample in selections.values():
        sample['errors'] = (sample['X'][:, 3] != target.ravel()[sample['indices']]).astype(np.uint8)
    summary = dict(num_valid=int(np.count_nonzero(target < 19)), num_errors=num_errors,
                   num_ignored=int(np.count_nonzero(target == 255)),
                   probability_normalization_max_error=max_normalization_error,
                   historical_prediction_changes=historical_changes if old_prediction is not None else None,
                   prediction_definition='argmax of float32 full-resolution raw logits',
                   historical_comparison='all image pixels; new canonical predictions shared by all four methods')
    return selections, summary


def extract(rows, device, output, protocol_sha256):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure_torch(device)
    feature_source_sha256 = sha256(__file__)
    backbone, records = None, []
    started = time.perf_counter()
    for number, row in enumerate(rows, 1):
        path = output / (row['image_id'] + '.npz')
        record_path = path.with_suffix('.json')
        if path.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if (record['protocol_sha256'] != protocol_sha256 or record['sha256'] != sha256(path)
                    or record.get('device') != device or record['gt_sha256'] != row['gt_sha256']
                    or record['rgb_sha256'] != row['rgb_sha256']
                    or record.get('feature_source_sha256') != feature_source_sha256):
                raise RuntimeError('Feature sample provenance mismatch')
        else:
            if path.exists() or record_path.exists():
                raise RuntimeError('Partial feature artifact requires review')
            before = time.perf_counter()
            native, cache, backbone = native_cache(row, device, backbone)
            samples, summary = sample_image(row, native, device)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                np.savez_compressed(handle, **{f'{kind}_{key}': value for kind, sample in samples.items()
                    for key, value in sample.items() if key != 'num_valid'})
            record = dict(image_id=row['image_id'], condition=row['condition'], recording_group=row['recording_group'],
                          protocol_sha256=protocol_sha256, path=str(path), sha256=sha256(path),
                          feature_source_sha256=feature_source_sha256,
                          native_cache=cache, gt_sha256=row['gt_sha256'], rgb_sha256=row['rgb_sha256'],
                          device=device, elapsed_seconds=time.perf_counter() - before, **summary)
            save_json(record_path, record)
        records.append(record)
        if number == 1 or number % 25 == 0 or number == len(rows):
            print(f'Feature extraction {number}/{len(rows)}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
            event('feature_progress', completed=number, total=len(rows), elapsed_seconds=time.perf_counter()-started)
    save_json(output / 'manifest.json', dict(protocol_sha256=protocol_sha256, records=records,
              elapsed_seconds=time.perf_counter() - started, device=device))
    return records
