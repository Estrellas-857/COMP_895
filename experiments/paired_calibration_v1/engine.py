"""Immutable native-logit caching and matched-population scalar calibration.

This module never invokes the historical experimental drivers. The numerical
NLL terms and safeguarded scalar solver are reused from Stage 3B.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from stage1.dataset import PROJECT_ROOT, acdc_path
from stage3.fit_temperature import pixel_terms, solve_temperature


MODEL = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
REVISION = "7a91500a7086b805eeb868719ea5542a3e0bdeb3"
NATIVE_SHAPE = (1, 19, 128, 256)
IMAGE_SHAPE = (1080, 1920)
SCHEMA = 1


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write_once(path, value):
    """Atomically publish JSON without replacing a completed artifact."""
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise RuntimeError(f"Refusing to replace a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as f:
        temporary = Path(f.name)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)  # Fails rather than overwrites if another writer won.
    finally:
        temporary.unlink(missing_ok=True)


def _cache_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _device(value):
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA execution are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; fix execution access before inference")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device


def _resources(started, device):
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda" else None,
        "device": str(device),
    }


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _load_gt(row):
    data = acdc_path(row["gt_path"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if row.get("gt_sha256") and row["gt_sha256"] != digest:
        raise RuntimeError(f"GT changed: {row['gt_path']}")
    with Image.open(io.BytesIO(data)) as image:
        target = np.array(image)
    if target.shape != IMAGE_SHAPE or not np.isin(target, list(range(19)) + [255]).all():
        raise ValueError(f"Unexpected semantic labels: {row['gt_path']}")
    count = int(np.count_nonzero(target < 19))
    if not count:
        raise ValueError("No valid semantic pixels")
    if row.get("num_valid_pixels") is not None and int(row["num_valid_pixels"]) != count:
        raise RuntimeError(f"Valid-pixel count changed: {row['gt_path']}")
    return target, count, digest


def _load_native(row):
    path = _cache_path(row["cache_path"])
    if _sha(path) != row["cache_sha256"]:
        raise RuntimeError(f"Native-logit cache changed: {path}")
    native = np.load(path, allow_pickle=False)
    if native.shape != NATIVE_SHAPE or native.dtype != np.float32 or not np.isfinite(native).all():
        raise ValueError(f"Invalid native logits: {path}")
    return native


def _row_identity(row):
    return {key: row.get(key) for key in (
        "split", "domain", "pair_id", "sequence_id", "rgb_path", "gt_path",
        "rgb_sha256", "gt_sha256", "cache_path", "cache_sha256", "num_valid_pixels")}


def _validate_records(records):
    if not records:
        raise ValueError("An empty image population is not allowed")
    seen = set()
    for row in records:
        if row.get("split") not in {"fit", "val"} or row.get("domain") not in {"normal", "night"}:
            raise ValueError("Records require split fit/val and domain normal/night")
        for field in ("pair_id", "sequence_id", "rgb_path", "gt_path"):
            if not row.get(field):
                raise ValueError(f"Missing image field: {field}")
        key = row["split"], row["domain"], row["pair_id"]
        if key in seen:
            raise ValueError(f"Duplicate image record: {key}")
        seen.add(key)
        expected_split = "train" if row["split"] == "fit" else "val"
        suffix = "_ref" if row["domain"] == "normal" else ""
        for field in ("rgb_path", "gt_path"):
            parts = acdc_path(row[field]).parts
            if parts[-3] != expected_split + suffix or parts[-4] != "night":
                raise ValueError(f"Record points outside its declared Night split/domain: {row[field]}")


def cache_images(records, out_dir, device, protocol_sha256):
    """Verify existing Night caches and generate each missing native cache once.

    A completed image's sidecar binds protocol, input hashes, and output hash.
    It is published before the cache: interrupted publication can be resumed by
    recomputing and requiring the same native-logit hash. Indexes are separate
    for fit and validation. Existing artifacts are never overwritten.
    """
    records = [dict(row) for row in records]
    _validate_records(records)
    if not protocol_sha256:
        raise ValueError("A frozen protocol hash is required")
    device = _device(device)
    out_dir = Path(out_dir).resolve()
    started = time.perf_counter()
    _reset_peak(device)
    model = processor = None
    completed = []
    generated = reused = 0
    for index, source in enumerate(records, 1):
        row = dict(source)
        rgb_path = acdc_path(row["rgb_path"])
        rgb_sha = _sha(rgb_path)
        if row.get("rgb_sha256") and rgb_sha != row["rgb_sha256"]:
            raise RuntimeError(f"RGB changed: {rgb_path}")
        _, count, gt_sha = _load_gt(row)
        row.update(rgb_sha256=rgb_sha, gt_sha256=gt_sha, num_valid_pixels=count)
        if row.get("cache_path"):
            if not row.get("cache_sha256"):
                raise ValueError("An existing cache needs its frozen hash")
            existing_path = _cache_path(row["cache_path"]).resolve()
            if (row.get("cache_origin") == "paired_calibration_v1"
                    or existing_path.is_relative_to(out_dir / "logits")):
                sidecar = existing_path.with_suffix(".json")
                if not sidecar.is_file():
                    raise RuntimeError(f"Generated cache lacks provenance: {existing_path}")
                prior = _read(sidecar)
                identity = {k: row[k] for k in ("split", "domain", "pair_id", "sequence_id",
                                                "rgb_path", "gt_path", "rgb_sha256", "gt_sha256")}
                if (prior.get("protocol_sha256") != protocol_sha256
                        or prior.get("input") != identity
                        or prior.get("cache_sha256") != row["cache_sha256"]
                        or prior.get("cache_path") != str(existing_path)
                        or prior.get("model_revision") != REVISION):
                    raise RuntimeError(f"Generated cache provenance mismatch: {existing_path}")
            _load_native(row)
            row["cache_path"] = str(existing_path)
            row["cache_origin"] = row.get("cache_origin", "frozen_stage3b")
            reused += 1
        else:
            key = _digest([row["split"], row["domain"], row["pair_id"]])[:16]
            destination = out_dir / "logits" / row["split"] / row["domain"] / f"{key}_{rgb_path.stem}.npy"
            sidecar = destination.with_suffix(".json")
            identity = {
                "schema_version": SCHEMA, "protocol_sha256": protocol_sha256,
                "model": MODEL, "model_revision": REVISION,
                "input": {k: row[k] for k in ("split", "domain", "pair_id", "sequence_id",
                                                "rgb_path", "gt_path", "rgb_sha256", "gt_sha256")},
                "cache_path": str(destination), "num_valid_pixels": count,
                "cache_dtype": "float32", "cache_shape": list(NATIVE_SHAPE),
            }
            prior = _read(sidecar) if sidecar.exists() else None
            if prior and any(prior.get(k) != v for k, v in identity.items()):
                raise RuntimeError(f"Cache provenance does not match this run: {sidecar}")
            if destination.exists() and not prior:
                raise RuntimeError(f"Unprovenanced cache must not be adopted: {destination}")
            if destination.exists():
                row.update(cache_path=str(destination), cache_sha256=prior["cache_sha256"])
                _load_native(row)
                reused += 1
            else:
                if model is None:
                    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
                    kwargs = dict(revision=REVISION, local_files_only=True)
                    processor = AutoImageProcessor.from_pretrained(MODEL, **kwargs)
                    model = SegformerForSemanticSegmentation.from_pretrained(MODEL, **kwargs)
                    if model.config._commit_hash != REVISION or type(processor).__name__ != "SegformerImageProcessor":
                        raise RuntimeError("Model revision/processor does not match the frozen baseline")
                    model.to(device=device, dtype=torch.float32).eval()
                    model.requires_grad_(False)
                with Image.open(rgb_path) as image:
                    if image.size != IMAGE_SHAPE[::-1]:
                        raise ValueError(f"Unexpected RGB shape: {rgb_path}")
                    inputs = processor(images=image.convert("RGB"), return_tensors="pt",
                                       size={"height": 512, "width": 1024})
                with torch.inference_mode():
                    native = model(**{k: v.to(device) for k, v in inputs.items()}).logits.cpu().numpy()
                if native.shape != NATIVE_SHAPE or native.dtype != np.float32 or not np.isfinite(native).all():
                    raise RuntimeError("Unexpected generated native logits")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".pending-", delete=False) as f:
                    temporary = Path(f.name)
                    np.save(f, native, allow_pickle=False)
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    cache_sha = _sha(temporary)
                    metadata = dict(identity, cache_sha256=cache_sha)
                    _write_once(sidecar, metadata)
                    os.link(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                row.update(cache_path=str(destination), cache_sha256=cache_sha)
                generated += 1
                del native, inputs
            row["cache_origin"] = "paired_calibration_v1"
            row["cache_sidecar"] = str(sidecar)
        completed.append(row)
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"Cache {index}/{len(records)}: {row['split']}/{row['domain']}; generated={generated}, reused={reused}", flush=True)
    for split in sorted({row["split"] for row in completed}):
        subset = [row for row in completed if row["split"] == split]
        _write_once(out_dir / f"cache_index_{split}.json", {
            "schema_version": SCHEMA, "protocol_sha256": protocol_sha256,
            "model": MODEL, "model_revision": REVISION, "rows": subset,
            "rows_sha256": _digest([_row_identity(row) for row in subset]),
        })
    # Invocation-specific resource reports do not change completed indexes.
    run_id = time.time_ns()
    _write_once(out_dir / "resources" / f"cache_{run_id}.json", {
        "protocol_sha256": protocol_sha256, "num_images": len(completed),
        "generated_images": generated, "reused_images": reused, **_resources(started, device),
    })
    return completed


def validate_fit_population(rows, expected_pairs=101):
    """Require the same scene population in both domains, without validation."""
    groups = {domain: [] for domain in ("normal", "night")}
    for row in rows:
        if row.get("split") != "fit" or row.get("domain") not in groups:
            raise ValueError("Only declared fitting rows can enter temperature optimization")
        groups[row["domain"]].append(row)
    ids = {}
    for domain, records in groups.items():
        ids[domain] = {row["pair_id"] for row in records}
        if len(records) != expected_pairs or len(ids[domain]) != expected_pairs:
            raise ValueError(f"Expected {expected_pairs} unique fitting pairs in {domain}")
    if ids["normal"] != ids["night"]:
        raise ValueError("Normal and Night fitting scene populations are not matched")
    normal = {row["pair_id"]: row for row in groups["normal"]}
    for row in groups["night"]:
        if normal[row["pair_id"]]["sequence_id"] != row["sequence_id"]:
            raise ValueError("A paired scene has inconsistent sequence identifiers")
    return {domain: sorted(records, key=lambda row: row["pair_id"]) for domain, records in groups.items()}


def combine_domain_terms(sums, counts, weights):
    """Weight domain means, never concatenate populations for the Mix loss."""
    if not weights or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("Domain weights must sum to one")
    combined = np.zeros(4, dtype=np.float64)
    for domain, weight in weights.items():
        if not math.isfinite(weight) or weight <= 0 or counts.get(domain, 0) <= 0:
            raise ValueError("Each active domain requires positive weight and valid pixels")
        combined += weight * np.asarray(sums[domain], dtype=np.float64) / counts[domain]
    return combined


class _Objective:
    def __init__(self, rows, weights, device, directory, protocol_sha256):
        self.rows = [row for row in rows if row["domain"] in weights]
        self.weights = weights
        self.device = device
        self.directory = Path(directory)
        self.protocol = protocol_sha256
        self.rows_digest = _digest([_row_identity(row) for row in self.rows])
        self.trace = []

    @torch.inference_mode()
    def __call__(self, beta, phase="optimization", check_predictions=False):
        beta = float(beta)
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("Inverse temperature must be positive and finite")
        identity = dict(schema_version=SCHEMA, protocol_sha256=self.protocol,
                        rows_sha256=self.rows_digest, domain_weights=self.weights,
                        inverse_temperature=beta, phase=phase, check_predictions=check_predictions)
        path = self.directory / f"{_digest(identity)}.json"
        if path.exists():
            record = _read(path)
            if record["identity"] != identity:
                raise RuntimeError("Stored objective provenance mismatch")
            self.trace.append(str(path))
            return record["result"]
        started = time.perf_counter()
        sums = {domain: np.zeros(4, dtype=np.float64) for domain in self.weights}
        counts = {domain: 0 for domain in self.weights}
        images = {domain: 0 for domain in self.weights}
        changes = 0
        for row in self.rows:
            native = _load_native(row)
            gt, count, _ = _load_gt(row)
            logits32 = F.interpolate(torch.from_numpy(native).to(self.device), size=IMAGE_SHAPE,
                                     mode="bilinear", align_corners=False)
            logits64 = logits32.to(torch.float64)
            target = torch.as_tensor(gt.astype(np.int64), device=self.device)
            valid = (target >= 0) & (target < 19)
            terms = np.asarray(pixel_terms(logits64, target, valid, 1.0 / beta))
            if not np.isfinite(terms).all() or terms[0] < 0 or terms[2] < 0:
                raise RuntimeError("Nonfinite NLL derivatives or invalid curvature")
            domain = row["domain"]
            sums[domain] += terms
            counts[domain] += count
            images[domain] += 1
            if check_predictions:
                before = logits32.argmax(dim=1)[0]
                after = (logits64 / (1.0 / beta)).argmax(dim=1)[0]
                changes += int(torch.count_nonzero((before != after) & valid).item())
                del before, after
            del native, gt, logits32, logits64, target, valid
        nll, gradient, hessian, gradient0 = combine_domain_terms(sums, counts, self.weights)
        result = dict(nll=float(nll), gradient=float(gradient), hessian=float(hessian),
                      gradient0=float(gradient0), beta=beta, num_valid_pixels=sum(counts.values()),
                      prediction_changes=changes, by_domain={
                          domain: dict(num_images=images[domain], num_valid_pixels=counts[domain],
                                       nll=float(sums[domain][0] / counts[domain]))
                          for domain in self.weights})
        _write_once(path, dict(identity=identity, result=result, resources=_resources(started, self.device)))
        self.trace.append(str(path))
        print(f"Fit {self.directory.name}: T={1/beta:.9f}, NLL={nll:.9f}, gradient={gradient:+.3e}", flush=True)
        return result


def fit_temperatures(rows, out_dir, device, protocol_sha256):
    """Fit NightPair, NormalPair, and prespecified equal-domain Mix temperatures."""
    rows = [dict(row) for row in rows]
    _validate_records(rows)
    groups = validate_fit_population(rows)
    if not protocol_sha256:
        raise ValueError("A frozen protocol hash is required")
    # Revalidate immutable cache/GT identities even when resuming completed fits.
    for row in rows:
        _load_native(row)
        _load_gt(row)
    ordered = groups["normal"] + groups["night"]
    device = _device(device)
    out_dir = Path(out_dir).resolve() / "temperatures"
    specifications = {"NightPair": {"night": 1.0}, "NormalPair": {"normal": 1.0},
                      "Mix": {"normal": 0.5, "night": 0.5}}
    results = {}
    for name, weights in specifications.items():
        objective = _Objective(ordered, weights, device, out_dir / f"{name}_evaluations", protocol_sha256)
        result_path = out_dir / f"{name}.json"
        identity = dict(protocol_sha256=protocol_sha256, rows_sha256=objective.rows_digest,
                        domain_weights=weights, name=name, model=MODEL, model_revision=REVISION)
        if result_path.exists():
            result = _read(result_path)
            if any(result.get(k) != v for k, v in identity.items()):
                raise RuntimeError(f"Fitted-temperature provenance mismatch: {result_path}")
            if _sha(result["trace_path"]) != result["trace_sha256"]:
                raise RuntimeError("A completed fit's trace changed")
            results[name] = result
            continue
        started = time.perf_counter()
        _reset_peak(device)
        raw, optimum, iterations, reason = solve_temperature(objective)
        final = objective(optimum["beta"], phase="final_verification", check_predictions=True)
        if final["prediction_changes"] or final["nll"] > raw["nll"] + 1e-10:
            raise RuntimeError("Fit failed NLL/prediction invariance checks")
        if not math.isclose(final["nll"], optimum["nll"], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Final NLL does not reproduce the optimizer result")
        trace_path = out_dir / f"{name}_trace.json"
        trace = [{"path": path, "sha256": _sha(path), **_read(path)} for path in objective.trace]
        _write_once(trace_path, dict(identity=identity, evaluations=trace))
        result = dict(identity, schema_version=SCHEMA, temperature=1.0 / optimum["beta"],
                      raw_nll=raw["nll"], calibrated_nll=final["nll"],
                      raw_by_domain=raw["by_domain"], calibrated_by_domain=final["by_domain"],
                      num_pairs=101, num_images=len(objective.rows), num_valid_pixels=final["num_valid_pixels"],
                      optimizer="Stage3B safeguarded Newton in beta=1/T",
                      optimizer_iterations=iterations, objective_evaluations=len(objective.trace),
                      converged=True, convergence_reason=reason, final_gradient_beta=final["gradient"],
                      prediction_changes=final["prediction_changes"],
                      prediction_pixels_checked=final["num_valid_pixels"], validation_used=False,
                      precision="float32 native logits/interpolation; float64 NLL/derivatives/reductions",
                      trace_path=str(trace_path), trace_sha256=_sha(trace_path),
                      resources=_resources(started, device))
        _write_once(result_path, result)
        results[name] = result
    return results
