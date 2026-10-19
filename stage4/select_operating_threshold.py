"""Stage 4B only: GT-free exact 80% quantile, then separate fixed-point diagnostics.

Run --phase select before --phase evaluate. No validation data, model fitting,
threshold sweeps, or condition-specific operating points are supported.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

from stage4.reliability_pipeline import (
    ReliabilityPipeline, apply_reliability_threshold, load_frozen_temperature,
    TEMPERATURE_PATH, MODEL_NAME, MODEL_REVISION,
)

from stage1._layout import manifest_key, matches_frozen_hash

from stage1.dataset import ACDC_ROOT, acdc_path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "results/stage4_reliability/stage4b"
SPLITS = PROJECT / "results/stage3_calibration/stage3b/splits"
EXPECTED = {"fog": 200, "rain": 201, "snow": 200, "night": 197}
TARGET_COVERAGE = 0.80
PREFIX_BITS = 20
MAX_CANDIDATE_BYTES = 128 * 1024**2
QUANTILE_METHOD = ("Exact inverse empirical CDF (nearest rank ceil(4*N/5), equivalent to "
                   "NumPy quantile method=inverted_cdf); float64 external-memory radix "
                   "narrowing then exact order-statistic partition; no rounding or subsampling")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def protected(path):
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    relative = manifest_key(path, manifest)
    if not matches_frozen_hash(path, manifest[relative]):
        raise RuntimeError(f"Protected artifact changed: {path}")


def load_selection():
    """Read frozen split paths only. No image/GT content is opened here."""
    for path in (TEMPERATURE_PATH, SPLITS / "reliability_selection.csv", SPLITS / "calibration_fit.csv",
                 PROJECT / "stage4/reliability_pipeline.py"):
        protected(path)
    original_manifest = json.loads((SPLITS.parent / "split_manifest.json").read_text())
    for path in (SPLITS / "reliability_selection.csv", SPLITS / "calibration_fit.csv"):
        assert sha256(path) == original_manifest[str(path.relative_to(PROJECT))]
    selected = read_csv(SPLITS / "reliability_selection.csv")
    fit = read_csv(SPLITS / "calibration_fit.csv")
    assert len(selected) == 798 and Counter(r["condition"] for r in selected) == EXPECTED
    assert len({r["rgb_path"] for r in selected}) == len({r["gt_path"] for r in selected}) == 798
    assert not {r["rgb_path"] for r in selected} & {r["rgb_path"] for r in fit}
    assert not {r["gt_path"] for r in selected} & {r["gt_path"] for r in fit}
    assert not {r["sequence_id"] for r in selected} & {r["sequence_id"] for r in fit}
    for row in selected:
        condition, seq = row["condition"], row["sequence_id"]
        rgb, gt = acdc_path(row["rgb_path"]), acdc_path(row["gt_path"])
        assert row["subset"] == "reliability_selection"
        assert rgb.parts[-4:-1] == (condition, "train", seq)
        assert gt.parts[-4:-1] == (condition, "train", seq)
        assert rgb.parent.name == rgb.name.split("_frame_")[0] == seq
        assert gt.name == rgb.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        assert "val" not in rgb.parts and "val" not in gt.parts
        assert rgb.is_file() and gt.is_file()
    return selected


class DatasetAccessGuard:
    """Block all dataset opens except the phase's explicit allowlist.

    Python's file-open audit covers PIL/NumPy access in this implementation.
    Path existence checks and opaque result hashing are not GT content reads.
    """

    def __init__(self, rows, phase):
        self.phase = phase
        self.dataset = ACDC_ROOT
        key = "rgb_path" if phase == "select" else "gt_path"
        self.allowed = {acdc_path(row[key]) for row in rows}
        self.opened = set()
        self.blocked = []
        self.first_open_utc = None
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(self.dataset):
            if path not in self.allowed:
                self.blocked.append(str(path))
                raise RuntimeError(f"Dataset access blocked in phase {self.phase}: {path}")
            if self.first_open_utc is None:
                self.first_open_utc = now()
            self.opened.add(str(path))

    def report(self):
        return dict(phase=self.phase, opened_dataset_files=sorted(self.opened),
                    num_distinct_opened_files=len(self.opened), blocked_open_attempts=self.blocked,
                    first_dataset_open_utc=self.first_open_utc,
                    only_manifest_files_opened=set(self.opened) == {str(p) for p in self.allowed},
                    gt_opened=self.phase == "evaluate" and bool(self.opened),
                    validation_opened=False, calibration_fit_images_opened=False)


def float_prefix_hist(values, bits=PREFIX_BITS):
    """Lossless radix indexing, NOT a rounded-score quantile approximation."""
    values = np.asarray(values)
    assert values.dtype == np.float64 and np.isfinite(values).all()
    assert values.min() >= 0 and values.max() <= 1 and not np.signbit(values).any()
    keys = (values.reshape(-1).view(np.uint64) >> np.uint64(64 - bits)).astype(np.int64)
    return np.bincount(keys, minlength=1 << bits)


def exact_external_quantile(paths, total, prefix_hist, *, max_candidate_bytes=MAX_CANDIDATE_BYTES):
    """Select kth float64 value exactly, using bounded memory and sequential reads.

    Positive IEEE-754 float64 bit patterns have the same ordering as values.
    Prefix counts locate the order statistic's bucket. If its contents exceed
    the memory cap, refine the prefix by eight bits and rescan. Once it fits,
    partition the ORIGINAL float64 values. At 64 bits the bucket is an exact tie.
    No bins' edges or centers are used as the selected threshold.
    """
    assert int(prefix_hist.sum()) == total and total > 0
    rank = (4 * total + 4) // 5  # Exact integer ceil(0.80*N), one-based.
    cumulative = prefix_hist.cumsum(dtype=np.int64)
    prefix = int(np.searchsorted(cumulative, rank, side="left"))
    below = int(cumulative[prefix - 1]) if prefix else 0
    count = int(prefix_hist[prefix])
    bits = PREFIX_BITS
    trace = []
    while True:
        trace.append(dict(prefix_bits=bits, prefix=prefix, bucket_count=count,
                          smaller_value_count=below, local_rank=rank - below))
        if bits == 64:
            threshold = float(np.array([prefix], dtype=np.uint64).view(np.float64)[0])
            return threshold, rank, trace
        if count * 8 <= max_candidate_bytes:
            candidates = np.empty(count, dtype=np.float64)
            offset = 0
            for path in paths:
                data = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
                keep = (data.view(np.uint64) >> np.uint64(64 - bits)) == prefix
                chosen = data[keep]
                candidates[offset:offset + chosen.size] = chosen
                offset += chosen.size
            assert offset == count
            local = rank - below - 1
            candidates.partition(local)
            return float(candidates[local]), rank, trace
        advance = min(8, 64 - bits)
        refined = np.zeros(1 << advance, dtype=np.int64)
        for path in paths:
            data = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
            encoded = data.view(np.uint64)
            selected = encoded[(encoded >> np.uint64(64 - bits)) == prefix]
            keys = ((selected >> np.uint64(64 - bits - advance)) & ((1 << advance) - 1)).astype(np.int64)
            refined += np.bincount(keys, minlength=len(refined))
        assert int(refined.sum()) == count
        cumulative = refined.cumsum(dtype=np.int64)
        part = int(np.searchsorted(cumulative, rank - below, side="left"))
        below += int(cumulative[part - 1]) if part else 0
        count = int(refined[part])
        prefix = (prefix << advance) | part
        bits += advance


def phase_select():
    if (ROOT / "frozen_operating_point.json").exists() or (ROOT / "pixel_cache").exists():
        raise RuntimeError("Refusing to repeat or overwrite Stage 4B selection/cache")
    rows = load_selection()
    temperature, temp_sha = load_frozen_temperature()
    estimated_bytes = len(rows) * (1080 * 1920 * 9 + 256)
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < estimated_bytes + 2 * 1024**3:
        raise RuntimeError(f"Insufficient disk: projected cache {estimated_bytes}, free {free_bytes}")
    guard = DatasetAccessGuard(rows, "select")
    # The Stage 4A constructor requires a threshold; 1.0 is UNUSED here.
    # Only infer_logits/compute_uncertainty are called, so no placeholder masks
    # or coverages influence the empirical quantile.
    pipeline = ReliabilityPipeline(threshold=1.0, uncertainty_method="calibrated_msp",
                                   device="cuda", local_files_only=True)
    assert pipeline.temperature == temperature
    cache = ROOT / "pixel_cache"
    cache.mkdir()
    histogram = np.zeros(1 << PREFIX_BITS, dtype=np.int64)
    index = []
    total, changed = 0, 0
    started = time.perf_counter()
    print(f"PHASE A: {len(rows)} reserved RGBs; ALL pixels; no GT; T={temperature!r}", flush=True)
    print(f"Projected cache: {estimated_bytes / 1024**3:.3f} GiB; free={free_bytes / 1024**3:.1f} GiB", flush=True)
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            logits = pipeline.infer_logits(acdc_path(row["rgb_path"]))
            uncertainty, confidence = pipeline.compute_uncertainty(logits, "calibrated_msp")
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            scaled_prediction = (logits.to(torch.float64) / temperature).argmax(dim=1)[0].cpu().numpy()
            changed += int(np.count_nonzero(prediction != scaled_prediction))
            assert uncertainty.shape == prediction.shape == (1080, 1920)
            assert uncertainty.dtype == np.float64 and changed == 0
            histogram += float_prefix_hist(uncertainty)
            total += uncertainty.size
            folder = cache / row["condition"]
            folder.mkdir(exist_ok=True)
            stem = Path(row["rgb_path"]).stem
            u_path, p_path = folder / f"{stem}_uncertainty.npy", folder / f"{stem}_prediction.npy"
            np.save(u_path, uncertainty, allow_pickle=False)
            np.save(p_path, prediction, allow_pickle=False)
            index.append(dict(**row, height=prediction.shape[0], width=prediction.shape[1],
                              num_pixels=uncertainty.size,
                              uncertainty_path=str(u_path.relative_to(PROJECT)), uncertainty_sha256=sha256(u_path),
                              prediction_path=str(p_path.relative_to(PROJECT)), prediction_sha256=sha256(p_path)))
            del logits, uncertainty, confidence, prediction, scaled_prediction
            if number == 1 or number % 25 == 0 or number == len(rows):
                print(f"[{number}/{len(rows)}] {row['condition']}; {total:,} pixels; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    assert total == 798 * 1080 * 1920 and changed == 0
    assert guard.report()["only_manifest_files_opened"] and not guard.blocked
    write_csv(ROOT / "cache_index.csv", index)
    np.save(ROOT / "radix_prefix_counts.npy", histogram)
    paths = [PROJECT / row["uncertainty_path"] for row in index]
    threshold, rank, radix_trace = exact_external_quantile(paths, total, histogram)
    print(f"Exact quantile found: tau={threshold!r}; order-statistic rank={rank:,}", flush=True)
    # Independently verify the exact order statistic and tie-inclusive coverage.
    counts = {condition: dict(num_pixels=0, less_than_threshold=0, equal_to_threshold=0,
                             accepted_pixels=0, rejected_pixels=0) for condition in EXPECTED}
    for row, path in zip(index, paths):
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        c = counts[row["condition"]]
        c["num_pixels"] += values.size
        c["less_than_threshold"] += int(np.count_nonzero(values < threshold))
        c["equal_to_threshold"] += int(np.count_nonzero(values == threshold))
        accepted = int(np.count_nonzero(values <= threshold))
        c["accepted_pixels"] += accepted
        c["rejected_pixels"] += values.size - accepted
    pooled = {key: sum(c[key] for c in counts.values()) for key in next(iter(counts.values()))}
    assert pooled["less_than_threshold"] < rank <= pooled["accepted_pixels"]
    assert pooled["accepted_pixels"] == pooled["less_than_threshold"] + pooled["equal_to_threshold"]
    assert pooled["accepted_pixels"] + pooled["rejected_pixels"] == total
    # This artifact is exclusively created BEFORE any GT file can be opened.
    artifact = dict(uncertainty_method="calibrated_msp", model=MODEL_NAME, model_revision=MODEL_REVISION,
        temperature=temperature, temperature_artifact_sha256=temp_sha, target_coverage=TARGET_COVERAGE,
        threshold=threshold, acceptance_rule="uncertainty <= threshold", selection_dataset="ACDC train/reliability_selection only",
        split_manifest=str((SPLITS / "reliability_selection.csv").relative_to(PROJECT)),
        split_sha256=sha256(SPLITS / "reliability_selection.csv"),
        num_images=len(rows), num_pixels=total, images_by_condition=EXPECTED,
        quantile_method=QUANTILE_METHOD, quantile_rank_one_based=rank,
        quantile_discretization_error=0.0, random_pixel_subsampling=False,
        all_image_pixels_used=True, gt_used_for_selection=False,
        selection_all_pixel_coverage=pooled["accepted_pixels"] / total,
        all_pixel_selection_counts=pooled, condition_all_pixel_counts=counts,
        prediction_changes=changed, prediction_pixels_checked=total,
        created_utc=now(), stage4a_pipeline_sha256=sha256(PROJECT / "stage4/reliability_pipeline.py"),
        selection_code_sha256=sha256(Path(__file__)), cache_index_sha256=sha256(ROOT / "cache_index.csv"))
    artifact_path = ROOT / "frozen_operating_point.json"
    write_json(artifact_path, artifact)
    freeze = dict(artifact_sha256=sha256(artifact_path), frozen_utc=now(),
                  cache_index_sha256=sha256(ROOT / "cache_index.csv"), gt_files_opened_at_freeze=0)
    write_json(ROOT / "threshold_freeze_record.json", freeze)
    write_json(ROOT / "phase_a_access_audit.json", guard.report())
    write_json(ROOT / "quantile_search_audit.json", dict(radix_trace=radix_trace,
        candidate_memory_cap_bytes=MAX_CANDIDATE_BYTES, exact_order_statistic_verified=True,
        selection_duration_seconds=time.perf_counter()-started,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
    print(f"FROZEN BEFORE GT: {artifact_path}; SHA256={freeze['artifact_sha256']}", flush=True)
    print(f"All-pixel coverage={artifact['selection_all_pixel_coverage']:.12f}; prediction changes=0", flush=True)


def safe_ratio(numerator, denominator):
    """Undefined ratios are explicitly null/empty, never silently zero."""
    return numerator / denominator if denominator else None


def metrics_from_counts(condition, c, threshold):
    tp, fp, fn, tn = (c[k] for k in ("TP", "FP", "FN", "TN"))
    valid = tp + fp + fn + tn
    accepted, incorrect = fn + tn, tp + fn
    assert valid == c["valid_pixels"]
    assert c["all_accepted_pixels"] + c["all_rejected_pixels"] == c["all_pixels"]
    baseline, risk = safe_ratio(incorrect, valid), safe_ratio(fn, accepted)
    reduction = baseline - risk if baseline is not None and risk is not None else None
    result = dict(condition=condition, threshold=threshold, uncertainty_method="calibrated_msp", **c,
        accepted_valid_pixels=accepted, rejected_valid_pixels=tp+fp, incorrect_valid_pixels=incorrect,
        all_pixel_coverage=safe_ratio(c["all_accepted_pixels"], c["all_pixels"]),
        coverage=safe_ratio(accepted, valid), risk=risk, baseline_risk=baseline,
        risk_reduction_absolute=reduction,
        risk_reduction_relative=safe_ratio(reduction, baseline) if reduction is not None and baseline is not None else None,
        error_recall=safe_ratio(tp, incorrect), rejection_precision=safe_ratio(tp, tp+fp),
        incorrect_accepted_fraction=safe_ratio(fn, incorrect), correct_rejection_rate=safe_ratio(fp, fp+tn),
        rejection_rate=safe_ratio(tp+fp, valid))
    denominators = dict(valid_pixels=valid, accepted_valid_pixels=accepted,
                        incorrect_valid_pixels=incorrect, rejected_valid_pixels=tp+fp,
                        correct_valid_pixels=fp+tn, baseline_risk=baseline)
    result["zero_denominators"] = ";".join(k for k, v in denominators.items() if v == 0 or v is None)
    for key in ("coverage", "risk", "baseline_risk", "error_recall", "rejection_precision",
                "incorrect_accepted_fraction", "correct_rejection_rate", "rejection_rate"):
        assert result[key] is None or 0 <= result[key] <= 1
    return result


def preservation_check():
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    failures = [p for p, value in manifest.items() if not matches_frozen_hash(PROJECT / p, value)]
    if failures:
        raise RuntimeError(f"Preservation failed: {failures}")
    report = (f"PASS: all {len(manifest)} pre-existing protected files pass SHA-256 checks with the documented source relocation.\n"
              "Stages 1, 2, 3A–3D, and 4A, temperature, train split manifests, and prior sources match the documented relocation.\n"
              "Protected files were read as opaque bytes for hashing, not used as validation metrics or fitting data.\n")
    (ROOT / "preservation_check.txt").write_text(report)
    return report


def phase_evaluate():
    if (ROOT / "stage4b_summary.csv").exists():
        raise RuntimeError("Refusing to overwrite completed Stage 4B diagnostics")
    rows = load_selection()
    temperature, temp_sha = load_frozen_temperature()
    artifact_path = ROOT / "frozen_operating_point.json"
    artifact = json.loads(artifact_path.read_text())
    freeze = json.loads((ROOT / "threshold_freeze_record.json").read_text())
    assert sha256(artifact_path) == freeze["artifact_sha256"]
    assert artifact["temperature"] == temperature and artifact["temperature_artifact_sha256"] == temp_sha
    assert artifact["gt_used_for_selection"] is False and artifact["target_coverage"] == TARGET_COVERAGE
    assert artifact["uncertainty_method"] == "calibrated_msp" and artifact["num_images"] == len(rows)
    assert matches_frozen_hash(Path(__file__), artifact["selection_code_sha256"])
    assert sha256(ROOT / "cache_index.csv") == freeze["cache_index_sha256"] == artifact["cache_index_sha256"]
    index = read_csv(ROOT / "cache_index.csv")
    assert len(index) == len(rows)
    assert all(all(cached[key] == row[key] for key in row) for cached, row in zip(index, rows))
    phase_a_audit = json.loads((ROOT / "phase_a_access_audit.json").read_text())
    assert not phase_a_audit["gt_opened"] and not phase_a_audit["blocked_open_attempts"]
    assert phase_a_audit["num_distinct_opened_files"] == 798 and phase_a_audit["only_manifest_files_opened"]
    threshold = artifact["threshold"]
    guard = DatasetAccessGuard(rows, "evaluate")
    totals = {condition: dict(num_images=0, all_pixels=0, all_accepted_pixels=0, all_rejected_pixels=0,
                             valid_pixels=0, ignored_pixels=0, TP=0, FP=0, FN=0, TN=0) for condition in EXPECTED}
    started = time.perf_counter()
    print(f"PHASE B: fixed tau={threshold!r}; threshold SHA256={freeze['artifact_sha256']}", flush=True)
    for number, row in enumerate(index, 1):
        assert sha256(artifact_path) == freeze["artifact_sha256"]
        u_path, p_path = PROJECT / row["uncertainty_path"], PROJECT / row["prediction_path"]
        assert sha256(u_path) == row["uncertainty_sha256"] and sha256(p_path) == row["prediction_sha256"]
        uncertainty = np.load(u_path, allow_pickle=False)
        prediction = np.load(p_path, allow_pickle=False)
        before = hashlib.sha256(prediction.tobytes()).hexdigest()
        accepted = apply_reliability_threshold(uncertainty, threshold)
        with Image.open(acdc_path(row["gt_path"])) as source:
            gt = np.array(source)
        assert gt.shape == prediction.shape == uncertainty.shape == (int(row["height"]), int(row["width"]))
        valid = (gt >= 0) & (gt < 19)
        assert np.all(valid | (gt == 255))
        incorrect = prediction != gt
        c = totals[row["condition"]]
        c["num_images"] += 1
        c["all_pixels"] += gt.size
        c["all_accepted_pixels"] += int(accepted.sum())
        c["all_rejected_pixels"] += int((~accepted).sum())
        c["valid_pixels"] += int(valid.sum())
        c["ignored_pixels"] += int((~valid).sum())
        c["TP"] += int(np.count_nonzero(valid & incorrect & ~accepted))
        c["FP"] += int(np.count_nonzero(valid & ~incorrect & ~accepted))
        c["FN"] += int(np.count_nonzero(valid & incorrect & accepted))
        c["TN"] += int(np.count_nonzero(valid & ~incorrect & accepted))
        assert hashlib.sha256(prediction.tobytes()).hexdigest() == before
        if number == 1 or number % 100 == 0 or number == len(index):
            print(f"[{number}/{len(index)}] fixed-point diagnostics; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    assert guard.report()["only_manifest_files_opened"] and not guard.blocked
    assert freeze["frozen_utc"] < guard.first_open_utc
    pooled = {key: sum(c[key] for c in totals.values()) for key in next(iter(totals.values()))}
    assert pooled["all_pixels"] == artifact["num_pixels"]
    assert pooled["all_accepted_pixels"] == artifact["all_pixel_selection_counts"]["accepted_pixels"]
    for condition, c in totals.items():
        assert c["num_images"] == EXPECTED[condition]
        assert c["all_accepted_pixels"] == artifact["condition_all_pixel_counts"][condition]["accepted_pixels"]
        assert c["valid_pixels"] + c["ignored_pixels"] == c["all_pixels"]
    condition_metrics = [metrics_from_counts(condition, c, threshold) for condition, c in totals.items()]
    pooled_metrics = metrics_from_counts("pooled", pooled, threshold)
    write_csv(ROOT / "condition_metrics.csv", condition_metrics)
    write_csv(ROOT / "stage4b_summary.csv", [pooled_metrics])
    write_json(ROOT / "phase_b_access_audit.json", guard.report())
    assert sha256(artifact_path) == freeze["artifact_sha256"]
    write_json(ROOT / "post_evaluation_threshold_check.json", dict(
        status="PASS", before_sha256=freeze["artifact_sha256"], after_sha256=sha256(artifact_path),
        threshold=threshold, check_utc=now(), prediction_changes=artifact["prediction_changes"],
        cached_predictions_unchanged=True, threshold_modified_after_gt=False))
    preservation = preservation_check()
    audit_text = ("Stage 4B selection audit: PASS\n"
        f"Frozen reliability_selection split: {len(rows)} RGBs; {EXPECTED}.\n"
        "No overlap with calibration_fit image paths, GT paths, or sequence IDs. No validation paths used.\n"
        "Phase A allowlist permits only the 798 reserved RGBs; zero GT opens; all pixels included.\n"
        "Exact external-memory float64 order statistic, no rounding/subsampling or histogram approximation.\n"
        f"Rank: {artifact['quantile_rank_one_based']}; total image pixels: {artifact['num_pixels']}.\n"
        f"Threshold artifact frozen at {freeze['frozen_utc']}; first GT open at {guard.first_open_utc}.\n"
        f"Threshold SHA-256 before and after diagnostics: {freeze['artifact_sha256']}.\n"
        f"Prediction changes: 0 across {artifact['prediction_pixels_checked']} image pixels; cache retained unchanged.\n"
        "Phase B opens only reserved GTs, using Phase A uncertainty/prediction caches.\n"
        "Accepted + rejected = all pixels. TP+FP+FN+TN = valid pixels. All metric denominators checked.\n"
        "One global threshold; no condition-specific thresholds, risk sweep, model fitting, or temperature fitting.\n"
        "No validation evaluation; Stage 4C/5 not begun.\n")
    (ROOT / "selection_audit.txt").write_text(audit_text)
    summary = ["Stage 4B — frozen 80% coverage operating threshold", "",
        f"Method: calibrated_msp; frozen shared T={temperature!r}", f"Frozen shared tau={threshold!r}",
        f"Selection: {QUANTILE_METHOD}",
        f"All-pixel selection coverage={artifact['selection_all_pixel_coverage']:.12f} over {artifact['num_pixels']:,} pixels / 798 RGBs.",
        "GT was not used to choose the threshold or which pixels entered its distribution.",
        "Only after freezing was labelTrainIds GT used for diagnostics (classes 0..18; ignore 255).", "",
        "Valid-pixel diagnostics (fractions unless otherwise noted):", json.dumps(pooled_metrics, indent=2), "",
        "Per-condition diagnostics, same global tau:", json.dumps(condition_metrics, indent=2), "",
        "Definitions: coverage=(TN+FN)/N; risk=FN/(TN+FN); baseline_risk=(TP+FN)/N.",
        "Risk reduction absolute=baseline-risk; relative=(baseline-risk)/baseline.",
        "TP=incorrect rejected; FP=correct rejected; FN=incorrect accepted; TN=correct accepted.",
        "Error recall=TP/(TP+FN); rejection precision=TP/(TP+FP); incorrect accepted fraction=FN/(TP+FN).",
        "Correct rejection rate=FP/(FP+TN); rejection rate=(TP+FP)/N.",
        "At 100% coverage risk equals the baseline pixel error rate by definition.",
        "Undefined ratios would be empty in CSV; zero_denominators lists affected denominators.", "",
        "The 80% target applies to ALL image pixels. Diagnostic coverage excludes ignore=255, so may differ.",
        "Ties at the exact order statistic are all accepted (<=), potentially giving slightly over 80%.",
        "Radix histograms are lossless order-statistic search indexes; the threshold is an original float64 value.",
        "Stage 4A inference/precision reused unchanged. Constructor threshold=1.0 was unused during selection;",
        "only infer_logits/compute_uncertainty ran before tau was selected. No placeholder mask was analyzed.",
        "Uncertainty and canonical prediction caches avoid repeating model inference in Phase B.",
        "Optional raw-MSP comparison was not performed; the primary experiment remains focused.", "",
        audit_text, preservation,
        "These are descriptive results on RESERVED TRAIN SELECTION DATA only. Rejected predictions are abstained, not corrected.",
        "No final validation performance, generalization, or real-world safety claim is established.",
        "STOP: no Stage 4C or Stage 5."]
    (ROOT / "stage4b_summary.txt").write_text("\n".join(summary) + "\n")
    print(json.dumps(pooled_metrics, indent=2), flush=True)
    for row in condition_metrics:
        print(f"{row['condition']}: coverage={row['coverage']:.9f}, baseline risk={row['baseline_risk']:.9f}, risk={row['risk']:.9f}", flush=True)
    print(preservation, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("select", "evaluate"))
    args = parser.parse_args()
    os.chdir(PROJECT)
    if args.phase == "select":
        phase_select()
    else:
        phase_evaluate()
