"""Class composition at frozen Night/Fog curve cutoffs; cached logits only."""
import argparse
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .. import features
from ..evaluation import CONFIGS, HISTOGRAM_BINS, histogram_indices, predict_scores
from .shared import BASE, OUT, ROWS, CLASSES, dump, models, read, sha256, table, verify_execution

TARGETS = (0.01, 0.05, 0.1, 0.2, 0.5, 0.8)


def confusion_at_cutoffs(pair_ids, score_bins, cutoff_bins):
    """One bincount yields cumulative GT-by-prediction counts at every cutoff."""
    pair_ids, score_bins, cutoffs = map(np.asarray, (pair_ids, score_bins, cutoff_bins))
    if (pair_ids.ndim != 1 or score_bins.shape != pair_ids.shape or cutoffs.ndim != 1
            or not cutoffs.size or not all(np.issubdtype(a.dtype, np.integer) for a in (pair_ids, score_bins, cutoffs))
            or np.any((pair_ids < 0) | (pair_ids >= 361)) or np.any(score_bins < 0)
            or np.any(cutoffs < 0) or np.any(np.diff(cutoffs) < 0)):
        raise ValueError("Expected valid semantic pair IDs, score bins and sorted integer cutoffs")
    bucket = np.searchsorted(cutoffs, score_bins, side="left")
    counts = np.bincount(bucket * 361 + pair_ids, minlength=(len(cutoffs) + 1) * 361)
    return np.cumsum(counts.reshape(len(cutoffs) + 1, 19, 19), axis=0)[:-1]


def operating_points(group, result, inputs):
    points = {}
    for method in CONFIGS:
        path = BASE / "results/folds" / group / (method + "_curve.npz")
        inputs[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as curve:
            if curve["coverage"].ndim != 1 or not curve["coverage"].size:
                raise ValueError("Frozen curve must contain attainable coverage points")
            points[method] = []
            for target in TARGETS:
                index = int(np.argmin(np.abs(curve["coverage"] - target)))
                point = dict(requested_coverage=target, curve_index=index,
                    cutoff_bin=int(curve["bin_index"][index]),
                    threshold_upper=float(curve["threshold_upper"][index]),
                    actual_coverage=float(curve["coverage"][index]),
                    accepted_pixels=int(curve["accepted"][index]),
                    accepted_errors=int(curve["errors"][index]), risk=float(curve["risk"][index]))
                if target == .8:
                    original = next(p for p in result["methods"][method]["checkpoints"]
                                    if p["requested_coverage"] == .8)
                    if any(point[k] != original[k] for k in ("actual_coverage", "accepted_pixels", "accepted_errors", "risk")):
                        raise RuntimeError("Saved curve differs from the frozen primary80 checkpoint")
                points[method].append(point)
    return points


def _cached_native(row, device, inputs):
    path = Path(row["native_cache_path"]) if row.get("native_cache_path") else BASE / "data/native_logits" / (row["image_id"] + ".npy")
    if not path.is_file() or (not row.get("native_cache_path") and not path.with_suffix(".json").is_file()):
        raise RuntimeError("Required native cache is absent; backbone inference is forbidden")
    native, record, _ = features.native_cache(row, device)
    inputs[str(path)] = record["sha256"]
    meta_path = BASE / "data/features" / (row["image_id"] + ".json")
    meta = read(meta_path)
    if meta["device"] != device or meta["feature_source_sha256"] != sha256(BASE / "features.py"):
        raise RuntimeError("Feature backend or source differs from frozen evaluation")
    if meta["gt_sha256"] != row["gt_sha256"]:
        raise RuntimeError("Feature label provenance differs from metadata")
    inputs[str(meta_path)] = sha256(meta_path)
    inputs[row["gt_path"]] = row["gt_sha256"]
    return native


def diagnose_group(group, rows, device, inputs):
    started = time.perf_counter()
    result_path = BASE / "results/folds" / group / "result.json"
    result = read(result_path)
    inputs[str(result_path)] = sha256(result_path)
    fitted, selection = models(group)
    selection_path = BASE / "data/models" / group / "selection.json"
    inputs[str(selection_path)] = sha256(selection_path)
    for method in ("lr3", "lr3_class"):
        inputs[str(selection_path.parent / (method + ".joblib"))] = selection["model_hashes"][method]
    points = operating_points(group, result, inputs)
    cutoffs = {method: np.array([p["cutoff_bin"] for p in points[method]], dtype=np.int64) for method in CONFIGS}
    totals = np.zeros((19, 19), dtype=np.int64)
    accepted = {method: np.zeros((len(TARGETS), 19, 19), dtype=np.int64) for method in CONFIGS}
    for number, row in enumerate(rows, 1):
        native = _cached_native(row, device, inputs)
        target = features.load_target(row).ravel()
        for first, last, X, _ in features.iter_feature_blocks(native, device):
            gt = target[first:last]
            valid = gt < 19
            values = X[valid]
            pair_ids = gt[valid].astype(np.int64) * 19 + values[:, 3].astype(np.int64)
            totals += np.bincount(pair_ids, minlength=361).reshape(19, 19)
            for method in CONFIGS:
                scores = predict_scores(method, values, fitted.get(method))
                accepted[method] += confusion_at_cutoffs(pair_ids, histogram_indices(scores), cutoffs[method])
        if number == 1 or number % 25 == 0 or number == len(rows):
            print(f"Pixel diagnostics {group}: {number}/{len(rows)}; {time.perf_counter() - started:.1f}s", flush=True)
    valid_count, error_count = int(totals.sum()), int(totals.sum() - np.trace(totals))
    for method in CONFIGS:
        original = result["methods"][method]
        if (valid_count, error_count) != (original["num_pixels"], original["num_errors"]):
            raise RuntimeError("Full-pixel population differs from frozen evaluation")
        for index, point in enumerate(points[method]):
            counts = accepted[method][index]
            if (int(counts.sum()), int(counts.sum() - np.trace(counts))) != (point["accepted_pixels"], point["accepted_errors"]):
                raise RuntimeError(f"Frozen curve parity failed: {group}/{method}/{point['requested_coverage']}")
    elapsed = time.perf_counter() - started
    dump(f"logs/pixel_{group}.json", dict(status="PASS", group=group, condition=rows[0]["condition"],
        num_images=len(rows), num_valid=valid_count, num_errors=error_count,
        compared_operating_points=len(CONFIGS) * len(TARGETS), elapsed_seconds=elapsed))
    return dict(group=group, condition=rows[0]["condition"], totals=totals, accepted=accepted,
                points=points, num_images=len(rows), elapsed_seconds=elapsed)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def make_tables(groups):
    classes, confusions, operating, population, population_confusions = [], [], [], [], []
    for group in groups:
        identity = dict(group=group["group"], condition=group["condition"])
        total = group["totals"]
        class_pixels = total.sum(axis=0)
        class_errors = class_pixels - np.diag(total)
        for pred in range(19):
            population.append(dict(**identity, predicted_class=pred, predicted_class_name=CLASSES[pred],
                population_class_pixels=int(class_pixels[pred]), population_class_errors=int(class_errors[pred]),
                population_class_error_rate=_ratio(class_errors[pred], class_pixels[pred]),
                population_class_share=_ratio(class_pixels[pred], total.sum())))
        for gt, pred in np.argwhere(total > 0):
            if gt != pred:
                population_confusions.append(dict(**identity, gt_class=int(gt), gt_class_name=CLASSES[gt],
                    predicted_class=int(pred), predicted_class_name=CLASSES[pred], error_pixels=int(total[gt, pred])))
        for method in CONFIGS:
            for index, point in enumerate(group["points"][method]):
                operation = dict(**identity, method=method, **point)
                operating.append(dict(**operation, histogram_bins=HISTOGRAM_BINS, frozen_count_parity=True))
                matrix = group["accepted"][method][index]
                pixels = matrix.sum(axis=0)
                errors = pixels - np.diag(matrix)
                shared = dict(**identity, method=method, requested_coverage=point["requested_coverage"],
                              actual_coverage=point["actual_coverage"])
                for pred in range(19):
                    classes.append(dict(**shared, predicted_class=pred, predicted_class_name=CLASSES[pred],
                        population_class_pixels=int(class_pixels[pred]), population_class_errors=int(class_errors[pred]),
                        accepted_class_pixels=int(pixels[pred]), accepted_class_errors=int(errors[pred]),
                        accepted_class_error_rate=_ratio(errors[pred], pixels[pred]),
                        accepted_class_share=_ratio(pixels[pred], matrix.sum()),
                        accepted_error_share=_ratio(errors[pred], errors.sum()),
                        class_retention=_ratio(pixels[pred], class_pixels[pred]),
                        class_error_retention=_ratio(errors[pred], class_errors[pred])))
                for gt, pred in np.argwhere(matrix > 0):
                    if gt != pred:
                        confusions.append(dict(**shared, gt_class=int(gt), gt_class_name=CLASSES[gt],
                            predicted_class=int(pred), predicted_class_name=CLASSES[pred],
                            accepted_error_pixels=int(matrix[gt, pred]),
                            share_of_accepted_errors=_ratio(matrix[gt, pred], errors.sum()),
                            population_pair_errors=int(total[gt, pred]),
                            pair_error_retention=_ratio(matrix[gt, pred], total[gt, pred])))
    return {"pixel_class_acceptance.csv": classes, "pixel_confusions.csv": confusions,
            "pixel_group_operating_points.csv": operating, "pixel_class_population.csv": population,
            "pixel_confusion_population.csv": population_confusions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    complete = verify_execution()
    inputs = {str(BASE / "data/metadata_train.csv"): sha256(BASE / "data/metadata_train.csv"),
              str(BASE / "results/evaluation_complete.json"): sha256(BASE / "results/evaluation_complete.json")}
    selected = [row for row in ROWS if row["condition"] in ("fog", "night")]
    group_names = sorted({row["recording_group"] for row in selected})
    if len(selected) != 800 or len(group_names) != 9:
        raise RuntimeError("Expected all 800 Night/Fog images in nine recording proxies")
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    features.configure_torch(args.device)
    with patch.object(features, "FrozenBackbone", side_effect=RuntimeError("Backbone construction is forbidden in interpretation")):
        groups = [diagnose_group(name, [row for row in selected if row["recording_group"] == name], args.device, inputs)
                  for name in group_names]
    outputs = {}
    for name, rows in make_tables(groups).items():
        path = table(name, rows)
        outputs[str(path.relative_to(OUT))] = dict(rows=len(rows), sha256=sha256(path))
    verify_execution()
    manifest = dict(status="PASS", diagnostic_only=True, device=args.device, num_images=800, num_groups=9,
        requested_coverages=TARGETS, methods=CONFIGS, input_hashes=inputs, outputs=outputs,
        execution_source_hashes=complete["source_hashes"],
        interpretation_source_hashes={name: sha256(OUT / name) for name in ("pixel_diagnostics.py", "shared.py", "analysis_plan.txt")},
        elapsed_seconds=time.perf_counter() - started,
        group_runtime_seconds={group["group"]: group["elapsed_seconds"] for group in groups},
        compared_operating_points=9 * len(CONFIGS) * len(TARGETS), all_count_comparisons_exact=True,
        segmentation_forward_passes=0, trained_models=0,
        limitations=["Post-hoc descriptive counts at existing per-group cutoffs, not a new primary result",
                     "Cutoff uses histogram bin membership; inverse score edges are descriptive only",
                     "Only nonzero off-diagonal confusion pairs are serialized; absent rows mean zero",
                     "Recording families are filename proxies, not verified independent sessions"])
    dump("logs/pixel_diagnostics_manifest.json", manifest)
    print(f"PASS: {len(selected)} images, {manifest['compared_operating_points']} exact operating-point comparisons; {manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
