"""Evaluate saved fold models on every valid pixel using the original CUDA extractor."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from experiments.cse805_error_prediction_v1 import features
from experiments.cse805_error_prediction_v1.evaluation import HistogramAccumulator, HISTOGRAM_BINS, histogram_indices
from experiments.cse805_error_prediction_v1.interpretation_v1.shared import group_rows
from experiments.paired_calibration_v1.metrics import weighted_risk_coverage
from .model import BASE, DATA, OUT, OLD, TARGETS, read, score_rows, sha256, write


def evaluate(group, records, manifest_hash, source_digest):
    folder = OUT / "folds" / group
    selection = read(folder/"selection.json")
    model_path = folder/"interaction.joblib"
    assert selection["manifest_sha256"] == manifest_hash and selection["model_sha256"] == sha256(model_path)
    path = folder/"evaluation.json"
    histogram = folder/"histograms.npz"
    if path.exists():
        old = read(path)
        if (old["model_sha256"] != selection["model_sha256"] or old["evaluate_sha256"] != source_digest
                or old["histogram_sha256"] != sha256(histogram)):
            raise RuntimeError(f"Stale evaluation {group}")
        return
    rows = group_rows(group)
    weather = rows[0]["condition"]
    assert {r["condition"] for r in rows} == {weather}
    model = joblib.load(model_path)
    acc = HistogramAccumulator(threshold=selection["threshold"])
    veg_pixels, veg_errors = np.zeros((2, HISTOGRAM_BINS), dtype=np.int64)
    night_qa = {k:[] for k in ("scores", "errors", "weights", "tie_keys")}
    n_qa = 0
    started = time.perf_counter()
    for i, row in enumerate(rows, 1):
        r = records[row["image_id"]]
        native = r["native_cache"]
        assert r["recording_group"] == group and r["gt_sha256"] == row["gt_sha256"]
        if sha256(native["path"]) != native["sha256"] or sha256(r["path"]) != r["sha256"]:
            raise RuntimeError("Cache or pixel sample hash mismatch: " + row["image_id"])
        logits = np.load(native["path"], allow_pickle=False)
        gt = features.load_target(row).ravel()
        with np.load(r["path"], allow_pickle=False) as sample:
            ids, qx, qy = (sample["qa_" + k] for k in ("indices", "X", "errors"))
            assert len(ids) == 8192 and np.all(np.diff(ids) > 0) and np.all(gt[ids] < 19)
            np.testing.assert_array_equal(qx[:,3] != gt[ids], qy)
            reference = model.predict_proba(qx)[:,1]
            np.testing.assert_array_equal(score_rows(model, qx), reference)
            if weather == "night":
                night_qa["scores"].append(reference)
                for key in ("errors", "weights", "tie_keys"):
                    night_qa[key].append(sample["qa_"+key])
        num_valid = num_errors = qa_hits = 0
        for first, last, X, _ in features.iter_feature_blocks(logits, "cuda"):
            lo, hi = np.searchsorted(ids, (first,last))
            np.testing.assert_array_equal(X[ids[lo:hi]-first], qx[lo:hi])
            qa_hits += hi-lo
            target = gt[first:last]
            valid = target < 19
            if not valid.any():
                continue
            values = X[valid]
            errors = values[:,3] != target[valid]
            scores = score_rows(model, values)
            if i == 1 and first == 0:
                np.testing.assert_array_equal(scores, model.predict_proba(values)[:,1])
            acc.update(scores, errors)
            if weather == "night":
                veg = values[:,3] == 8
                bins = histogram_indices(scores[veg])
                veg_pixels += np.bincount(bins, minlength=HISTOGRAM_BINS)
                veg_errors += np.bincount(bins[errors[veg]], minlength=HISTOGRAM_BINS)
            num_valid += len(values)
            num_errors += int(errors.sum())
        assert (num_valid, num_errors, qa_hits) == (r["num_valid"], r["num_errors"], 8192), row["image_id"]
        n_qa += qa_hits
        if i % 40 == 0:
            print(f"Evaluate {group}: {i}/{len(rows)} images", flush=True)
    metrics = acc.finalize(TARGETS)
    del metrics["risk_coverage"]
    frozen = read(BASE/"results/folds"/group/"result.json")["methods"][OLD]
    assert (metrics["num_pixels"],metrics["num_errors"]) == (frozen["num_pixels"],frozen["num_errors"])
    assert metrics["fixed_threshold"]["complete_population"]
    qa_points = []
    if weather == "night":
        array = {k:np.concatenate(v) for k,v in night_qa.items()}
        qa_points = weighted_risk_coverage(array["scores"],array["errors"],array["weights"],
                                           array["tie_keys"],TARGETS)["checkpoints"]
    with histogram.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, correct=acc.correct, incorrect=acc.incorrect,
                            **(dict(vegetation=veg_pixels, vegetation_errors=veg_errors) if weather=="night" else {}))
    histogram.with_suffix(".tmp").replace(histogram)
    write(path, dict(group=group, weather=weather, num_images=len(rows), metrics=metrics,
                     qa_raw_secondary=qa_points, qa_features_and_scores_verified=n_qa,
                     evaluated_image_ids=[r["image_id"] for r in rows], eval_seconds=time.perf_counter()-started,
                     gpu=torch.cuda.get_device_name(0), manifest_sha256=manifest_hash,
                     model_sha256=selection["model_sha256"], histogram_sha256=sha256(histogram),
                     evaluate_sha256=source_digest))
    print(f"Done {group} ({weather}): {len(rows)} images, {metrics['num_pixels']} valid pixels", flush=True)


def main():
    features.configure_torch("cuda")
    seal = read(OUT/"run_manifest.json")
    datafile = DATA/"features/manifest.json"
    assert sha256(datafile) == seal["inputs"]["data/features/manifest.json"]
    assert seal["model_sha256"] == sha256(OUT/"model.py")
    records = {r["image_id"]:r for r in read(datafile)["records"]}
    groups = sys.argv[1:] or seal["outer_groups"]
    if len(groups) != len(set(groups)) or not set(groups) <= set(seal["outer_groups"]):
        raise ValueError("Unknown or duplicate evaluation group")
    for group in groups:
        evaluate(group, records, sha256(OUT/"run_manifest.json"), sha256(Path(__file__)))


if __name__ == "__main__":
    main()
