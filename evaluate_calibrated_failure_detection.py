"""Stage 3D: frozen-temperature failure discrimination using Stage 2D metrics."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from evaluate_condition import (MODEL_NAME, RGB_ROOT, INPUT_HEIGHT, INPUT_WIDTH,
                                NUM_CLASSES, IGNORE_INDEX, find_gt_path, update_confusion_matrix)
from evaluate_uncertainty import UncertaintyAccumulator, HIST_EDGES, NUM_BINS
from evaluate_failure_detection import compute_curves
from evaluate_calibration import EXPECTED_IMAGES, write_csv
from evaluate_calibrated import sha256, rows_from, FROZEN_TEMPERATURE, FROZEN_REVISION

ROOT = Path("results/stage3_calibration/stage3d")
STAGE2_ROOT = Path("results/stage2_uncertainty")
CALIBRATION_ROOT = Path("results/stage3_calibration")
TEMPERATURE_PATH = CALIBRATION_ROOT / "stage3b/fitted_temperature.json"
METHOD = "1000-bin histogram approximation"
AUDIT_FIELDS = ["condition", "rgb_path", "row", "column", "gt", "logit_prediction",
                "raw_softmax_prediction", "logit_at_logit_prediction", "logit_at_softmax_prediction",
                "probability_at_logit_prediction", "probability_at_softmax_prediction", "probability_tie"]


def protected(path):
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    assert sha256(path) == manifest[str(path)], f"Protected file changed: {path}"


def load_temperature():
    protected(TEMPERATURE_PATH)
    artifact = json.loads(TEMPERATURE_PATH.read_text())
    assert artifact["temperature"] == FROZEN_TEMPERATURE
    assert artifact["temperature"] > 0 and artifact["validation_used"] is False
    assert artifact["model"] == MODEL_NAME and artifact["model_revision"] == FROZEN_REVISION
    return artifact["temperature"]


def load_baselines(condition):
    raw_path = STAGE2_ROOT / "failure_detection" / condition / "summary.csv"
    hist_path = STAGE2_ROOT / condition / "uncertainty_histograms.npz"
    stage3c_path = CALIBRATION_ROOT / "stage3c" / condition / "calibrated_summary.csv"
    stage3a_path = CALIBRATION_ROOT / condition / "raw_calibration_summary.csv"
    for path in (raw_path, hist_path, stage3c_path, stage3a_path):
        protected(path)
    rows = rows_from(raw_path)
    assert len(rows) == 2 and {r["method"] for r in rows} == {"msp", "entropy"}
    raw = {r["method"]: r for r in rows}
    with np.load(hist_path) as data:
        hist = {k: data[k].copy() for k in data.files}
    assert NUM_BINS == 1000
    np.testing.assert_array_equal(hist["bin_edges"], HIST_EDGES)
    for method in raw:
        metrics = compute_curves(hist[f"{method}_correct"], hist[f"{method}_incorrect"])
        assert int(raw[method]["histogram_bins"]) == NUM_BINS
        for key in ("auroc", "auprc"):
            assert metrics[key] == float(raw[method][key])
        assert metrics["prevalence"] == float(raw[method]["positive_prevalence"])
        assert metrics["positive_pixels"] == int(raw[method]["positive_pixels"])
        assert metrics["negative_pixels"] == int(raw[method]["negative_pixels"])
    stage3c, stage3a = rows_from(stage3c_path)[0], rows_from(stage3a_path)[0]
    for key in ("num_images", "num_valid_pixels", "num_correct_pixels", "num_incorrect_pixels", "pixel_accuracy"):
        assert stage3c[key] == stage3a[key]
    return raw, hist, stage3c


def uncertainty_arrays(probs):
    maximum, prediction = probs.max(dim=1)
    # Exact Stage 2 entropy definition, including its numerical log clamp.
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1) / math.log(NUM_CLASSES)
    maximum = maximum[0].cpu().numpy()
    return 1.0 - maximum, entropy[0].cpu().numpy(), prediction[0].cpu().numpy().astype(np.uint8)


def update_histograms(accumulators, msp, entropy, correct, incorrect):
    for method, values in (("msp", msp), ("entropy", entropy)):
        assert np.isfinite(values).all()
        assert values.min() >= -1e-7 and values.max() <= 1 + 1e-7
        for label, mask in (("correct", correct), ("incorrect", incorrect)):
            accumulators[f"{method}_{label}"].update(values[mask])


def evaluate(condition):
    temperature = load_temperature()
    raw, frozen_hist, baseline = load_baselines(condition)
    output = ROOT / condition
    if (output / "calibrated_failure_detection_summary.csv").exists():
        raise RuntimeError("Completed Stage 3D output already exists; refusing overwrite")
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((RGB_ROOT / condition / "val").rglob("*_rgb_anon.png"))
    assert len(paths) == EXPECTED_IMAGES[condition] == int(baseline["num_images"])
    assert all(find_gt_path(p, condition).is_file() for p in paths)
    assert all("val" in p.parts and "train" not in p.parts for p in paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 3D {condition}: {len(paths)} images; frozen T={temperature!r}; {METHOD}; device={device}", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, revision=FROZEN_REVISION)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, revision=FROZEN_REVISION).to(device)
    assert model.config._commit_hash == FROZEN_REVISION
    assert type(processor).__name__ == "SegformerImageProcessor"
    model.eval()
    if device.type == "cuda":
        with torch.inference_mode():
            dummy = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
            for _ in range(10):
                model(pixel_values=dummy)
            del dummy
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    keys = [f"{m}_{label}" for m in ("msp", "entropy") for label in ("correct", "incorrect")]
    calibrated = {k: UncertaintyAccumulator() for k in keys}
    raw_recheck = {k: UncertaintyAccumulator() for k in keys}
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    raw_softmax_confusion = np.zeros_like(confusion)
    changed, valid_count, correct_count, incorrect_count = 0, 0, 0, 0
    audit = []
    started = time.perf_counter()
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = source.convert("RGB")
        gt_path = find_gt_path(path, condition)
        assert "val" in gt_path.parts and "train" not in gt_path.parts
        with Image.open(gt_path) as source:
            gt = np.array(source)
        assert image.size == (1920, 1080) and gt.shape == (1080, 1920)
        valid = (gt != IGNORE_INDEX) & (gt >= 0) & (gt < NUM_CLASSES)
        assert not np.any(~valid & (gt != IGNORE_INDEX))
        inputs = processor(images=image, return_tensors="pt", size={"height": INPUT_HEIGHT, "width": INPUT_WIDTH})
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            native = model(**inputs).logits
            assert native.shape == (1, NUM_CLASSES, 128, 256) and native.dtype == torch.float32
            logits = F.interpolate(native, size=(1080, 1920), mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            raw_probs = torch.softmax(logits, dim=1)
            raw_msp, raw_entropy, raw_pred = uncertainty_arrays(raw_probs)
            # Audit historical float32 softmax ties, without changing either frozen definition.
            for y, x in np.argwhere((pred != raw_pred) & valid):
                a, b = int(pred[y, x]), int(raw_pred[y, x])
                pa, pb = float(raw_probs[0, a, y, x].item()), float(raw_probs[0, b, y, x].item())
                audit.append(dict(condition=condition, rgb_path=str(path), row=int(y), column=int(x),
                                  gt=int(gt[y, x]), logit_prediction=a, raw_softmax_prediction=b,
                                  logit_at_logit_prediction=float(logits[0, a, y, x].item()),
                                  logit_at_softmax_prediction=float(logits[0, b, y, x].item()),
                                  probability_at_logit_prediction=pa, probability_at_softmax_prediction=pb,
                                  probability_tie=pa == pb))
            del raw_probs
            # Retain calibrated-branch precision from Stage 3B/3C.
            scaled = logits.to(torch.float64) / temperature
            scaled_pred = scaled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            changed += int(((scaled_pred != pred) & valid).sum())
            probs = torch.softmax(scaled, dim=1)
            msp, entropy, probability_pred = uncertainty_arrays(probs)
            del probs
        correct, incorrect = (pred == gt) & valid, (pred != gt) & valid
        update_histograms(calibrated, msp, entropy, correct, incorrect)
        update_histograms(raw_recheck, raw_msp, raw_entropy, (raw_pred == gt) & valid, (raw_pred != gt) & valid)
        update_confusion_matrix(confusion, pred, gt)
        update_confusion_matrix(raw_softmax_confusion, raw_pred, gt)
        valid_count += int(valid.sum())
        correct_count += int(correct.sum())
        incorrect_count += int(incorrect.sum())
        del image, gt, valid, inputs, native, logits, scaled, pred, scaled_pred, probability_pred
        del msp, entropy, raw_msp, raw_entropy, raw_pred, correct, incorrect
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(f"[{index}/{len(paths)}] elapsed={time.perf_counter()-started:.1f}s", flush=True)
    assert changed == 0
    assert (valid_count, correct_count, incorrect_count) == tuple(int(baseline[k]) for k in
           ("num_valid_pixels", "num_correct_pixels", "num_incorrect_pixels"))
    np.testing.assert_array_equal(confusion, np.load(f"results/{condition}_val/confusion_matrix.npy"))
    np.testing.assert_array_equal(raw_softmax_confusion, np.load(STAGE2_ROOT / condition / "confusion_matrix.npy"))
    for key in keys:
        np.testing.assert_array_equal(raw_recheck[key].hist, frozen_hist[key])
        assert calibrated[key].hist.dtype == np.int64
        assert calibrated[key].count == (correct_count if key.endswith("_correct") else incorrect_count)
        assert int(calibrated[key].hist.sum()) == calibrated[key].count
    assert all(row["probability_tie"] for row in audit), "Unexpected historical argmax discrepancy"
    historical_correct_change = sum(int(a["raw_softmax_prediction"] == a["gt"]) -
                                    int(a["logit_prediction"] == a["gt"]) for a in audit)
    assert correct_count + historical_correct_change == int(raw["msp"]["negative_pixels"])
    np.savez(output / "calibrated_uncertainty_histograms.npz", bin_edges=HIST_EDGES,
             **{f"{k}_hist": calibrated[k].hist for k in keys})
    np.save(output / "confusion_matrix.npy", confusion)
    np.save(output / "raw_softmax_confusion_recheck.npy", raw_softmax_confusion)
    with (output / "raw_softmax_tie_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    result_rows = []
    for method in ("msp", "entropy"):
        metric = compute_curves(calibrated[f"{method}_correct"].hist, calibrated[f"{method}_incorrect"].hist)
        assert metric["positive_pixels"] == incorrect_count and metric["negative_pixels"] == correct_count
        assert all(0 <= metric[k] <= 1 for k in ("auroc", "auprc", "prevalence"))
        raw_prevalence = float(raw[method]["positive_prevalence"])
        result_rows.append(dict(condition=condition, method=method, metric_method=METHOD,
            temperature=temperature, histogram_bins=NUM_BINS, num_images=len(paths),
            num_valid_pixels=valid_count, num_correct_pixels=correct_count, num_incorrect_pixels=incorrect_count,
            pixel_accuracy=correct_count/valid_count, prediction_changes=changed, prediction_pixels_checked=valid_count,
            error_prevalence=metric["prevalence"], raw_error_prevalence=raw_prevalence,
            raw_num_incorrect_pixels=int(raw[method]["positive_pixels"]),
            incorrect_count_change_vs_stage2d=incorrect_count-int(raw[method]["positive_pixels"]),
            raw_auroc=float(raw[method]["auroc"]), calibrated_auroc=metric["auroc"],
            auroc_change=metric["auroc"]-float(raw[method]["auroc"]),
            raw_auprc=float(raw[method]["auprc"]), calibrated_auprc=metric["auprc"],
            auprc_change=metric["auprc"]-float(raw[method]["auprc"]),
            pr_random_baseline=metric["prevalence"], raw_pr_random_baseline=raw_prevalence,
            raw_auprc_gain=float(raw[method]["auprc"])-raw_prevalence,
            calibrated_auprc_gain=metric["auprc"]-metric["prevalence"],
            raw_histogram_exact_reproduction=True, stage1_3c_confusion_exact_match=True,
            historical_softmax_argmax_differences=len(audit)))
    write_csv(output / "calibrated_failure_detection_summary.csv", result_rows)
    provenance = dict(model=MODEL_NAME, model_revision=FROZEN_REVISION, temperature_sha256=sha256(TEMPERATURE_PATH),
                      histogram_implementation="evaluate_uncertainty.UncertaintyAccumulator (unchanged)",
                      metric_implementation="evaluate_failure_detection.compute_curves (unchanged)",
                      raw_precision="Stage 2 float32", calibrated_precision="Stage 3B/3C float64 after float32 interpolation",
                      reserved_reliability_selection_data_used=False, parameters_fitted=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    lines = [f"STAGE 3D {condition.upper()} - CALIBRATED FAILURE DISCRIMINATION", ""]
    lines += [str(r) for r in result_rows]
    lines += ["", "Positive class=incorrect pixel; higher MSP/entropy uncertainty indicates failure.",
              "1000 equal-width [0,1] bins; [lower,upper), final bin includes 1.0; int64 counts.",
              "AUROC: reverse-bin accumulation from (0,0), trapezoidal integration.",
              "AUPRC: sum(delta_recall * precision[1:]); precision=1 when nothing is predicted positive.",
              "PR baseline=incorrect-pixel prevalence; PR gain=AUPRC minus the corresponding baseline.",
              "Raw metrics are unchanged frozen Stage 2D CSV values. Changes=calibrated minus raw.",
              "Entropy uses the unchanged Stage 2 probability log clamp of 1e-12 and normalization by log(19).",
              "PASS: raw float32 histograms reproduced exactly from inference using Stage 2 softmax-argmax labels.",
              "PASS: calibrated labels use frozen full-resolution logit argmax; exact Stage 1/3C counts/confusion.",
              "PASS: zero temperature-induced prediction changes; complete histograms; frozen T unchanged.",
              f"Historical softmax-vs-logit argmax differences: {len(audit)}; all verified float32 probability ties.",
              f"Incorrect-count difference vs Stage 2D: {result_rows[0]['incorrect_count_change_vs_stage2d']:+d}.",
              "The historical discrepancy is preserved, not corrected: raw PR baselines use Stage 2D counts, calibrated PR baselines use Stage 3C counts.",
              "Metrics are histogram approximations. Small changes are descriptive, not statistically tested.",
              "No reserved train data, parameter fitting, rejection-threshold selection, risk-coverage or Stage 4 analysis."]
    if device.type == "cuda":
        lines.append(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated()/2**30:.3f} GiB")
    (output / "summary.txt").write_text("\n".join(lines)+"\n")
    assert load_temperature() == temperature
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=list(EXPECTED_IMAGES))
    evaluate(parser.parse_args().condition)
