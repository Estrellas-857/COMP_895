"""Stage 3C: validation evaluation with the frozen shared Stage 3B temperature."""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from stage1.evaluate_condition import (
    MODEL_NAME, RGB_ROOT, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES, IGNORE_INDEX,
    find_gt_path, update_confusion_matrix,
)
from stage3.evaluate_calibration import (
    CalibrationAccumulator, valid_pixel_nll_sum, EXPECTED_IMAGES, write_csv,
)

from stage1._layout import manifest_key, matches_frozen_hash

ROOT = Path("results/stage3_calibration/stage3c")
RAW_ROOT = Path("results/stage3_calibration")
TEMPERATURE_PATH = RAW_ROOT / "stage3b/fitted_temperature.json"
FROZEN_TEMPERATURE = 2.124186626456894  # Integrity assertion only; loaded JSON supplies T.
FROZEN_REVISION = "7a91500a7086b805eeb868719ea5542a3e0bdeb3"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def protected_file_check(path):
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    assert matches_frozen_hash(path, manifest[manifest_key(path, manifest)]), f"Protected file changed: {path}"


def load_temperature():
    protected_file_check(TEMPERATURE_PATH)
    artifact = json.loads(TEMPERATURE_PATH.read_text())
    temperature = artifact["temperature"]
    assert temperature == FROZEN_TEMPERATURE and temperature > 0
    assert artifact["validation_used"] is False and artifact["converged"] is True
    assert artifact["model"] == MODEL_NAME and artifact["model_revision"] == FROZEN_REVISION
    assert "ACDC train/calibration_fit only" in artifact["fit_dataset"]
    return temperature, artifact


def load_raw(condition):
    path = RAW_ROOT / condition / "raw_calibration_summary.csv"
    bins_path = RAW_ROOT / condition / "confidence_bins.csv"
    protected_file_check(path)
    protected_file_check(bins_path)
    rows = rows_from(path)
    assert len(rows) == 1 and rows[0]["condition"] == condition
    return rows[0], rows_from(bins_path)


def check_raw_reproduction(summary, bins, frozen, frozen_bins):
    for key, value in summary.items():
        if key == "condition":
            assert value == frozen[key]
        elif isinstance(value, (int, np.integer)):
            assert value == int(frozen[key]), (key, value, frozen[key])
        else:
            assert math.isclose(value, float(frozen[key]), rel_tol=1e-12, abs_tol=1e-12), key
    assert len(bins) == len(frozen_bins) == 15
    for actual, expected in zip(bins, frozen_bins):
        for key, value in actual.items():
            other = float(expected[key])
            if math.isnan(value):
                assert math.isnan(other)
            else:
                assert math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-12), key


def comparison_row(calibrated, raw, temperature, prediction_changes):
    result = {key: calibrated[key] for key in (
        "condition", "num_images", "num_valid_pixels", "num_correct_pixels",
        "num_incorrect_pixels", "pixel_accuracy", "error_rate", "num_bins", "high_conf_threshold")}
    result.update(temperature=temperature, prediction_changes=prediction_changes,
                  prediction_pixels_checked=calibrated["num_valid_pixels"])
    for key in ("ece", "nll", "high_conf_coverage", "high_conf_error_rate",
                "fraction_incorrect_that_are_high_conf", "num_high_conf_pixels", "num_high_conf_incorrect"):
        result[f"raw_{key}"] = int(raw[key]) if key.startswith("num_") else float(raw[key])
        result[f"calibrated_{key}"] = calibrated[key]
    for metric in ("ece", "nll"):
        before, after = float(raw[metric]), calibrated[metric]
        result[f"{metric}_absolute_change"] = after - before
        result[f"{metric}_relative_change"] = (after - before) / before
        result[f"{metric}_improved"] = after < before
    result.update(baseline_accuracy_check="PASS", stage1_confusion_matrix_exact_match=True,
                  raw_stage3a_reproduction_check="PASS", frozen_temperature_check="PASS")
    return result


def plot_reliability(condition, summary, output):
    # Both panels read their saved bin CSVs, with identical axes and no connecting lines.
    sources = (RAW_ROOT / condition / "confidence_bins.csv",
               output / "calibrated_confidence_bins.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), sharex=True, sharey=True, layout="constrained")
    for ax, source, label, color, key in zip(
        axes, sources, ("Raw", "Calibrated"), ("#1764a0", "#c45b16"), ("raw_ece", "calibrated_ece")
    ):
        bins = [r for r in rows_from(source) if int(r["count"]) > 0]
        ax.plot([0, 1], [0, 1], "--", color="0.45", label="Ideal calibration")
        ax.scatter([float(r["mean_confidence"]) for r in bins],
                   [float(r["accuracy"]) for r in bins], color=color, s=38,
                   edgecolors="white", linewidths=0.5, zorder=3, label="Non-empty bins")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence",
               title=f"{label} · ECE = {summary[key]:.4f}")
        ax.set_aspect("equal")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Empirical accuracy")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(f"{condition.capitalize()}: raw vs temperature-scaled confidence\n"
                 f"{summary['num_images']} validation images · 15 equal-width bins · T = {summary['temperature']:.6f}")
    fig.savefig(output / "raw_vs_calibrated_reliability.png", dpi=300)
    plt.close(fig)


def evaluate(condition):
    temperature, artifact = load_temperature()
    frozen, frozen_bins = load_raw(condition)
    output = ROOT / condition
    if (output / "calibrated_summary.csv").exists():
        raise RuntimeError(f"Refuse to overwrite completed Stage 3C results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((RGB_ROOT / condition / "val").rglob("*_rgb_anon.png"))
    assert len(paths) == EXPECTED_IMAGES[condition] == int(frozen["num_images"])
    assert all(find_gt_path(p, condition).is_file() for p in paths)
    assert all("val" in p.parts and "train" not in p.parts for p in paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 3C {condition}: {len(paths)} validation images; T={temperature!r}; device={device}", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, revision=artifact["model_revision"])
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, revision=artifact["model_revision"]).to(device)
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
    raw_acc, calibrated_acc = CalibrationAccumulator(), CalibrationAccumulator()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    prediction_changes = 0
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
        inputs = processor(images=image, return_tensors="pt",
                           size={"height": INPUT_HEIGHT, "width": INPUT_WIDTH})
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            native = model(**inputs).logits
            assert native.shape == (1, NUM_CLASSES, 128, 256) and native.dtype == torch.float32
            logits = F.interpolate(native, size=(1080, 1920), mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            raw_confidence = torch.softmax(logits, dim=1).max(dim=1).values[0].cpu().numpy()
            raw_nll_sum = valid_pixel_nll_sum(logits, gt)
            # Stage 3B precision: float32 interpolation, then float64 scaling.
            scaled = logits.to(torch.float64) / temperature
            scaled_prediction = scaled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            prediction_changes += int(((prediction != scaled_prediction) & valid).sum())
            calibrated_confidence = torch.softmax(scaled, dim=1).max(dim=1).values[0].cpu().numpy()
            calibrated_nll_sum = valid_pixel_nll_sum(scaled, gt)
        correct = prediction[valid] == gt[valid]
        raw_acc.update(raw_confidence[valid], correct, raw_nll_sum)
        calibrated_acc.update(calibrated_confidence[valid], correct, calibrated_nll_sum)
        update_confusion_matrix(confusion, prediction, gt)
        del image, gt, inputs, native, logits, scaled, prediction, scaled_prediction
        del valid, correct, raw_confidence, calibrated_confidence
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(f"[{index}/{len(paths)}] elapsed={time.perf_counter()-started:.1f}s", flush=True)
    raw_check, raw_check_bins = raw_acc.finalize(condition, len(paths))
    calibrated, calibrated_bins = calibrated_acc.finalize(condition, len(paths))
    check_raw_reproduction(raw_check, raw_check_bins, frozen, frozen_bins)
    baseline = np.load(Path("results") / f"{condition}_val" / "confusion_matrix.npy")
    assert np.array_equal(confusion, baseline), "Frozen Stage 1 confusion matrix mismatch"
    assert prediction_changes == 0, "Temperature scaling changed valid-pixel predictions"
    assert calibrated["pixel_accuracy"] == float(frozen["pixel_accuracy"])
    assert int(confusion.sum()) == calibrated["num_valid_pixels"]
    assert int(np.trace(confusion)) == calibrated["num_correct_pixels"]
    for k in ("ece", "high_conf_coverage", "high_conf_error_rate", "fraction_incorrect_that_are_high_conf"):
        assert 0 <= calibrated[k] <= 1
    summary = comparison_row(calibrated, frozen, temperature, prediction_changes)
    write_csv(output / "calibrated_confidence_bins.csv", calibrated_bins)
    write_csv(output / "calibrated_summary.csv", [summary])
    np.save(output / "confusion_matrix.npy", confusion)
    plot_reliability(condition, summary, output)
    metadata = dict(temperature_artifact=str(TEMPERATURE_PATH), temperature_sha256=sha256(TEMPERATURE_PATH),
                    model=MODEL_NAME, model_revision=FROZEN_REVISION, processor_class=type(processor).__name__,
                    torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                    precision="float32 model/interpolation/raw branch; float64 temperature division, calibrated softmax and log-softmax (same as Stage 3B)",
                    validation_parameters_fitted=False,
                    raw_summary_sha256=sha256(RAW_ROOT / condition / "raw_calibration_summary.csv"),
                    raw_bins_sha256=sha256(RAW_ROOT / condition / "confidence_bins.csv"))
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = [f"STAGE 3C {condition.upper()} - FROZEN TEMPERATURE VALIDATION", ""]
    lines += [f"{k}: {v}" for k, v in summary.items()]
    lines += ["", "absolute_change = calibrated - raw; relative_change = (calibrated - raw) / raw.",
              "Negative changes indicate improvement for ECE/NLL; relative changes in CSV are fractions.",
              "HC means confidence >= 0.90; high_conf_error_rate = P(incorrect | HC).",
              "fraction_incorrect_that_are_high_conf = P(HC | incorrect).",
              "All metrics are pixel-weighted across valid trainIds 0-18; ignore trainId 255.",
              "15 bins: [lower, upper), with 1.0 included in the final bin; empty-bin statistics are NaN.",
              "NLL uses stable log-softmax, in nats. Raw comparisons use frozen Stage 3A CSV values.",
              metadata["precision"],
              "PASS: raw metrics and bins reproduced in the same inference pass; frozen artifacts not rewritten.",
              "PASS: zero prediction changes across all valid pixels; exact Stage 1 confusion-matrix agreement.",
              "Reliability panels read frozen raw and new calibrated bin CSVs; both axes span [0,1].",
              "No parameter fitting, threshold selection or failure-ranking analysis performed."]
    if device.type == "cuda":
        lines.append(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated()/2**30:.3f} GiB")
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    load_temperature()  # Temperature must still match its frozen artifact hash.
    print("\n".join(lines), flush=True)
    print(f"Results saved: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=list(EXPECTED_IMAGES))
    evaluate(parser.parse_args().condition)
