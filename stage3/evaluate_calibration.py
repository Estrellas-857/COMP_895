"""Stage 3A: streaming, raw pixel calibration on ACDC validation sets."""

import argparse
import csv
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

# Reuse the frozen Stage 1 paths, matching, dimensions, and confusion logic.
from stage1.evaluate_condition import (
    IGNORE_INDEX, INPUT_HEIGHT, INPUT_WIDTH, MODEL_NAME, NUM_CLASSES,
    RGB_ROOT, find_gt_path, update_confusion_matrix,
)

OUTPUT_ROOT = Path("results/stage3_calibration")
NUM_BINS = 15
BIN_EDGES = np.linspace(0.0, 1.0, NUM_BINS + 1)
HIGH_CONF_THRESHOLD = 0.90  # Prespecified descriptive statistic, not tuned.
ACCURACY_TOLERANCE = 1e-5  # Absolute accuracy; fixed before evaluation.
EXPECTED_IMAGES = {"fog": 100, "rain": 100, "snow": 100, "night": 106}


class CalibrationAccumulator:
    """Pixel-weighted statistics; never retain tensors across images."""

    def __init__(self):
        self.count = np.zeros(NUM_BINS, dtype=np.int64)
        self.confidence_sum = np.zeros(NUM_BINS, dtype=np.float64)
        self.correct_count = np.zeros(NUM_BINS, dtype=np.int64)
        self.num_valid = 0
        self.num_correct = 0
        self.num_high_conf = 0
        self.num_high_conf_incorrect = 0
        self.nll_sum = 0.0

    def update(self, confidence, correct, nll_sum):
        confidence = np.asarray(confidence)
        correct = np.asarray(correct, dtype=bool)
        if confidence.ndim != 1 or confidence.shape != correct.shape:
            raise ValueError("Expected matching one-dimensional valid-pixel arrays")
        if not np.all(np.isfinite(confidence)) or np.any(
            (confidence < 0) | (confidence > 1)
        ):
            raise ValueError("Confidence must be finite and within [0, 1]")
        if not np.isfinite(nll_sum) or nll_sum < 0:
            raise ValueError("NLL sum must be finite and nonnegative")

        # [lower, upper), except the final bin [14/15, 1]. Interior edges
        # belong to the right bin; both 0 and 1 are included exactly once.
        indices = np.searchsorted(
            BIN_EDGES[1:-1], confidence.astype(np.float64), side="right"
        )
        self.count += np.bincount(indices, minlength=NUM_BINS)
        self.confidence_sum += np.bincount(
            indices, weights=confidence, minlength=NUM_BINS
        )
        self.correct_count += np.bincount(indices[correct], minlength=NUM_BINS)
        high_conf = confidence >= HIGH_CONF_THRESHOLD
        self.num_valid += confidence.size
        self.num_correct += int(correct.sum())
        self.num_high_conf += int(high_conf.sum())
        self.num_high_conf_incorrect += int((high_conf & ~correct).sum())
        self.nll_sum += float(nll_sum)

    def finalize(self, condition, num_images):
        if self.num_valid == 0:
            raise RuntimeError("No valid pixels were evaluated")
        rows = []
        for i, count in enumerate(self.count):
            mean = self.confidence_sum[i] / count if count else np.nan
            accuracy = self.correct_count[i] / count if count else np.nan
            rows.append({
                "bin_index": i,
                "bin_lower": BIN_EDGES[i],
                "bin_upper": BIN_EDGES[i + 1],
                "count": int(count),
                "fraction": count / self.num_valid,
                "mean_confidence": mean,
                "accuracy": accuracy,
                "error_rate": 1 - accuracy,
                "calibration_gap": abs(accuracy - mean),
            })
        ece = sum(r["fraction"] * r["calibration_gap"] for r in rows if r["count"])
        incorrect = self.num_valid - self.num_correct
        summary = {
            "condition": condition,
            "num_images": num_images,
            "num_valid_pixels": self.num_valid,
            "num_correct_pixels": self.num_correct,
            "num_incorrect_pixels": incorrect,
            "pixel_accuracy": self.num_correct / self.num_valid,
            "error_rate": incorrect / self.num_valid,
            "ece": ece,
            "nll": self.nll_sum / self.num_valid,
            "num_bins": NUM_BINS,
            "high_conf_threshold": HIGH_CONF_THRESHOLD,
            "num_high_conf_pixels": self.num_high_conf,
            "high_conf_coverage": self.num_high_conf / self.num_valid,
            "num_high_conf_incorrect": self.num_high_conf_incorrect,
            # P(incorrect | high confidence), denominator = all high confidence.
            "high_conf_error_rate": self.num_high_conf_incorrect / self.num_high_conf
            if self.num_high_conf else np.nan,
            # P(high confidence | incorrect), denominator = all incorrect.
            "fraction_incorrect_that_are_high_conf": self.num_high_conf_incorrect / incorrect
            if incorrect else np.nan,
        }
        assert int(self.count.sum()) == self.num_valid
        assert int(self.correct_count.sum()) == self.num_correct
        assert np.all(self.correct_count <= self.count)
        assert 0 <= ece <= 1
        assert 0 <= self.num_high_conf_incorrect <= self.num_high_conf <= self.num_valid
        assert self.num_high_conf_incorrect <= incorrect
        return summary, rows


def valid_pixel_nll_sum(logits, gt):
    """-log p(GT) in natural-log units, without clamping or log(0)."""
    target = torch.as_tensor(gt.astype(np.int64), device=logits.device)
    valid = (target != IGNORE_INDEX) & (target >= 0) & (target < NUM_CLASSES)
    safe_target = target.masked_fill(~valid, 0)
    log_probabilities = F.log_softmax(logits, dim=1)
    gt_log_probability = log_probabilities[0].gather(0, safe_target[None])[0]
    return float((-gt_log_probability[valid]).sum(dtype=torch.float64).item())


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_reliability(bins_path, output_path, summary):
    # Read the saved CSV itself so the diagram and exported statistics agree.
    with bins_path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["count"]) > 0]
    fig, ax = plt.subplots(figsize=(6.6, 6.6), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="0.45", label="Ideal calibration", zorder=1)
    ax.scatter(
        [float(r["mean_confidence"]) for r in rows],
        [float(r["accuracy"]) for r in rows],
        s=58, color="#1764a0", edgecolors="white", linewidths=0.6,
        label="Non-empty confidence bins", zorder=3,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence",
           ylabel="Empirical accuracy",
           title=f"{summary['condition'].capitalize()}: raw confidence reliability")
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.97, 0.04,
            f"ECE = {summary['ece']:.4f}\n15 equal-width bins\n"
            f"{summary['num_images']} images; valid pixels only",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate(condition):
    if condition not in EXPECTED_IMAGES:
        raise ValueError(f"Unsupported condition: {condition}")
    image_paths = sorted((RGB_ROOT / condition / "val").rglob("*_rgb_anon.png"))
    if len(image_paths) != EXPECTED_IMAGES[condition]:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES[condition]} {condition} validation images; found {len(image_paths)}")
    for path in image_paths:
        if not find_gt_path(path, condition).is_file():
            raise FileNotFoundError(find_gt_path(path, condition))

    baseline_path = Path("results") / f"{condition}_val" / "confusion_matrix.npy"
    baseline_confusion = np.load(baseline_path)
    baseline_accuracy = np.trace(baseline_confusion) / baseline_confusion.sum()
    output_dir = OUTPUT_ROOT / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 3A - Raw Confidence Calibration ({condition.capitalize()})", flush=True)
    print(f"Images: {len(image_paths)} | Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading frozen SegFormer-B0...", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if device.type == "cuda":
        with torch.inference_mode():
            dummy = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
            for _ in range(10):
                model(pixel_values=dummy)
            del dummy
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    accumulator = CalibrationAccumulator()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    start = time.perf_counter()
    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(find_gt_path(image_path, condition)) as source:
            gt = np.array(source)
        if gt.shape != (1080, 1920) or image.size != (1920, 1080):
            raise ValueError(f"Unexpected RGB/GT resolution: {image_path}")
        valid = (gt != IGNORE_INDEX) & (gt >= 0) & (gt < NUM_CLASSES)
        if np.any(~valid & (gt != IGNORE_INDEX)):
            raise ValueError(f"Unexpected trainIds outside 0-18 and 255: {image_path}")
        inputs = processor(images=image, return_tensors="pt",
                           size={"height": INPUT_HEIGHT, "width": INPUT_WIDTH})
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            if outputs.logits.shape != (1, NUM_CLASSES, 128, 256):
                raise ValueError(f"Unexpected raw logits shape: {outputs.logits.shape}")
            # Frozen protocol: bilinear upsampling of LOGITS before softmax.
            logits = F.interpolate(outputs.logits, size=gt.shape,
                                   mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities.max(dim=1).values[0].cpu().numpy()
            del probabilities
            nll_sum = valid_pixel_nll_sum(logits, gt)
        accumulator.update(confidence[valid], prediction[valid] == gt[valid], nll_sum)
        update_confusion_matrix(confusion, prediction, gt)
        del outputs, logits, inputs, confidence, prediction, image, gt, valid
        if index == 1 or index % 10 == 0 or index == len(image_paths):
            print(f"[{index:3d}/{len(image_paths)}] processed | "
                  f"elapsed={time.perf_counter() - start:.1f}s", flush=True)

    summary, bins = accumulator.finalize(condition, index)
    assert int(confusion.sum()) == summary["num_valid_pixels"]
    assert int(np.trace(confusion)) == summary["num_correct_pixels"]
    difference = summary["pixel_accuracy"] - baseline_accuracy
    baseline_pass = abs(difference) <= ACCURACY_TOLERANCE
    summary.update({
        "stage1_pixel_accuracy": baseline_accuracy,
        "pixel_accuracy_difference": difference,
        "pixel_accuracy_tolerance": ACCURACY_TOLERANCE,
        "baseline_accuracy_check": "PASS" if baseline_pass else "WARNING",
        "stage1_valid_pixel_count_matches": int(confusion.sum()) == int(baseline_confusion.sum()),
        "stage1_confusion_matrix_exact_match": bool(np.array_equal(confusion, baseline_confusion)),
    })
    summary_path = output_dir / "raw_calibration_summary.csv"
    bins_path = output_dir / "confidence_bins.csv"
    diagram_path = output_dir / "reliability_diagram.png"
    write_csv(summary_path, [summary])
    write_csv(bins_path, bins)
    plot_reliability(bins_path, diagram_path, summary)
    lines = [
        f"STAGE 3A {condition.upper()} - RAW CALIBRATION RESULTS",
        *[f"{key}: {value}" for key, value in summary.items()],
        f"Pixel accuracy: {summary['pixel_accuracy'] * 100:.6f}%",
        f"Bin count sum: {sum(r['count'] for r in bins)} (PASS)",
        "ECE range / high-confidence count consistency: PASS",
        "Bins: [lower, upper); final bin includes 1.0. Indices are zero-based.",
        "Empty-bin means, accuracies, error rates and gaps are NaN; ECE contribution is zero.",
        "high_conf_error_rate = P(incorrect | confidence >= 0.90).",
        "fraction_incorrect_that_are_high_conf = P(confidence >= 0.90 | incorrect).",
        "All fractions use valid pixels (trainIds 0-18); trainId 255 is ignored.",
        "ECE is pixel-weighted over the whole subset, not averaged across images.",
        "NLL = mean(-log p(GT)), using stable log_softmax of full-resolution logits; units: nats.",
        "Diagram: unconnected points for non-empty bins, read from confidence_bins.csv.",
        f"Model: {MODEL_NAME}",
        f"Model revision: {getattr(model.config, '_commit_hash', None)}",
        f"Processor class: {type(processor).__name__}",
        f"torch={torch.__version__}; transformers={transformers.__version__}; numpy={np.__version__}",
        "Protocol: 512x1024 preprocessing; float32 model.eval()/torch.inference_mode();",
        "bilinear logits -> 1080x1920, align_corners=False; logits argmax; then softmax confidence.",
        f"Stage 1 reference: {baseline_path}; absolute accuracy tolerance={ACCURACY_TOLERANCE}.",
        "Raw evaluation only: no parameters fitted, no threshold selected, no model modification.",
    ]
    if device.type == "cuda":
        lines.append(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB")
    if not baseline_pass:
        lines.append("WARNING: Pixel accuracy differs from frozen Stage 1 beyond tolerance; investigate before continuing.")
    if not summary["stage1_valid_pixel_count_matches"]:
        lines.append("WARNING: Valid-pixel count differs from frozen Stage 1.")
    text_summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text_summary)
    print("\n" + text_summary)
    print(f"Files saved to: {output_dir}")
    if not baseline_pass or not summary["stage1_valid_pixel_count_matches"]:
        raise RuntimeError("Stage 1 reproducibility check failed; stop before evaluating another condition.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=list(EXPECTED_IMAGES))
    evaluate(parser.parse_args().condition)
