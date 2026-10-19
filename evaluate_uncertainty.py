import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = (
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
)

RGB_ROOT = Path(
    "data/ACDC/rgb_anon_trainvaltest/rgb_anon"
)

GT_ROOT = Path(
    "data/ACDC/gt_trainval/gt"
)

RESULT_ROOT = Path(
    "results/stage2_uncertainty"
)

NUM_CLASSES = 19
IGNORE_INDEX = 255

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

HIGH_CONFIDENCE_THRESHOLD = 0.90

# Histogram used for approximate median / P90.
# 1000 bins => approximately 0.001 resolution.
NUM_BINS = 1000

HIST_EDGES = np.linspace(
    0.0,
    1.0,
    NUM_BINS + 1
)

# Frozen Stage 1 reference values.
# Stage 2 prediction must reproduce these.
STAGE1_MIOU = {
    "fog": 0.637933,
    "rain": 0.496401,
    "snow": 0.485679,
    "night": 0.223781,
}


# ============================================================
# Confusion matrix
# ============================================================

def update_confusion_matrix(
    confusion,
    pred,
    gt
):

    valid = (
        (gt != IGNORE_INDEX)
        & (gt >= 0)
        & (gt < NUM_CLASSES)
    )

    gt_valid = gt[
        valid
    ].astype(np.int64)

    pred_valid = pred[
        valid
    ].astype(np.int64)

    indices = (
        NUM_CLASSES * gt_valid
        + pred_valid
    )

    counts = np.bincount(
        indices,
        minlength=NUM_CLASSES ** 2
    )

    confusion += counts.reshape(
        NUM_CLASSES,
        NUM_CLASSES
    )


def compute_segmentation_metrics(
    confusion
):

    tp = np.diag(
        confusion
    ).astype(np.float64)

    gt_pixels = confusion.sum(
        axis=1
    ).astype(np.float64)

    pred_pixels = confusion.sum(
        axis=0
    ).astype(np.float64)

    union = (
        gt_pixels
        + pred_pixels
        - tp
    )

    iou = np.full(
        NUM_CLASSES,
        np.nan,
        dtype=np.float64
    )

    valid_classes = (
        union > 0
    )

    iou[
        valid_classes
    ] = (
        tp[valid_classes]
        / union[valid_classes]
    )

    miou = np.nanmean(
        iou[valid_classes]
    )

    total_pixels = (
        confusion.sum()
    )

    pixel_accuracy = (
        tp.sum() / total_pixels
        if total_pixels > 0
        else np.nan
    )

    return (
        iou,
        miou,
        pixel_accuracy
    )


# ============================================================
# GT matching
# ============================================================

def find_gt_path(
    image_path,
    condition
):

    sequence = (
        image_path.parent.name
    )

    gt_name = (
        image_path.name.replace(
            "_rgb_anon.png",
            "_gt_labelTrainIds.png"
        )
    )

    return (
        GT_ROOT
        / condition
        / "val"
        / sequence
        / gt_name
    )


# ============================================================
# Streaming uncertainty statistics
# ============================================================

class UncertaintyAccumulator:

    def __init__(self):

        self.count = 0

        self.sum = 0.0

        self.hist = np.zeros(
            NUM_BINS,
            dtype=np.int64
        )

        self.minimum = np.inf
        self.maximum = -np.inf

    def update(
        self,
        values
    ):

        if values.size == 0:
            return

        values = np.asarray(
            values,
            dtype=np.float64
        )

        # Numerical safety
        values = np.clip(
            values,
            0.0,
            1.0
        )

        self.count += (
            values.size
        )

        self.sum += float(
            values.sum()
        )

        self.minimum = min(
            self.minimum,
            float(values.min())
        )

        self.maximum = max(
            self.maximum,
            float(values.max())
        )

        counts, _ = np.histogram(
            values,
            bins=HIST_EDGES
        )

        self.hist += counts

    @property
    def mean(self):

        if self.count == 0:
            return np.nan

        return (
            self.sum
            / self.count
        )

    def percentile(
        self,
        q
    ):

        """
        Approximate percentile using
        the fixed histogram.
        """

        if self.count == 0:
            return np.nan

        cumulative = np.cumsum(
            self.hist
        )

        target = (
            q * self.count
        )

        index = int(
            np.searchsorted(
                cumulative,
                target,
                side="left"
            )
        )

        index = min(
            index,
            NUM_BINS - 1
        )

        # Use bin midpoint
        return float(
            (
                HIST_EDGES[index]
                + HIST_EDGES[index + 1]
            )
            / 2.0
        )


# ============================================================
# Safe mean
# ============================================================

def safe_mean(values):

    if values.size == 0:
        return np.nan

    return float(
        values.mean()
    )


# ============================================================
# Main evaluation
# ============================================================

def evaluate(
    condition
):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "STAGE 2B — DATASET UNCERTAINTY PROFILING"
    )
    print("=" * 80)

    print(
        f"Condition: {condition}"
    )

    print(
        f"Device:    {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU:       "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading SegFormer-B0..."
    )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            MODEL_NAME
        )
        .to(device)
    )

    model.eval()

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    if device.type == "cuda":

        print(
            "\nWarming up GPU..."
        )

        dummy = torch.randn(
            1,
            3,
            INPUT_HEIGHT,
            INPUT_WIDTH,
            device=device
        )

        with torch.inference_mode():

            for _ in range(10):

                _ = model(
                    pixel_values=dummy
                )

        torch.cuda.synchronize()

        print(
            "Warm-up complete."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    image_root = (
        RGB_ROOT
        / condition
        / "val"
    )

    image_paths = sorted(
        image_root.rglob(
            "*_rgb_anon.png"
        )
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            f"No images found: "
            f"{image_root}"
        )

    print(
        f"\nFound "
        f"{len(image_paths)} "
        f"validation images."
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        RESULT_ROOT
        / condition
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    per_image_csv = (
        output_dir
        / "per_image_uncertainty.csv"
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    msp_correct = (
        UncertaintyAccumulator()
    )

    msp_incorrect = (
        UncertaintyAccumulator()
    )

    entropy_correct = (
        UncertaintyAccumulator()
    )

    entropy_incorrect = (
        UncertaintyAccumulator()
    )

    total_valid_pixels = 0
    total_correct_pixels = 0
    total_incorrect_pixels = 0

    total_high_conf_wrong = 0

    per_image_rows = []

    # ========================================================
    # Evaluation loop
    # ========================================================

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        gt_path = find_gt_path(
            image_path,
            condition
        )

        if not gt_path.exists():

            raise FileNotFoundError(
                f"Missing GT: {gt_path}"
            )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        gt = np.array(
            Image.open(
                gt_path
            )
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        inputs = processor(
            images=image,
            return_tensors="pt",
            size={
                "height": INPUT_HEIGHT,
                "width": INPUT_WIDTH
            }
        )

        inputs = {
            key: value.to(
                device
            )
            for key, value
            in inputs.items()
        }

        # ----------------------------------------------------
        # Inference + uncertainty
        #
        # IMPORTANT:
        # resize logits BEFORE softmax,
        # exactly as Stage 1 protocol.
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                **inputs
            )

            logits = F.interpolate(
                outputs.logits,
                size=gt.shape,
                mode="bilinear",
                align_corners=False
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )
            )

            (
                max_probability_tensor,
                prediction_tensor
            ) = probabilities.max(
                dim=1
            )

            entropy_tensor = (
                -torch.sum(
                    probabilities
                    * torch.log(
                        probabilities.clamp_min(
                            1e-12
                        )
                    ),
                    dim=1
                )
                / math.log(
                    NUM_CLASSES
                )
            )

        # ----------------------------------------------------
        # CPU arrays
        # ----------------------------------------------------

        prediction = (
            prediction_tensor[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        max_probability = (
            max_probability_tensor[0]
            .cpu()
            .numpy()
        )

        entropy_uncertainty = (
            entropy_tensor[0]
            .cpu()
            .numpy()
        )

        msp_uncertainty = (
            1.0
            - max_probability
        )

        # ----------------------------------------------------
        # Masks
        # ----------------------------------------------------

        valid = (
            gt != IGNORE_INDEX
        )

        correct = (
            (prediction == gt)
            & valid
        )

        incorrect = (
            (prediction != gt)
            & valid
        )

        valid_count = int(
            valid.sum()
        )

        correct_count = int(
            correct.sum()
        )

        incorrect_count = int(
            incorrect.sum()
        )

        # ----------------------------------------------------
        # Stage 1 segmentation consistency
        # ----------------------------------------------------

        update_confusion_matrix(
            confusion,
            prediction,
            gt
        )

        # ----------------------------------------------------
        # Dataset uncertainty accumulation
        # ----------------------------------------------------

        msp_correct.update(
            msp_uncertainty[
                correct
            ]
        )

        msp_incorrect.update(
            msp_uncertainty[
                incorrect
            ]
        )

        entropy_correct.update(
            entropy_uncertainty[
                correct
            ]
        )

        entropy_incorrect.update(
            entropy_uncertainty[
                incorrect
            ]
        )

        # ----------------------------------------------------
        # High-confidence incorrect pixels
        # ----------------------------------------------------

        high_conf_wrong = (
            incorrect
            & (
                max_probability
                >= HIGH_CONFIDENCE_THRESHOLD
            )
        )

        high_conf_wrong_count = int(
            high_conf_wrong.sum()
        )

        # ----------------------------------------------------
        # Global counts
        # ----------------------------------------------------

        total_valid_pixels += (
            valid_count
        )

        total_correct_pixels += (
            correct_count
        )

        total_incorrect_pixels += (
            incorrect_count
        )

        total_high_conf_wrong += (
            high_conf_wrong_count
        )

        # ----------------------------------------------------
        # Per-image values
        # ----------------------------------------------------

        image_accuracy = (
            correct_count
            / valid_count
            if valid_count > 0
            else np.nan
        )

        image_error_rate = (
            incorrect_count
            / valid_count
            if valid_count > 0
            else np.nan
        )

        image_high_conf_wrong_fraction = (
            high_conf_wrong_count
            / incorrect_count
            if incorrect_count > 0
            else np.nan
        )

        per_image_rows.append({
            "condition":
                condition,

            "sequence":
                image_path.parent.name,

            "image":
                image_path.name,

            "valid_pixels":
                valid_count,

            "correct_pixels":
                correct_count,

            "incorrect_pixels":
                incorrect_count,

            "pixel_accuracy":
                image_accuracy,

            "error_rate":
                image_error_rate,

            "msp_correct_mean":
                safe_mean(
                    msp_uncertainty[
                        correct
                    ]
                ),

            "msp_incorrect_mean":
                safe_mean(
                    msp_uncertainty[
                        incorrect
                    ]
                ),

            "entropy_correct_mean":
                safe_mean(
                    entropy_uncertainty[
                        correct
                    ]
                ),

            "entropy_incorrect_mean":
                safe_mean(
                    entropy_uncertainty[
                        incorrect
                    ]
                ),

            "high_conf_wrong_fraction":
                image_high_conf_wrong_fraction,
        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or index % 25 == 0
            or index == len(
                image_paths
            )
        ):

            print(
                f"[{index:4d}/"
                f"{len(image_paths)}] "
                f"processed"
            )

        # Explicit cleanup
        del (
            outputs,
            logits,
            probabilities,
            max_probability_tensor,
            prediction_tensor,
            entropy_tensor,
        )

    # ========================================================
    # Segmentation metrics
    # ========================================================

    (
        ious,
        miou,
        pixel_accuracy
    ) = compute_segmentation_metrics(
        confusion
    )

    expected_miou = (
        STAGE1_MIOU[
            condition
        ]
    )

    miou_difference = (
        miou
        - expected_miou
    )

    # ========================================================
    # Dataset uncertainty metrics
    # ========================================================

    error_rate = (
        total_incorrect_pixels
        / total_valid_pixels
    )

    high_conf_wrong_fraction = (
        total_high_conf_wrong
        / total_incorrect_pixels
        if total_incorrect_pixels > 0
        else np.nan
    )

    # ========================================================
    # Print
    # ========================================================

    print("\n")
    print("=" * 80)
    print("STAGE 1 CONSISTENCY CHECK")
    print("=" * 80)

    print(
        f"Stage 2 mIoU:       "
        f"{miou:.6f}"
    )

    print(
        f"Stage 1 mIoU:       "
        f"{expected_miou:.6f}"
    )

    print(
        f"Difference:          "
        f"{miou_difference:+.8f}"
    )

    print(
        f"Pixel accuracy:      "
        f"{pixel_accuracy:.6f}"
    )

    if abs(
        miou_difference
    ) < 1e-5:

        print(
            "Consistency check:  PASS"
        )

    else:

        print(
            "Consistency check:  WARNING"
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("DATASET ERROR PROFILE")
    print("=" * 80)

    print(
        f"Valid pixels:        "
        f"{total_valid_pixels:,}"
    )

    print(
        f"Correct pixels:      "
        f"{total_correct_pixels:,}"
    )

    print(
        f"Incorrect pixels:    "
        f"{total_incorrect_pixels:,}"
    )

    print(
        f"Error rate:          "
        f"{error_rate*100:.2f}%"
    )

    # --------------------------------------------------------
    # MSP
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("MSP UNCERTAINTY")
    print("=" * 80)

    print(
        "Correct pixels:"
    )

    print(
        f"  mean    = "
        f"{msp_correct.mean:.4f}"
    )

    print(
        f"  median  ~= "
        f"{msp_correct.percentile(0.50):.4f}"
    )

    print(
        f"  p90     ~= "
        f"{msp_correct.percentile(0.90):.4f}"
    )

    print(
        "Incorrect pixels:"
    )

    print(
        f"  mean    = "
        f"{msp_incorrect.mean:.4f}"
    )

    print(
        f"  median  ~= "
        f"{msp_incorrect.percentile(0.50):.4f}"
    )

    print(
        f"  p90     ~= "
        f"{msp_incorrect.percentile(0.90):.4f}"
    )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("NORMALIZED ENTROPY")
    print("=" * 80)

    print(
        "Correct pixels:"
    )

    print(
        f"  mean    = "
        f"{entropy_correct.mean:.4f}"
    )

    print(
        f"  median  ~= "
        f"{entropy_correct.percentile(0.50):.4f}"
    )

    print(
        f"  p90     ~= "
        f"{entropy_correct.percentile(0.90):.4f}"
    )

    print(
        "Incorrect pixels:"
    )

    print(
        f"  mean    = "
        f"{entropy_incorrect.mean:.4f}"
    )

    print(
        f"  median  ~= "
        f"{entropy_incorrect.percentile(0.50):.4f}"
    )

    print(
        f"  p90     ~= "
        f"{entropy_incorrect.percentile(0.90):.4f}"
    )

    # --------------------------------------------------------
    # High confidence failures
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("HIGH-CONFIDENCE FAILURES")
    print("=" * 80)

    print(
        f"Confidence threshold: "
        f">= "
        f"{HIGH_CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"High-confidence wrong:"
        f" {total_high_conf_wrong:,}"
    )

    print(
        f"All wrong pixels:     "
        f"{total_incorrect_pixels:,}"
    )

    print(
        f"Fraction:             "
        f"{high_conf_wrong_fraction*100:.2f}%"
    )

    # ========================================================
    # Save per-image CSV
    # ========================================================

    with open(
        per_image_csv,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "condition",
            "sequence",
            "image",
            "valid_pixels",
            "correct_pixels",
            "incorrect_pixels",
            "pixel_accuracy",
            "error_rate",
            "msp_correct_mean",
            "msp_incorrect_mean",
            "entropy_correct_mean",
            "entropy_incorrect_mean",
            "high_conf_wrong_fraction",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            per_image_rows
        )

    # ========================================================
    # Save summary
    # ========================================================

    summary = {
        "condition":
            condition,

        "images":
            len(image_paths),

        "miou":
            miou,

        "stage1_expected_miou":
            expected_miou,

        "miou_difference":
            miou_difference,

        "pixel_accuracy":
            pixel_accuracy,

        "error_rate":
            error_rate,

        "msp_correct_mean":
            msp_correct.mean,

        "msp_correct_median_approx":
            msp_correct.percentile(
                0.50
            ),

        "msp_correct_p90_approx":
            msp_correct.percentile(
                0.90
            ),

        "msp_incorrect_mean":
            msp_incorrect.mean,

        "msp_incorrect_median_approx":
            msp_incorrect.percentile(
                0.50
            ),

        "msp_incorrect_p90_approx":
            msp_incorrect.percentile(
                0.90
            ),

        "entropy_correct_mean":
            entropy_correct.mean,

        "entropy_correct_median_approx":
            entropy_correct.percentile(
                0.50
            ),

        "entropy_correct_p90_approx":
            entropy_correct.percentile(
                0.90
            ),

        "entropy_incorrect_mean":
            entropy_incorrect.mean,

        "entropy_incorrect_median_approx":
            entropy_incorrect.percentile(
                0.50
            ),

        "entropy_incorrect_p90_approx":
            entropy_incorrect.percentile(
                0.90
            ),

        "high_conf_wrong_fraction":
            high_conf_wrong_fraction,

        "high_conf_wrong_pixels":
            total_high_conf_wrong,

        "incorrect_pixels":
            total_incorrect_pixels,

        "valid_pixels":
            total_valid_pixels,
    }

    summary_csv = (
        output_dir
        / "summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary.keys()
            )
        )

        writer.writeheader()

        writer.writerow(
            summary
        )

    # ========================================================
    # Save histograms for later plots / AUROC preparation
    # ========================================================

    histogram_path = (
        output_dir
        / "uncertainty_histograms.npz"
    )

    np.savez(
        histogram_path,

        bin_edges=HIST_EDGES,

        msp_correct=
            msp_correct.hist,

        msp_incorrect=
            msp_incorrect.hist,

        entropy_correct=
            entropy_correct.hist,

        entropy_incorrect=
            entropy_incorrect.hist,
    )

    # ========================================================
    # Save confusion matrix
    # ========================================================

    np.save(
        output_dir
        / "confusion_matrix.npy",
        confusion
    )

    print("\n")
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        per_image_csv
    )

    print(
        summary_csv
    )

    print(
        histogram_path
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "fog",
            "rain",
            "snow",
            "night"
        ]
    )

    args = parser.parse_args()

    evaluate(
        args.condition
    )