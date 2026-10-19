from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

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

OUTPUT_ROOT = Path(
    "results/stage2_uncertainty/failure_audit"
)

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

NUM_CLASSES = 19
IGNORE_INDEX = 255

HIGH_CONFIDENCE_THRESHOLD = 0.90


CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


CITYSCAPES_COLORS = np.array([
    [128,  64, 128],   # road
    [244,  35, 232],   # sidewalk
    [ 70,  70,  70],   # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170,  30],   # traffic light
    [220, 220,   0],   # traffic sign
    [107, 142,  35],   # vegetation
    [152, 251, 152],   # terrain
    [ 70, 130, 180],   # sky
    [220,  20,  60],   # person
    [255,   0,   0],   # rider
    [  0,   0, 142],   # car
    [  0,   0,  70],   # truck
    [  0,  60, 100],   # bus
    [  0,  80, 100],   # train
    [  0,   0, 230],   # motorcycle
    [119,  11,  32],   # bicycle
], dtype=np.uint8)


# ============================================================
# Frozen representative cases
# ============================================================

CASES = [
    {
        "name": "night_sky",
        "condition": "night",
        "sequence": "GOPR0356",
        "frame": "GOPR0356_frame_000363",
        "target_id": 10,
        "target_name": "sky",
    },

    {
        "name": "snow_terrain",
        "condition": "snow",
        "sequence": "GOPR0122",
        "frame": "GOPR0122_frame_000364",
        "target_id": 9,
        "target_name": "terrain",
    },
]


# ============================================================
# Utility
# ============================================================

def colorize(mask):

    h, w = mask.shape

    result = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    for class_id in range(NUM_CLASSES):

        result[
            mask == class_id
        ] = CITYSCAPES_COLORS[
            class_id
        ]

    result[
        mask == IGNORE_INDEX
    ] = [0, 0, 0]

    return result


def uncertainty_stats(
    values
):

    if values.size == 0:

        return {
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
        }

    return {
        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90
                )
            ),
    }


# ============================================================
# Target-class error map
# ============================================================

def make_target_error_map(
    gt,
    pred,
    target_id
):

    valid = (
        gt != IGNORE_INDEX
    )

    gt_target = (
        gt == target_id
    )

    pred_target = (
        pred == target_id
    )

    correct_target = (
        gt_target
        & pred_target
    )

    missed_target = (
        gt_target
        & (~pred_target)
    )

    false_positive_target = (
        (~gt_target)
        & pred_target
        & valid
    )

    result = np.zeros(
        (*gt.shape, 3),
        dtype=np.uint8
    )

    # Green = correctly predicted target class
    result[
        correct_target
    ] = [0, 255, 0]

    # Red = GT target missed
    result[
        missed_target
    ] = [255, 0, 0]

    # Blue = falsely predicted as target
    result[
        false_positive_target
    ] = [0, 0, 255]

    return (
        result,
        correct_target,
        missed_target,
        false_positive_target
    )


# ============================================================
# Main inference for one case
# ============================================================

def process_case(
    case,
    processor,
    model,
    device
):

    condition = case[
        "condition"
    ]

    sequence = case[
        "sequence"
    ]

    frame = case[
        "frame"
    ]

    target_id = case[
        "target_id"
    ]

    target_name = case[
        "target_name"
    ]

    image_path = (
        RGB_ROOT
        / condition
        / "val"
        / sequence
        / f"{frame}_rgb_anon.png"
    )

    gt_path = (
        GT_ROOT
        / condition
        / "val"
        / sequence
        / f"{frame}_gt_labelTrainIds.png"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            image_path
        )

    if not gt_path.exists():

        raise FileNotFoundError(
            gt_path
        )

    print("\n")
    print("=" * 82)

    print(
        f"CASE: {case['name']}"
    )

    print("=" * 82)

    print(
        f"Image:  {image_path}"
    )

    print(
        f"GT:     {gt_path}"
    )

    print(
        f"Target: {target_name} "
        f"(class {target_id})"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    gt = np.array(
        Image.open(
            gt_path
        )
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt",
        size={
            "height": INPUT_HEIGHT,
            "width": INPUT_WIDTH,
        }
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    # --------------------------------------------------------
    # Inference
    #
    # Frozen Stage 1 / Stage 2 protocol:
    # resize logits BEFORE softmax.
    # --------------------------------------------------------

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

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        (
            max_prob_tensor,
            pred_tensor
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

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    pred = (
        pred_tensor[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    max_prob = (
        max_prob_tensor[0]
        .cpu()
        .numpy()
    )

    entropy = (
        entropy_tensor[0]
        .cpu()
        .numpy()
    )

    msp_uncertainty = (
        1.0
        - max_prob
    )

    # --------------------------------------------------------
    # Target error masks
    # --------------------------------------------------------

    (
        target_error_map,
        correct_target,
        missed_target,
        false_positive_target
    ) = make_target_error_map(
        gt,
        pred,
        target_id
    )

    gt_target = (
        gt == target_id
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    target_pixels = int(
        gt_target.sum()
    )

    correct_target_pixels = int(
        correct_target.sum()
    )

    missed_target_pixels = int(
        missed_target.sum()
    )

    false_positive_pixels = int(
        false_positive_target.sum()
    )

    target_recall = (
        correct_target_pixels
        / target_pixels
        if target_pixels > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Uncertainty statistics
    # --------------------------------------------------------

    msp_correct_stats = (
        uncertainty_stats(
            msp_uncertainty[
                correct_target
            ]
        )
    )

    msp_missed_stats = (
        uncertainty_stats(
            msp_uncertainty[
                missed_target
            ]
        )
    )

    entropy_correct_stats = (
        uncertainty_stats(
            entropy[
                correct_target
            ]
        )
    )

    entropy_missed_stats = (
        uncertainty_stats(
            entropy[
                missed_target
            ]
        )
    )

    # --------------------------------------------------------
    # High-confidence target failures
    # --------------------------------------------------------

    high_conf_missed = (
        missed_target
        & (
            max_prob
            >= HIGH_CONFIDENCE_THRESHOLD
        )
    )

    high_conf_missed_count = int(
        high_conf_missed.sum()
    )

    high_conf_missed_fraction = (
        high_conf_missed_count
        / missed_target_pixels
        if missed_target_pixels > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Where did missed target pixels go?
    # --------------------------------------------------------

    missed_predictions = (
        pred[
            missed_target
        ]
    )

    confusion_counts = np.bincount(
        missed_predictions,
        minlength=NUM_CLASSES
    )

    confusion_rows = []

    if missed_target_pixels > 0:

        order = np.argsort(
            confusion_counts
        )[::-1]

        for class_id in order:

            count = int(
                confusion_counts[
                    class_id
                ]
            )

            if count == 0:
                continue

            fraction = (
                count
                / missed_target_pixels
            )

            confusion_rows.append({
                "predicted_class_id":
                    class_id,

                "predicted_class":
                    CLASS_NAMES[
                        class_id
                    ],

                "pixels":
                    count,

                "fraction":
                    fraction,
            })

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nTarget-class performance")

    print(
        f"GT target pixels:       "
        f"{target_pixels:,}"
    )

    print(
        f"Correct target pixels:  "
        f"{correct_target_pixels:,}"
    )

    print(
        f"Missed target pixels:   "
        f"{missed_target_pixels:,}"
    )

    print(
        f"False-positive target:  "
        f"{false_positive_pixels:,}"
    )

    print(
        f"Target recall:          "
        f"{target_recall:.4f}"
    )

    print("\nMSP uncertainty")

    print(
        f"Correct target mean:    "
        f"{msp_correct_stats['mean']:.4f}"
    )

    print(
        f"Missed target mean:     "
        f"{msp_missed_stats['mean']:.4f}"
    )

    print(
        f"Correct target median:  "
        f"{msp_correct_stats['median']:.4f}"
    )

    print(
        f"Missed target median:   "
        f"{msp_missed_stats['median']:.4f}"
    )

    print("\nPredictive entropy")

    print(
        f"Correct target mean:    "
        f"{entropy_correct_stats['mean']:.4f}"
    )

    print(
        f"Missed target mean:     "
        f"{entropy_missed_stats['mean']:.4f}"
    )

    print(
        f"Correct target median:  "
        f"{entropy_correct_stats['median']:.4f}"
    )

    print(
        f"Missed target median:   "
        f"{entropy_missed_stats['median']:.4f}"
    )

    print("\nHigh-confidence missed target")

    print(
        f"Confidence >= 0.90:     "
        f"{high_conf_missed_count:,} / "
        f"{missed_target_pixels:,}"
    )

    print(
        f"Fraction:               "
        f"{high_conf_missed_fraction*100:.2f}%"
    )

    print("\nTop misclassification destinations")

    for row in confusion_rows[
        :5
    ]:

        print(
            f"  {row['predicted_class']:<16}"
            f"{row['pixels']:>12,} "
            f"({row['fraction']*100:6.2f}%)"
        )

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        / case["name"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Visualization
    # ========================================================

    pred_color = colorize(
        pred
    )

    gt_color = colorize(
        gt
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(20, 11)
    )

    # RGB
    axes[0, 0].imshow(
        image
    )

    axes[0, 0].set_title(
        f"ACDC Input — "
        f"{condition.capitalize()}"
    )

    axes[0, 0].axis(
        "off"
    )

    # Prediction
    axes[0, 1].imshow(
        pred_color
    )

    axes[0, 1].set_title(
        "SegFormer-B0 Prediction"
    )

    axes[0, 1].axis(
        "off"
    )

    # GT
    axes[0, 2].imshow(
        gt_color
    )

    axes[0, 2].set_title(
        "Ground Truth"
    )

    axes[0, 2].axis(
        "off"
    )

    # Target error
    axes[1, 0].imshow(
        target_error_map
    )

    axes[1, 0].set_title(
        f"{target_name.capitalize()} Error Map\n"
        "Green=Correct, Red=Missed, Blue=False Positive"
    )

    axes[1, 0].axis(
        "off"
    )

    # MSP
    msp_plot = axes[1, 1].imshow(
        msp_uncertainty,
        cmap="inferno",
        vmin=0,
        vmax=1
    )

    axes[1, 1].set_title(
        "MSP Uncertainty"
    )

    axes[1, 1].axis(
        "off"
    )

    fig.colorbar(
        msp_plot,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04
    )

    # Entropy
    entropy_plot = axes[1, 2].imshow(
        entropy,
        cmap="inferno",
        vmin=0,
        vmax=1
    )

    axes[1, 2].set_title(
        "Normalized Predictive Entropy"
    )

    axes[1, 2].axis(
        "off"
    )

    fig.colorbar(
        entropy_plot,
        ax=axes[1, 2],
        fraction=0.046,
        pad=0.04
    )

    fig.suptitle(
        f"Stage 2E — "
        f"{condition.capitalize()} "
        f"{target_name.capitalize()} Failure Audit",
        fontsize=18
    )

    fig.tight_layout()

    figure_path = (
        output_dir
        / "visualization.png"
    )

    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # Save confusion CSV
    # ========================================================

    confusion_csv = (
        output_dir
        / "misclassification_destinations.csv"
    )

    with open(
        confusion_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "predicted_class_id",
                "predicted_class",
                "pixels",
                "fraction",
            ]
        )

        writer.writeheader()

        writer.writerows(
            confusion_rows
        )

    # ========================================================
    # Save text summary
    # ========================================================

    summary_path = (
        output_dir
        / "summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        f.write(
            f"STAGE 2E — {case['name']}\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Condition: {condition}\n"
        )

        f.write(
            f"Target class: "
            f"{target_name} "
            f"(id={target_id})\n\n"
        )

        f.write(
            "Target-class performance\n"
        )

        f.write(
            "------------------------\n"
        )

        f.write(
            f"GT target pixels: "
            f"{target_pixels}\n"
        )

        f.write(
            f"Correct target pixels: "
            f"{correct_target_pixels}\n"
        )

        f.write(
            f"Missed target pixels: "
            f"{missed_target_pixels}\n"
        )

        f.write(
            f"False-positive target pixels: "
            f"{false_positive_pixels}\n"
        )

        f.write(
            f"Target recall: "
            f"{target_recall:.6f}\n\n"
        )

        f.write(
            "MSP uncertainty\n"
        )

        f.write(
            "---------------\n"
        )

        f.write(
            f"Correct target mean: "
            f"{msp_correct_stats['mean']:.6f}\n"
        )

        f.write(
            f"Correct target median: "
            f"{msp_correct_stats['median']:.6f}\n"
        )

        f.write(
            f"Correct target p90: "
            f"{msp_correct_stats['p90']:.6f}\n"
        )

        f.write(
            f"Missed target mean: "
            f"{msp_missed_stats['mean']:.6f}\n"
        )

        f.write(
            f"Missed target median: "
            f"{msp_missed_stats['median']:.6f}\n"
        )

        f.write(
            f"Missed target p90: "
            f"{msp_missed_stats['p90']:.6f}\n\n"
        )

        f.write(
            "Predictive entropy\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            f"Correct target mean: "
            f"{entropy_correct_stats['mean']:.6f}\n"
        )

        f.write(
            f"Correct target median: "
            f"{entropy_correct_stats['median']:.6f}\n"
        )

        f.write(
            f"Correct target p90: "
            f"{entropy_correct_stats['p90']:.6f}\n"
        )

        f.write(
            f"Missed target mean: "
            f"{entropy_missed_stats['mean']:.6f}\n"
        )

        f.write(
            f"Missed target median: "
            f"{entropy_missed_stats['median']:.6f}\n"
        )

        f.write(
            f"Missed target p90: "
            f"{entropy_missed_stats['p90']:.6f}\n\n"
        )

        f.write(
            "High-confidence missed target\n"
        )

        f.write(
            "-----------------------------\n"
        )

        f.write(
            f"Threshold: "
            f"{HIGH_CONFIDENCE_THRESHOLD:.2f}\n"
        )

        f.write(
            f"High-confidence missed pixels: "
            f"{high_conf_missed_count}\n"
        )

        f.write(
            f"All missed target pixels: "
            f"{missed_target_pixels}\n"
        )

        f.write(
            f"Fraction: "
            f"{high_conf_missed_fraction:.6f}\n\n"
        )

        f.write(
            "Top misclassification destinations\n"
        )

        f.write(
            "----------------------------------\n"
        )

        for row in confusion_rows[
            :10
        ]:

            f.write(
                f"{row['predicted_class']:<16} "
                f"{row['pixels']:>10} "
                f"{row['fraction']*100:7.2f}%\n"
            )

    # ========================================================
    # Combined result row
    # ========================================================

    result_row = {
        "case":
            case["name"],

        "condition":
            condition,

        "target_class":
            target_name,

        "target_id":
            target_id,

        "target_pixels":
            target_pixels,

        "correct_target_pixels":
            correct_target_pixels,

        "missed_target_pixels":
            missed_target_pixels,

        "target_recall":
            target_recall,

        "false_positive_target_pixels":
            false_positive_pixels,

        "msp_correct_target_mean":
            msp_correct_stats["mean"],

        "msp_missed_target_mean":
            msp_missed_stats["mean"],

        "entropy_correct_target_mean":
            entropy_correct_stats["mean"],

        "entropy_missed_target_mean":
            entropy_missed_stats["mean"],

        "high_conf_missed_fraction":
            high_conf_missed_fraction,
    }

    print(
        f"\nSaved to: {output_dir}"
    )

    return result_row


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 82)
    print(
        "STAGE 2E — REPRESENTATIVE FAILURE AUDIT"
    )
    print("=" * 82)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
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
            "Warming up GPU..."
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
    # Run cases
    # --------------------------------------------------------

    results = []

    for case in CASES:

        result = process_case(
            case,
            processor,
            model,
            device
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Combined CSV
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    combined_csv = (
        OUTPUT_ROOT
        / "stage2e_summary.csv"
    )

    with open(
        combined_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print("\n")
    print("=" * 82)
    print(
        "STAGE 2E COMPLETE"
    )
    print("=" * 82)

    print(
        f"Combined summary: "
        f"{combined_csv}"
    )


if __name__ == "__main__":

    main()