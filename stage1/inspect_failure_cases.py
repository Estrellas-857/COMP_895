import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
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

RESULT_ROOT = Path("results")

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

NUM_CLASSES = 19
IGNORE_INDEX = 255


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


# Standard Cityscapes trainId colors
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
# Utilities
# ============================================================

def find_gt_path(image_path, condition):

    sequence = image_path.parent.name

    gt_name = image_path.name.replace(
        "_rgb_anon.png",
        "_gt_labelTrainIds.png"
    )

    return (
        GT_ROOT
        / condition
        / "val"
        / sequence
        / gt_name
    )


def colorize(mask):

    h, w = mask.shape

    color = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    for class_id in range(NUM_CLASSES):
        color[mask == class_id] = (
            CITYSCAPES_COLORS[class_id]
        )

    color[mask == IGNORE_INDEX] = [0, 0, 0]

    return color


# ============================================================
# Per-image target-class metrics
# ============================================================

def compute_target_metrics(
    pred,
    gt,
    target_id
):

    valid = gt != IGNORE_INDEX

    gt_target = (
        (gt == target_id)
        & valid
    )

    pred_target = (
        (pred == target_id)
        & valid
    )

    gt_pixels = int(
        gt_target.sum()
    )

    pred_pixels = int(
        pred_target.sum()
    )

    tp = int(
        np.logical_and(
            gt_target,
            pred_target
        ).sum()
    )

    fp = int(
        np.logical_and(
            ~gt_target,
            pred_target
        ).sum()
    )

    fn = int(
        np.logical_and(
            gt_target,
            ~pred_target
        ).sum()
    )

    union = (
        tp + fp + fn
    )

    if union > 0:
        iou = tp / union
    else:
        iou = np.nan

    if gt_pixels > 0:
        recall = tp / gt_pixels
    else:
        recall = np.nan

    if pred_pixels > 0:
        precision = tp / pred_pixels
    else:
        precision = np.nan

    return {
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "recall": recall,
        "precision": precision,
    }


# ============================================================
# Find where GT target pixels were predicted
# ============================================================

def target_confusion(
    pred,
    gt,
    target_id,
    top_k=5
):

    mask = (
        gt == target_id
    )

    predicted_ids = pred[mask]

    if predicted_ids.size == 0:
        return []

    counts = np.bincount(
        predicted_ids,
        minlength=NUM_CLASSES
    )

    total = counts.sum()

    order = np.argsort(
        counts
    )[::-1]

    result = []

    for pred_id in order[:top_k]:

        count = int(
            counts[pred_id]
        )

        if count == 0:
            continue

        result.append({
            "pred_id": int(pred_id),
            "pred_name": CLASS_NAMES[pred_id],
            "pixels": count,
            "percent": (
                100.0 * count / total
            )
        })

    return result


# ============================================================
# Target error visualization
# ============================================================

def make_target_error_image(
    pred,
    gt,
    target_id
):

    """
    Black background

    Green:
        GT target and prediction target
        = correct

    Red:
        GT target but prediction is different
        = false negative / miss

    Blue:
        prediction target but GT is different
        = false positive
    """

    h, w = gt.shape

    result = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    valid = gt != IGNORE_INDEX

    gt_target = (
        (gt == target_id)
        & valid
    )

    pred_target = (
        (pred == target_id)
        & valid
    )

    correct = (
        gt_target
        & pred_target
    )

    missed = (
        gt_target
        & ~pred_target
    )

    false_positive = (
        ~gt_target
        & pred_target
        & valid
    )

    # green
    result[correct] = [
        0, 255, 0
    ]

    # red
    result[missed] = [
        255, 0, 0
    ]

    # blue
    result[false_positive] = [
        0, 100, 255
    ]

    return result


# ============================================================
# Visualization
# ============================================================

def save_visualization(
    record,
    target_id,
    output_path
):

    image = Image.open(
        record["image_path"]
    ).convert("RGB")

    gt = record["gt"]
    pred = record["pred"]

    gt_color = colorize(gt)
    pred_color = colorize(pred)

    error_color = (
        make_target_error_image(
            pred,
            gt,
            target_id
        )
    )

    class_name = (
        CLASS_NAMES[target_id]
    )

    metrics = record["metrics"]

    plt.figure(
        figsize=(22, 6)
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    plt.subplot(1, 4, 1)

    plt.imshow(image)

    plt.title(
        "ACDC Input"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    plt.subplot(1, 4, 2)

    plt.imshow(
        pred_color
    )

    plt.title(
        "SegFormer Prediction"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    plt.subplot(1, 4, 3)

    plt.imshow(
        gt_color
    )

    plt.title(
        "Ground Truth"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Target-class errors
    # --------------------------------------------------------

    plt.subplot(1, 4, 4)

    plt.imshow(
        error_color
    )

    plt.title(
        f"{class_name} Errors\n"
        f"Green=Correct | "
        f"Red=Missed | "
        f"Blue=False Positive"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Whole figure title
    # --------------------------------------------------------

    plt.suptitle(
        f"{record['image_name']}\n"
        f"{class_name} "
        f"IoU={metrics['iou']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"GT pixels={metrics['gt_pixels']:,}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Main analysis
# ============================================================

def inspect(
    condition,
    class_name,
    top_k,
    min_gt_pixels
):

    target_id = CLASS_NAMES.index(
        class_name
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("ACDC FAILURE-CASE INSPECTION")
    print("=" * 80)

    print(
        f"Condition:       {condition}"
    )

    print(
        f"Target class:    "
        f"{class_name} "
        f"(trainId={target_id})"
    )

    print(
        f"Minimum GT px:   "
        f"{min_gt_pixels:,}"
    )

    print(
        f"Top-K failures:  "
        f"{top_k}"
    )

    print(
        f"Device:          "
        f"{device}"
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

    print("Model loaded.")

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

            for _ in range(5):
                _ = model(
                    pixel_values=dummy
                )

        torch.cuda.synchronize()

        print(
            "Warm-up complete."
        )

    # --------------------------------------------------------
    # Find dataset
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

    print(
        f"\nFound "
        f"{len(image_paths)} "
        f"validation images."
    )

    candidates = []

    # --------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        gt_path = find_gt_path(
            image_path,
            condition
        )

        if not gt_path.exists():
            continue

        gt = np.array(
            Image.open(gt_path)
        )

        # ----------------------------------------------------
        # First check GT support
        #
        # This lets us skip inference for images that do not
        # contain enough pixels of the target class.
        # ----------------------------------------------------

        gt_pixels = int(
            (gt == target_id).sum()
        )

        if gt_pixels < min_gt_pixels:
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt",
            size={
                "height": INPUT_HEIGHT,
                "width": INPUT_WIDTH
            }
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

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

        pred = (
            logits.argmax(
                dim=1
            )[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        metrics = (
            compute_target_metrics(
                pred,
                gt,
                target_id
            )
        )

        confusion = (
            target_confusion(
                pred,
                gt,
                target_id,
                top_k=5
            )
        )

        candidates.append({
            "image_name": (
                image_path.name
            ),
            "sequence": (
                image_path.parent.name
            ),
            "image_path": image_path,
            "gt_path": gt_path,
            "gt": gt,
            "pred": pred,
            "metrics": metrics,
            "confusion": confusion,
        })

        if index % 25 == 0:
            print(
                f"[{index:4d}/"
                f"{len(image_paths)}] "
                f"candidate images="
                f"{len(candidates)}"
            )

    # --------------------------------------------------------
    # Check candidates
    # --------------------------------------------------------

    if len(candidates) == 0:

        print(
            "\nNo images contain enough "
            "GT pixels for this class."
        )

        return

    # --------------------------------------------------------
    # Rank:
    #
    # Primary:
    #     lowest IoU
    #
    # Secondary:
    #     largest GT support
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x["metrics"]["iou"],
            -x["metrics"]["gt_pixels"]
        )
    )

    selected = (
        candidates[:top_k]
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        RESULT_ROOT
        / "failure_cases"
        / condition
        / class_name.replace(
            " ",
            "_"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        output_dir
        / "ranked_failures.csv"
    )

    # --------------------------------------------------------
    # Print selected failures
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        f"WORST {len(selected)} "
        f"{condition.upper()} / "
        f"{class_name.upper()} CASES"
    )
    print("=" * 80)

    rows = []

    for rank, record in enumerate(
        selected,
        start=1
    ):

        metrics = (
            record["metrics"]
        )

        print(
            f"\n#{rank}"
        )

        print(
            f"Image:     "
            f"{record['image_name']}"
        )

        print(
            f"Sequence:  "
            f"{record['sequence']}"
        )

        print(
            f"GT pixels: "
            f"{metrics['gt_pixels']:,}"
        )

        print(
            f"IoU:       "
            f"{metrics['iou']:.4f}"
        )

        print(
            f"Recall:    "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            "GT target pixels "
            "predicted as:"
        )

        confusion_strings = []

        for item in record[
            "confusion"
        ]:

            text = (
                f"{item['pred_name']}="
                f"{item['percent']:.2f}%"
            )

            confusion_strings.append(
                text
            )

            print(
                f"  "
                f"{item['pred_name']:<15}"
                f"{item['percent']:>7.2f}% "
                f"({item['pixels']:,} px)"
            )

        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        filename = (
            f"rank_{rank:02d}_"
            f"{Path(record['image_name']).stem}"
            f".png"
        )

        output_path = (
            output_dir
            / filename
        )

        save_visualization(
            record,
            target_id,
            output_path
        )

        print(
            f"Saved: "
            f"{output_path}"
        )

        rows.append({
            "rank": rank,
            "condition": condition,
            "class_name": class_name,
            "image": (
                record["image_name"]
            ),
            "sequence": (
                record["sequence"]
            ),
            "gt_pixels": (
                metrics["gt_pixels"]
            ),
            "pred_pixels": (
                metrics["pred_pixels"]
            ),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "iou": metrics["iou"],
            "recall": metrics["recall"],
            "precision": metrics[
                "precision"
            ],
            "top_gt_confusions": (
                "; ".join(
                    confusion_strings
                )
            ),
        })

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "rank",
            "condition",
            "class_name",
            "image",
            "sequence",
            "gt_pixels",
            "pred_pixels",
            "tp",
            "fp",
            "fn",
            "iou",
            "recall",
            "precision",
            "top_gt_confusions",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Candidates found: "
        f"{len(candidates)}"
    )

    print(
        f"Visualizations:   "
        f"{output_dir}"
    )

    print(
        f"CSV:              "
        f"{csv_path}"
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

    parser.add_argument(
        "--class-name",
        required=True,
        choices=CLASS_NAMES
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--min-gt-pixels",
        type=int,
        default=1000,
        help=(
            "Ignore images with fewer "
            "than this many GT pixels "
            "for the target class."
        )
    )

    args = parser.parse_args()

    inspect(
        condition=args.condition,
        class_name=args.class_name,
        top_k=args.top_k,
        min_gt_pixels=args.min_gt_pixels
    )