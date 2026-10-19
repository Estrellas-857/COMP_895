from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from stage1.dataset import RGB_ROOT, GT_ROOT as ADVERSE_GT_ROOT, REF_GT_ROOT


# ============================================================
# Configuration
# ============================================================


MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"


RESULT_ROOT = Path(
    "results/paired_normal_vs_adverse"
)

CONDITIONS = [
    "fog",
    "rain",
    "snow",
    "night"
]

NUM_CLASSES = 19
IGNORE_INDEX = 255

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

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


# ============================================================
# Confusion matrix
# ============================================================

def update_confusion_matrix(confusion, pred, gt):

    valid = (
        (gt != IGNORE_INDEX)
        & (gt >= 0)
        & (gt < NUM_CLASSES)
    )

    gt_valid = gt[valid].astype(np.int64)
    pred_valid = pred[valid].astype(np.int64)

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


def compute_metrics(confusion):

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
        np.nan
    )

    union_valid = union > 0

    iou[union_valid] = (
        tp[union_valid]
        / union[union_valid]
    )

    # Same rule as our original evaluator
    miou_union = np.nanmean(
        iou[union_valid]
    )

    # Sensitivity check:
    # only classes actually present in GT
    gt_valid = gt_pixels > 0

    miou_gt = np.mean(
        iou[gt_valid]
    )

    total_pixels = confusion.sum()

    pixel_accuracy = (
        tp.sum() / total_pixels
        if total_pixels > 0
        else np.nan
    )

    return {
        "iou": iou,
        "miou_union": miou_union,
        "miou_gt": miou_gt,
        "pixel_accuracy": pixel_accuracy,
        "gt_pixels": gt_pixels,
    }


# ============================================================
# Image inference
# ============================================================

def run_inference(
    image_path,
    gt_path,
    processor,
    model,
    device
):

    image = Image.open(
        image_path
    ).convert("RGB")

    gt = np.array(
        Image.open(gt_path)
    )

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

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latency = (
        time.perf_counter()
        - start
    )

    logits = F.interpolate(
        outputs.logits,
        size=gt.shape,
        mode="bilinear",
        align_corners=False
    )

    pred = (
        logits.argmax(dim=1)[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return pred, gt, latency


# ============================================================
# Per-image statistics
# ============================================================

def image_pixel_accuracy(
    pred,
    gt
):

    valid = (
        gt != IGNORE_INDEX
    )

    if valid.sum() == 0:
        return np.nan

    return (
        pred[valid] == gt[valid]
    ).mean()


# ============================================================
# Build paired paths
# ============================================================

def build_pairs(condition):

    ref_gt_dir = (
        REF_GT_ROOT
        / condition
        / "val_ref"
    )

    ref_gt_paths = sorted(
        ref_gt_dir.rglob(
            "*_gt_ref_labelTrainIds.png"
        )
    )

    pairs = []

    for ref_gt_path in ref_gt_paths:

        sequence = (
            ref_gt_path.parent.name
        )

        basename = (
            ref_gt_path.name.replace(
                "_gt_ref_labelTrainIds.png",
                ""
            )
        )

        # Normal reference RGB
        ref_rgb_path = (
            RGB_ROOT
            / condition
            / "val_ref"
            / sequence
            / f"{basename}_rgb_ref_anon.png"
        )

        # Corresponding adverse RGB
        adverse_rgb_path = (
            RGB_ROOT
            / condition
            / "val"
            / sequence
            / f"{basename}_rgb_anon.png"
        )

        # Corresponding adverse GT
        adverse_gt_path = (
            ADVERSE_GT_ROOT
            / condition
            / "val"
            / sequence
            / f"{basename}_gt_labelTrainIds.png"
        )

        missing = []

        if not ref_rgb_path.exists():
            missing.append(
                f"reference RGB: {ref_rgb_path}"
            )

        if not adverse_rgb_path.exists():
            missing.append(
                f"adverse RGB: {adverse_rgb_path}"
            )

        if not adverse_gt_path.exists():
            missing.append(
                f"adverse GT: {adverse_gt_path}"
            )

        if missing:
            print(
                "\n[PAIRING ERROR]"
            )

            for item in missing:
                print(item)

            raise FileNotFoundError(
                "Could not construct the "
                "normal/adverse pair."
            )

        pairs.append({
            "basename": basename,
            "sequence": sequence,
            "ref_rgb": ref_rgb_path,
            "ref_gt": ref_gt_path,
            "adverse_rgb": adverse_rgb_path,
            "adverse_gt": adverse_gt_path,
        })

    return pairs


# ============================================================
# Evaluate one condition
# ============================================================

def evaluate_condition(
    condition,
    processor,
    model,
    device
):

    print("\n")
    print("=" * 80)
    print(
        f"PAIRED EVALUATION: "
        f"{condition.upper()}"
    )
    print("=" * 80)

    pairs = build_pairs(
        condition
    )

    print(
        f"Matched pairs: {len(pairs)}"
    )

    ref_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    adverse_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    ref_latency_total = 0.0
    adverse_latency_total = 0.0

    per_image_rows = []

    for index, pair in enumerate(
        pairs,
        start=1
    ):

        # ----------------------------------------------------
        # Normal reference
        # ----------------------------------------------------

        ref_pred, ref_gt, ref_latency = (
            run_inference(
                pair["ref_rgb"],
                pair["ref_gt"],
                processor,
                model,
                device
            )
        )

        update_confusion_matrix(
            ref_confusion,
            ref_pred,
            ref_gt
        )

        # ----------------------------------------------------
        # Corresponding adverse image
        # ----------------------------------------------------

        adv_pred, adv_gt, adv_latency = (
            run_inference(
                pair["adverse_rgb"],
                pair["adverse_gt"],
                processor,
                model,
                device
            )
        )

        update_confusion_matrix(
            adverse_confusion,
            adv_pred,
            adv_gt
        )

        ref_latency_total += (
            ref_latency
        )

        adverse_latency_total += (
            adv_latency
        )

        ref_acc = (
            image_pixel_accuracy(
                ref_pred,
                ref_gt
            )
        )

        adv_acc = (
            image_pixel_accuracy(
                adv_pred,
                adv_gt
            )
        )

        per_image_rows.append({
            "condition": condition,
            "sequence": pair["sequence"],
            "image": pair["basename"],
            "normal_pixel_accuracy": ref_acc,
            "adverse_pixel_accuracy": adv_acc,
            "pixel_accuracy_change": (
                adv_acc - ref_acc
            ),
            "normal_latency_ms": (
                ref_latency * 1000.0
            ),
            "adverse_latency_ms": (
                adv_latency * 1000.0
            )
        })

        if (
            index == 1
            or index % 10 == 0
            or index == len(pairs)
        ):
            print(
                f"[{index:3d}/{len(pairs)}] "
                f"completed"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    ref_metrics = compute_metrics(
        ref_confusion
    )

    adverse_metrics = compute_metrics(
        adverse_confusion
    )

    ref_latency_ms = (
        ref_latency_total
        / len(pairs)
        * 1000.0
    )

    adverse_latency_ms = (
        adverse_latency_total
        / len(pairs)
        * 1000.0
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"{'':20s}"
        f"{'Normal Ref':>14s}"
        f"{'Adverse':>14s}"
        f"{'Change':>14s}"
    )

    print("-" * 62)

    print(
        f"{'Pixel Accuracy':20s}"
        f"{ref_metrics['pixel_accuracy']:14.4f}"
        f"{adverse_metrics['pixel_accuracy']:14.4f}"
        f"{adverse_metrics['pixel_accuracy'] - ref_metrics['pixel_accuracy']:14.4f}"
    )

    print(
        f"{'mIoU_union':20s}"
        f"{ref_metrics['miou_union']:14.4f}"
        f"{adverse_metrics['miou_union']:14.4f}"
        f"{adverse_metrics['miou_union'] - ref_metrics['miou_union']:14.4f}"
    )

    print(
        f"{'mIoU_GT':20s}"
        f"{ref_metrics['miou_gt']:14.4f}"
        f"{adverse_metrics['miou_gt']:14.4f}"
        f"{adverse_metrics['miou_gt'] - ref_metrics['miou_gt']:14.4f}"
    )

    print(
        f"{'Latency ms':20s}"
        f"{ref_latency_ms:14.2f}"
        f"{adverse_latency_ms:14.2f}"
        f"{adverse_latency_ms - ref_latency_ms:14.2f}"
    )

    return {
        "condition": condition,
        "pairs": len(pairs),

        "normal_pixel_accuracy":
            ref_metrics["pixel_accuracy"],

        "adverse_pixel_accuracy":
            adverse_metrics["pixel_accuracy"],

        "normal_miou_union":
            ref_metrics["miou_union"],

        "adverse_miou_union":
            adverse_metrics["miou_union"],

        "normal_miou_gt":
            ref_metrics["miou_gt"],

        "adverse_miou_gt":
            adverse_metrics["miou_gt"],

        "normal_latency_ms":
            ref_latency_ms,

        "adverse_latency_ms":
            adverse_latency_ms,

        "normal_iou":
            ref_metrics["iou"],

        "adverse_iou":
            adverse_metrics["iou"],

        "normal_confusion":
            ref_confusion,

        "adverse_confusion":
            adverse_confusion,

        "per_image_rows":
            per_image_rows,
    }


# ============================================================
# Main
# ============================================================

def main():

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "ACDC PAIRED NORMAL-vs-ADVERSE EVALUATION"
    )
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load model once
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

            for _ in range(10):
                _ = model(
                    pixel_values=dummy
                )

        torch.cuda.synchronize()

        print(
            "Warm-up complete."
        )

    # --------------------------------------------------------
    # Run all four conditions
    # --------------------------------------------------------

    all_results = []

    all_per_image_rows = []

    for condition in CONDITIONS:

        result = evaluate_condition(
            condition,
            processor,
            model,
            device
        )

        all_results.append(
            result
        )

        all_per_image_rows.extend(
            result["per_image_rows"]
        )

        # Save confusion matrices
        np.save(
            RESULT_ROOT
            / f"{condition}_normal_confusion.npy",
            result["normal_confusion"]
        )

        np.save(
            RESULT_ROOT
            / f"{condition}_adverse_confusion.npy",
            result["adverse_confusion"]
        )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n")
    print("=" * 94)
    print(
        "FINAL PAIRED NORMAL-vs-ADVERSE SUMMARY"
    )
    print("=" * 94)

    print(
        f"{'Condition':<12}"
        f"{'Pairs':>8}"
        f"{'Normal mIoU':>15}"
        f"{'Adverse mIoU':>16}"
        f"{'Delta':>12}"
        f"{'Normal Acc':>14}"
        f"{'Adverse Acc':>14}"
    )

    print("-" * 94)

    for result in all_results:

        delta = (
            result["adverse_miou_union"]
            - result["normal_miou_union"]
        )

        print(
            f"{result['condition']:<12}"
            f"{result['pairs']:>8}"
            f"{result['normal_miou_union']:>15.4f}"
            f"{result['adverse_miou_union']:>16.4f}"
            f"{delta:>12.4f}"
            f"{result['normal_pixel_accuracy']:>14.4f}"
            f"{result['adverse_pixel_accuracy']:>14.4f}"
        )

    # ========================================================
    # Save condition summary CSV
    # ========================================================

    summary_csv = (
        RESULT_ROOT
        / "paired_condition_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "condition",
            "pairs",

            "normal_pixel_accuracy",
            "adverse_pixel_accuracy",
            "pixel_accuracy_delta",

            "normal_miou_union",
            "adverse_miou_union",
            "miou_union_delta",

            "normal_miou_gt",
            "adverse_miou_gt",
            "miou_gt_delta",

            "normal_latency_ms",
            "adverse_latency_ms",
        ])

        for result in all_results:

            writer.writerow([
                result["condition"],
                result["pairs"],

                result[
                    "normal_pixel_accuracy"
                ],

                result[
                    "adverse_pixel_accuracy"
                ],

                (
                    result[
                        "adverse_pixel_accuracy"
                    ]
                    -
                    result[
                        "normal_pixel_accuracy"
                    ]
                ),

                result[
                    "normal_miou_union"
                ],

                result[
                    "adverse_miou_union"
                ],

                (
                    result[
                        "adverse_miou_union"
                    ]
                    -
                    result[
                        "normal_miou_union"
                    ]
                ),

                result[
                    "normal_miou_gt"
                ],

                result[
                    "adverse_miou_gt"
                ],

                (
                    result[
                        "adverse_miou_gt"
                    ]
                    -
                    result[
                        "normal_miou_gt"
                    ]
                ),

                result[
                    "normal_latency_ms"
                ],

                result[
                    "adverse_latency_ms"
                ],
            ])

    # ========================================================
    # Save per-image CSV
    # ========================================================

    per_image_csv = (
        RESULT_ROOT
        / "paired_per_image_results.csv"
    )

    with open(
        per_image_csv,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "condition",
            "sequence",
            "image",
            "normal_pixel_accuracy",
            "adverse_pixel_accuracy",
            "pixel_accuracy_change",
            "normal_latency_ms",
            "adverse_latency_ms",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_per_image_rows
        )

    print("\nSaved:")
    print(summary_csv)
    print(per_image_csv)


if __name__ == "__main__":
    main()
