import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from stage1.dataset import RGB_ROOT, GT_ROOT


# ============================================================
# Configuration
# ============================================================


MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"


RESULT_ROOT = Path("results")

NUM_CLASSES = 19
IGNORE_INDEX = 255

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024


# ============================================================
# Confusion matrix
# ============================================================

def update_confusion_matrix(confusion, pred, gt):
    """
    Update 19x19 confusion matrix.

    Rows = GT class
    Columns = predicted class
    """

    valid = (
        (gt != IGNORE_INDEX)
        & (gt >= 0)
        & (gt < NUM_CLASSES)
    )

    gt_valid = gt[valid].astype(np.int64)
    pred_valid = pred[valid].astype(np.int64)

    indices = NUM_CLASSES * gt_valid + pred_valid

    counts = np.bincount(
        indices,
        minlength=NUM_CLASSES ** 2
    )

    confusion += counts.reshape(
        NUM_CLASSES,
        NUM_CLASSES
    )


def compute_metrics(confusion):
    """
    Compute dataset-level IoU and pixel accuracy.
    """

    true_positive = np.diag(confusion)

    gt_pixels = confusion.sum(axis=1)
    pred_pixels = confusion.sum(axis=0)

    union = gt_pixels + pred_pixels - true_positive

    iou = np.full(NUM_CLASSES, np.nan)

    valid_classes = union > 0

    iou[valid_classes] = (
        true_positive[valid_classes]
        / union[valid_classes]
    )

    miou = np.nanmean(iou)

    total_correct = true_positive.sum()
    total_pixels = confusion.sum()

    pixel_accuracy = (
        total_correct / total_pixels
        if total_pixels > 0
        else np.nan
    )

    return iou, miou, pixel_accuracy


# ============================================================
# Match RGB image with GT
# ============================================================

def find_gt_path(image_path, condition):
    """
    Example:

    GOPR0476_frame_001019_rgb_anon.png

    becomes:

    GOPR0476_frame_001019_gt_labelTrainIds.png
    """

    sequence = image_path.parent.name

    gt_name = image_path.name.replace(
        "_rgb_anon.png",
        "_gt_labelTrainIds.png"
    )

    gt_path = (
        GT_ROOT
        / condition
        / "val"
        / sequence
        / gt_name
    )

    return gt_path


# ============================================================
# Main evaluation
# ============================================================

def evaluate(condition):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("ACDC SegFormer Baseline Evaluation")
    print("=" * 70)

    print(f"Condition: {condition}")
    print(f"Device:    {device}")

    if torch.cuda.is_available():
        print(
            f"GPU:       "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading SegFormer-B0...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    print("Model loaded.")

    # --------------------------------------------------------
    # GPU warm-up
    # --------------------------------------------------------

    if device.type == "cuda":
        print("\nWarming up GPU...")

        dummy = torch.randn(
            1,
            3,
            INPUT_HEIGHT,
            INPUT_WIDTH,
            device=device
        )

        with torch.inference_mode():
            for _ in range(10):
                _ = model(pixel_values=dummy)

        torch.cuda.synchronize()

        print("Warm-up complete.")

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_root = (
        RGB_ROOT
        / condition
        / "val"
    )

    image_paths = sorted(
        image_root.rglob("*_rgb_anon.png")
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found under {image_root}"
        )

    print(
        f"\nFound {len(image_paths)} "
        f"{condition} validation images."
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        RESULT_ROOT
        / f"{condition}_val"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        output_dir
        / "per_image_results.csv"
    )

    # --------------------------------------------------------
    # Evaluation state
    # --------------------------------------------------------

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    total_latency = 0.0
    processed = 0
    missing_gt = 0

    per_image_rows = []

    # --------------------------------------------------------
    # Loop
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
            print(
                f"[WARNING] Missing GT: {gt_path}"
            )

            missing_gt += 1
            continue

        # ----------------------------------------------------
        # Load image and GT
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        gt = np.array(
            Image.open(gt_path)
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
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        # ----------------------------------------------------
        # GPU inference timing
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            outputs = model(**inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latency = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Resize logits to GT resolution
        # ----------------------------------------------------

        logits = outputs.logits

        logits = F.interpolate(
            logits,
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

        # ----------------------------------------------------
        # Update global confusion matrix
        # ----------------------------------------------------

        update_confusion_matrix(
            confusion,
            pred,
            gt
        )

        # ----------------------------------------------------
        # Per-image pixel accuracy
        # ----------------------------------------------------

        valid = gt != IGNORE_INDEX

        valid_count = valid.sum()

        if valid_count > 0:
            image_pixel_acc = (
                (pred[valid] == gt[valid]).sum()
                / valid_count
            )
        else:
            image_pixel_acc = np.nan

        total_latency += latency
        processed += 1

        per_image_rows.append({
            "image": image_path.name,
            "sequence": image_path.parent.name,
            "condition": condition,
            "pixel_accuracy": image_pixel_acc,
            "latency_ms": latency * 1000.0
        })

        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        if (
            index == 1
            or index % 25 == 0
            or index == len(image_paths)
        ):
            print(
                f"[{index:4d}/{len(image_paths)}] "
                f"processed={processed:4d} | "
                f"latency={latency*1000:.2f} ms"
            )

    # ========================================================
    # Final metrics
    # ========================================================

    ious, miou, pixel_accuracy = (
        compute_metrics(confusion)
    )

    gt_pixels = confusion.sum(axis=1)

    avg_latency_ms = (
        total_latency
        / processed
        * 1000.0
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Condition:       {condition}")
    print(f"Images found:    {len(image_paths)}")
    print(f"Images evaluated:{processed}")
    print(f"Missing GT:      {missing_gt}")

    print(
        f"\nPixel accuracy:  "
        f"{pixel_accuracy:.4f}"
    )

    print(
        f"mIoU:            "
        f"{miou:.4f}"
    )

    print(
        f"Average latency: "
        f"{avg_latency_ms:.2f} ms/image"
    )

    # --------------------------------------------------------
    # Per-class IoU
    # --------------------------------------------------------

    print("\nPer-class IoU and GT pixels:")
    print("-" * 65)

    for class_id in range(NUM_CLASSES):

        class_name = model.config.id2label[class_id]
        support = gt_pixels[class_id]

        if np.isnan(ious[class_id]):
            value = "N/A"
        else:
            value = f"{ious[class_id]:.4f}"

        print(
            f"{class_id:2d} "
            f"{class_name:15s} "
            f"IoU={value:8s} "
            f"GT pixels={support}"
        )


    # ========================================================
    # Save CSV
    # ========================================================

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "sequence",
                "condition",
                "pixel_accuracy",
                "latency_ms"
            ]
        )

        writer.writeheader()
        writer.writerows(
            per_image_rows
        )

    # ========================================================
    # Save summary
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
            f"Condition: {condition}\n"
        )

        f.write(
            f"Images evaluated: "
            f"{processed}\n"
        )

        f.write(
            f"Pixel accuracy: "
            f"{pixel_accuracy:.6f}\n"
        )

        f.write(
            f"mIoU: "
            f"{miou:.6f}\n"
        )

        f.write(
            f"Average latency ms: "
            f"{avg_latency_ms:.6f}\n"
        )

        f.write(
            "\nPer-class IoU:\n"
        )

        for class_id in range(
            NUM_CLASSES
        ):

            class_name = (
                model.config.id2label[
                    class_id
                ]
            )

            f.write(
                f"{class_id:2d} "
                f"{class_name:15s} "
                f"{ious[class_id]}\n"
            )

    # Also save confusion matrix
    np.save(
        output_dir
        / "confusion_matrix.npy",
        confusion
    )

    print(
        f"\nResults saved to: "
        f"{output_dir}"
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

    evaluate(args.condition)