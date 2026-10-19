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

REF_GT_ROOT = Path(
    "data/ACDC/gt_trainval_ref/gt"
)

OUTPUT_ROOT = Path(
    "results/stage2_uncertainty/paired_normal_vs_adverse"
)

NUM_CLASSES = 19
IGNORE_INDEX = 255

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024


# Stage 1 paired mIoU values, rounded to 4 decimals.
# These are used only as a consistency check.
STAGE1_PAIRED_MIOU = {
    "fog": {
        "normal": 0.5760,
        "adverse": 0.5674,
    },
    "rain": {
        "normal": 0.5428,
        "adverse": 0.5075,
    },
    "snow": {
        "normal": 0.5391,
        "adverse": 0.4826,
    },
    "night": {
        "normal": 0.5037,
        "adverse": 0.2237,
    },
}


# ============================================================
# Filename utilities
# ============================================================

def get_frame_key(filename):

    suffixes = [
        "_gt_ref_labelTrainIds.png",
        "_gt_labelTrainIds.png",
        "_rgb_ref_anon.png",
        "_rgb_anon.png",
    ]

    for suffix in suffixes:

        if filename.endswith(suffix):

            return filename[
                :-len(suffix)
            ]

    raise ValueError(
        f"Unknown ACDC filename format: {filename}"
    )


def find_image(
    directory,
    frame_key,
    reference=False
):

    """
    Robustly handle either:
      *_rgb_ref_anon.png
    or
      *_rgb_anon.png
    """

    if reference:

        candidates = [
            directory
            / f"{frame_key}_rgb_ref_anon.png",

            directory
            / f"{frame_key}_rgb_anon.png",
        ]

    else:

        candidates = [
            directory
            / f"{frame_key}_rgb_anon.png"
        ]

    for path in candidates:

        if path.exists():
            return path

    matches = sorted(
        directory.glob(
            f"{frame_key}*rgb*anon.png"
        )
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not uniquely find RGB image for "
        f"{frame_key} in {directory}. "
        f"Matches={matches}"
    )


# ============================================================
# Confusion matrix
# ============================================================

def update_confusion(
    confusion,
    pred,
    gt
):

    valid = (
        (gt != IGNORE_INDEX)
        & (gt >= 0)
        & (gt < NUM_CLASSES)
    )

    gt_valid = (
        gt[valid]
        .astype(np.int64)
    )

    pred_valid = (
        pred[valid]
        .astype(np.int64)
    )

    encoded = (
        NUM_CLASSES * gt_valid
        + pred_valid
    )

    counts = np.bincount(
        encoded,
        minlength=NUM_CLASSES ** 2
    )

    confusion += counts.reshape(
        NUM_CLASSES,
        NUM_CLASSES
    )


def compute_metrics(
    confusion
):

    tp = np.diag(
        confusion
    ).astype(np.float64)

    gt_count = (
        confusion.sum(axis=1)
        .astype(np.float64)
    )

    pred_count = (
        confusion.sum(axis=0)
        .astype(np.float64)
    )

    union = (
        gt_count
        + pred_count
        - tp
    )

    valid_classes = (
        union > 0
    )

    iou = np.full(
        NUM_CLASSES,
        np.nan,
        dtype=np.float64
    )

    iou[valid_classes] = (
        tp[valid_classes]
        / union[valid_classes]
    )

    miou = np.nanmean(
        iou[valid_classes]
    )

    total = confusion.sum()

    accuracy = (
        tp.sum() / total
        if total > 0
        else np.nan
    )

    return (
        miou,
        accuracy
    )


# ============================================================
# Running mean accumulator
# ============================================================

class MeanAccumulator:

    def __init__(self):

        self.total = 0.0
        self.count = 0

    def update(
        self,
        values
    ):

        if values.size == 0:
            return

        self.total += float(
            values.sum()
        )

        self.count += int(
            values.size
        )

    @property
    def mean(self):

        if self.count == 0:
            return np.nan

        return (
            self.total
            / self.count
        )


# ============================================================
# Inference
# ============================================================

def infer_image(
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
        Image.open(
            gt_path
        )
    )

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

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

        # Keep protocol identical to Stage 1 / Stage 2B:
        # resize LOGITS first, then softmax.
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

    valid = (
        gt != IGNORE_INDEX
    )

    correct = (
        (pred == gt)
        & valid
    )

    incorrect = (
        (pred != gt)
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

    accuracy = (
        correct_count
        / valid_count
        if valid_count > 0
        else np.nan
    )

    mean_msp = float(
        msp_uncertainty[
            valid
        ].mean()
    )

    mean_entropy = float(
        entropy[
            valid
        ].mean()
    )

    high_conf_wrong = (
        incorrect
        & (
            max_prob >= 0.90
        )
    )

    high_conf_wrong_count = int(
        high_conf_wrong.sum()
    )

    high_conf_wrong_fraction = (
        high_conf_wrong_count
        / incorrect_count
        if incorrect_count > 0
        else np.nan
    )

    result = {
        "gt":
            gt,

        "pred":
            pred,

        "valid":
            valid,

        "msp_uncertainty":
            msp_uncertainty,

        "entropy":
            entropy,

        "accuracy":
            accuracy,

        "valid_count":
            valid_count,

        "incorrect_count":
            incorrect_count,

        "mean_msp":
            mean_msp,

        "mean_entropy":
            mean_entropy,

        "high_conf_wrong_fraction":
            high_conf_wrong_fraction,
    }

    del (
        outputs,
        logits,
        probabilities,
        max_prob_tensor,
        pred_tensor,
        entropy_tensor,
    )

    return result


# ============================================================
# Main evaluation
# ============================================================

def evaluate(
    condition
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 82)
    print(
        "STAGE 2C — PAIRED NORMAL VS ADVERSE UNCERTAINTY"
    )
    print("=" * 82)

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

    # ========================================================
    # Load model
    # ========================================================

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

    # ========================================================
    # GPU warm-up
    # ========================================================

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

    # ========================================================
    # Find reference GTs
    # ========================================================

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

    if len(ref_gt_paths) == 0:

        raise RuntimeError(
            f"No reference GT found in: "
            f"{ref_gt_dir}"
        )

    print(
        f"\nFound {len(ref_gt_paths)} "
        f"annotated reference images."
    )

    # ========================================================
    # Output
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        / condition
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Aggregates
    # ========================================================

    normal_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    adverse_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    normal_msp_global = (
        MeanAccumulator()
    )

    adverse_msp_global = (
        MeanAccumulator()
    )

    normal_entropy_global = (
        MeanAccumulator()
    )

    adverse_entropy_global = (
        MeanAccumulator()
    )

    rows = []

    # ========================================================
    # Pair loop
    # ========================================================

    for index, ref_gt_path in enumerate(
        ref_gt_paths,
        start=1
    ):

        sequence = (
            ref_gt_path.parent.name
        )

        frame_key = get_frame_key(
            ref_gt_path.name
        )

        # ----------------------------------------------------
        # Reference RGB
        # ----------------------------------------------------

        ref_rgb_dir = (
            RGB_ROOT
            / condition
            / "val_ref"
            / sequence
        )

        ref_rgb_path = find_image(
            ref_rgb_dir,
            frame_key,
            reference=True
        )

        # ----------------------------------------------------
        # Corresponding adverse RGB / GT
        # ----------------------------------------------------

        adverse_rgb_dir = (
            RGB_ROOT
            / condition
            / "val"
            / sequence
        )

        adverse_rgb_path = find_image(
            adverse_rgb_dir,
            frame_key,
            reference=False
        )

        adverse_gt_path = (
            GT_ROOT
            / condition
            / "val"
            / sequence
            / f"{frame_key}_gt_labelTrainIds.png"
        )

        if not adverse_gt_path.exists():

            raise FileNotFoundError(
                f"Missing adverse GT: "
                f"{adverse_gt_path}"
            )

        # ----------------------------------------------------
        # Print first pair for sanity
        # ----------------------------------------------------

        if index == 1:

            print(
                "\nFirst matched pair:"
            )

            print(
                f"  Reference RGB: {ref_rgb_path}"
            )

            print(
                f"  Reference GT:  {ref_gt_path}"
            )

            print(
                f"  Adverse RGB:   {adverse_rgb_path}"
            )

            print(
                f"  Adverse GT:    {adverse_gt_path}"
            )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        normal = infer_image(
            ref_rgb_path,
            ref_gt_path,
            processor,
            model,
            device
        )

        adverse = infer_image(
            adverse_rgb_path,
            adverse_gt_path,
            processor,
            model,
            device
        )

        # ----------------------------------------------------
        # Segmentation consistency
        # ----------------------------------------------------

        update_confusion(
            normal_confusion,
            normal["pred"],
            normal["gt"]
        )

        update_confusion(
            adverse_confusion,
            adverse["pred"],
            adverse["gt"]
        )

        # ----------------------------------------------------
        # Global pixel-weighted uncertainty
        # ----------------------------------------------------

        normal_msp_global.update(
            normal[
                "msp_uncertainty"
            ][
                normal["valid"]
            ]
        )

        adverse_msp_global.update(
            adverse[
                "msp_uncertainty"
            ][
                adverse["valid"]
            ]
        )

        normal_entropy_global.update(
            normal[
                "entropy"
            ][
                normal["valid"]
            ]
        )

        adverse_entropy_global.update(
            adverse[
                "entropy"
            ][
                adverse["valid"]
            ]
        )

        # ----------------------------------------------------
        # Pair-level record
        # ----------------------------------------------------

        rows.append({
            "condition":
                condition,

            "sequence":
                sequence,

            "frame_key":
                frame_key,

            "normal_accuracy":
                normal["accuracy"],

            "adverse_accuracy":
                adverse["accuracy"],

            "accuracy_delta":
                (
                    adverse["accuracy"]
                    - normal["accuracy"]
                ),

            "normal_msp_uncertainty":
                normal["mean_msp"],

            "adverse_msp_uncertainty":
                adverse["mean_msp"],

            "msp_delta":
                (
                    adverse["mean_msp"]
                    - normal["mean_msp"]
                ),

            "normal_entropy":
                normal["mean_entropy"],

            "adverse_entropy":
                adverse["mean_entropy"],

            "entropy_delta":
                (
                    adverse["mean_entropy"]
                    - normal["mean_entropy"]
                ),

            "normal_high_conf_wrong_fraction":
                normal[
                    "high_conf_wrong_fraction"
                ],

            "adverse_high_conf_wrong_fraction":
                adverse[
                    "high_conf_wrong_fraction"
                ],
        })

        if (
            index == 1
            or index % 10 == 0
            or index == len(
                ref_gt_paths
            )
        ):

            print(
                f"[{index:3d}/"
                f"{len(ref_gt_paths)}] "
                f"pairs processed"
            )

    # ========================================================
    # Global segmentation metrics
    # ========================================================

    (
        normal_miou,
        normal_accuracy
    ) = compute_metrics(
        normal_confusion
    )

    (
        adverse_miou,
        adverse_accuracy
    ) = compute_metrics(
        adverse_confusion
    )

    # ========================================================
    # Pair-level statistics
    # ========================================================

    accuracy_deltas = np.array(
        [
            row["accuracy_delta"]
            for row in rows
        ],
        dtype=np.float64
    )

    msp_deltas = np.array(
        [
            row["msp_delta"]
            for row in rows
        ],
        dtype=np.float64
    )

    entropy_deltas = np.array(
        [
            row["entropy_delta"]
            for row in rows
        ],
        dtype=np.float64
    )

    msp_increased = int(
        np.sum(
            msp_deltas > 0
        )
    )

    entropy_increased = int(
        np.sum(
            entropy_deltas > 0
        )
    )

    accuracy_decreased = int(
        np.sum(
            accuracy_deltas < 0
        )
    )

    # ========================================================
    # Stage 1 consistency
    # ========================================================

    expected = (
        STAGE1_PAIRED_MIOU[
            condition
        ]
    )

    normal_pass = (
        round(
            normal_miou,
            4
        )
        == expected["normal"]
    )

    adverse_pass = (
        round(
            adverse_miou,
            4
        )
        == expected["adverse"]
    )

    # ========================================================
    # Print summary
    # ========================================================

    print("\n")
    print("=" * 82)
    print(
        "STAGE 1 PAIRED CONSISTENCY CHECK"
    )
    print("=" * 82)

    print(
        f"Normal mIoU:   "
        f"{normal_miou:.6f} "
        f"(Stage 1 ≈ "
        f"{expected['normal']:.4f})"
    )

    print(
        f"Adverse mIoU:  "
        f"{adverse_miou:.6f} "
        f"(Stage 1 ≈ "
        f"{expected['adverse']:.4f})"
    )

    print(
        f"Normal check:   "
        f"{'PASS' if normal_pass else 'WARNING'}"
    )

    print(
        f"Adverse check:  "
        f"{'PASS' if adverse_pass else 'WARNING'}"
    )

    # --------------------------------------------------------

    print("\n")
    print("=" * 82)
    print(
        "PAIRED PERFORMANCE"
    )
    print("=" * 82)

    print(
        f"Pairs:          "
        f"{len(rows)}"
    )

    print(
        f"Normal mIoU:    "
        f"{normal_miou:.4f}"
    )

    print(
        f"Adverse mIoU:   "
        f"{adverse_miou:.4f}"
    )

    print(
        f"mIoU delta:     "
        f"{adverse_miou-normal_miou:+.4f}"
    )

    print(
        f"Normal Acc:     "
        f"{normal_accuracy:.4f}"
    )

    print(
        f"Adverse Acc:    "
        f"{adverse_accuracy:.4f}"
    )

    print(
        f"Accuracy delta: "
        f"{adverse_accuracy-normal_accuracy:+.4f}"
    )

    print(
        f"Pairs with accuracy decrease: "
        f"{accuracy_decreased}/"
        f"{len(rows)}"
    )

    # --------------------------------------------------------

    print("\n")
    print("=" * 82)
    print(
        "PAIRED MSP UNCERTAINTY"
    )
    print("=" * 82)

    print(
        f"Normal global mean:   "
        f"{normal_msp_global.mean:.4f}"
    )

    print(
        f"Adverse global mean:  "
        f"{adverse_msp_global.mean:.4f}"
    )

    print(
        f"Global delta:         "
        f"{adverse_msp_global.mean-normal_msp_global.mean:+.4f}"
    )

    print(
        f"Mean pair delta:      "
        f"{msp_deltas.mean():+.4f}"
    )

    print(
        f"Median pair delta:    "
        f"{np.median(msp_deltas):+.4f}"
    )

    print(
        f"Pairs with increase:  "
        f"{msp_increased}/"
        f"{len(rows)}"
    )

    # --------------------------------------------------------

    print("\n")
    print("=" * 82)
    print(
        "PAIRED PREDICTIVE ENTROPY"
    )
    print("=" * 82)

    print(
        f"Normal global mean:   "
        f"{normal_entropy_global.mean:.4f}"
    )

    print(
        f"Adverse global mean:  "
        f"{adverse_entropy_global.mean:.4f}"
    )

    print(
        f"Global delta:         "
        f"{adverse_entropy_global.mean-normal_entropy_global.mean:+.4f}"
    )

    print(
        f"Mean pair delta:      "
        f"{entropy_deltas.mean():+.4f}"
    )

    print(
        f"Median pair delta:    "
        f"{np.median(entropy_deltas):+.4f}"
    )

    print(
        f"Pairs with increase:  "
        f"{entropy_increased}/"
        f"{len(rows)}"
    )

    # ========================================================
    # Save pair CSV
    # ========================================================

    pair_csv = (
        output_dir
        / "paired_uncertainty.csv"
    )

    with open(
        pair_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # Save summary CSV
    # ========================================================

    summary = {
        "condition":
            condition,

        "pairs":
            len(rows),

        "normal_miou":
            normal_miou,

        "adverse_miou":
            adverse_miou,

        "miou_delta":
            adverse_miou
            - normal_miou,

        "normal_accuracy":
            normal_accuracy,

        "adverse_accuracy":
            adverse_accuracy,

        "accuracy_delta":
            adverse_accuracy
            - normal_accuracy,

        "normal_msp_mean":
            normal_msp_global.mean,

        "adverse_msp_mean":
            adverse_msp_global.mean,

        "msp_global_delta":
            adverse_msp_global.mean
            - normal_msp_global.mean,

        "msp_pair_mean_delta":
            msp_deltas.mean(),

        "msp_pair_median_delta":
            np.median(
                msp_deltas
            ),

        "msp_pairs_increased":
            msp_increased,

        "normal_entropy_mean":
            normal_entropy_global.mean,

        "adverse_entropy_mean":
            adverse_entropy_global.mean,

        "entropy_global_delta":
            adverse_entropy_global.mean
            - normal_entropy_global.mean,

        "entropy_pair_mean_delta":
            entropy_deltas.mean(),

        "entropy_pair_median_delta":
            np.median(
                entropy_deltas
            ),

        "entropy_pairs_increased":
            entropy_increased,

        "accuracy_pairs_decreased":
            accuracy_decreased,
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

    np.save(
        output_dir
        / "normal_confusion_matrix.npy",
        normal_confusion
    )

    np.save(
        output_dir
        / "adverse_confusion_matrix.npy",
        adverse_confusion
    )

    print("\n")
    print("=" * 82)
    print(
        "FILES SAVED"
    )
    print("=" * 82)

    print(
        pair_csv
    )

    print(
        summary_csv
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
            "night",
        ]
    )

    args = parser.parse_args()

    evaluate(
        args.condition
    )