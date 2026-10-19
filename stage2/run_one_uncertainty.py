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

IMAGE_PATH = (
    "data/ACDC/rgb_anon_trainvaltest/rgb_anon/"
    "night/val/GOPR0356/"
    "GOPR0356_frame_000363_rgb_anon.png"
)

GT_PATH = (
    "data/ACDC/gt_trainval/gt/"
    "night/val/GOPR0356/"
    "GOPR0356_frame_000363_gt_labelTrainIds.png"
)

OUTPUT_PATH = (
    "results/stage2_uncertainty/"
    "night_sky_uncertainty_example.png"
)

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

NUM_CLASSES = 19
IGNORE_INDEX = 255


# ============================================================
# Cityscapes palette
# ============================================================

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
        ] = CITYSCAPES_COLORS[class_id]

    result[
        mask == IGNORE_INDEX
    ] = [0, 0, 0]

    return result


def make_error_map(pred, gt):

    """
    Black = correct / ignored
    Red   = incorrect prediction
    """

    valid = (
        gt != IGNORE_INDEX
    )

    incorrect = (
        (pred != gt)
        & valid
    )

    result = np.zeros(
        (*gt.shape, 3),
        dtype=np.uint8
    )

    result[
        incorrect
    ] = [255, 0, 0]

    return result


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 72)
print("STAGE 2A — SINGLE IMAGE UNCERTAINTY CHECK")
print("=" * 72)

print(
    f"Device: {device}"
)

if device.type == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# Load model
# ============================================================

print("\nLoading SegFormer-B0...")

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


# ============================================================
# Load image and GT
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

gt = np.array(
    Image.open(
        GT_PATH
    )
)

print(
    f"\nImage size: "
    f"{image.size}"
)

print(
    f"GT shape: "
    f"{gt.shape}"
)


# ============================================================
# Preprocess
# ============================================================

inputs = processor(
    images=image,
    return_tensors="pt",
    size={
        "height": INPUT_HEIGHT,
        "width": INPUT_WIDTH
    }
)

print(
    f"Network input: "
    f"{inputs['pixel_values'].shape}"
)

inputs = {
    key: value.to(device)
    for key, value
    in inputs.items()
}


# ============================================================
# Inference
# ============================================================

with torch.inference_mode():

    outputs = model(
        **inputs
    )

raw_logits = (
    outputs.logits
)

print(
    f"Raw logits: "
    f"{raw_logits.shape}"
)


# ============================================================
# IMPORTANT:
# Resize LOGITS first
#
# Keep this identical to Stage 1.
# ============================================================

upsampled_logits = F.interpolate(
    raw_logits,
    size=gt.shape,
    mode="bilinear",
    align_corners=False
)

print(
    f"Upsampled logits: "
    f"{upsampled_logits.shape}"
)


# ============================================================
# Softmax probabilities
# ============================================================

probabilities = torch.softmax(
    upsampled_logits,
    dim=1
)


# ============================================================
# Prediction
# ============================================================

max_probabilities, prediction = (
    probabilities.max(
        dim=1
    )
)

prediction = (
    prediction[0]
    .cpu()
    .numpy()
    .astype(np.uint8)
)

max_probability = (
    max_probabilities[0]
    .cpu()
    .numpy()
)


# ============================================================
# Method 1:
# MSP uncertainty
#
# MSP itself is confidence.
# We convert it to uncertainty:
#
# U_MSP = 1 - MSP
# ============================================================

msp_uncertainty = (
    1.0
    - max_probability
)


# ============================================================
# Method 2:
# Normalized predictive entropy
#
# H = -sum(p log p) / log(C)
#
# Range approximately [0, 1]
# ============================================================

epsilon = 1e-12

entropy = -torch.sum(
    probabilities
    * torch.log(
        probabilities + epsilon
    ),
    dim=1
)

entropy = (
    entropy
    / np.log(NUM_CLASSES)
)

entropy_uncertainty = (
    entropy[0]
    .cpu()
    .numpy()
)


# ============================================================
# Valid / correct / incorrect masks
# ============================================================

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


# ============================================================
# Basic segmentation statistics
# ============================================================

num_valid = int(
    valid.sum()
)

num_correct = int(
    correct.sum()
)

num_incorrect = int(
    incorrect.sum()
)

pixel_accuracy = (
    num_correct
    / num_valid
)


# ============================================================
# Uncertainty statistics
# ============================================================

def summarize_uncertainty(
    uncertainty,
    mask
):

    values = uncertainty[
        mask
    ]

    return {
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "p90": float(
            np.percentile(
                values,
                90
            )
        ),
    }


msp_correct = (
    summarize_uncertainty(
        msp_uncertainty,
        correct
    )
)

msp_incorrect = (
    summarize_uncertainty(
        msp_uncertainty,
        incorrect
    )
)

entropy_correct = (
    summarize_uncertainty(
        entropy_uncertainty,
        correct
    )
)

entropy_incorrect = (
    summarize_uncertainty(
        entropy_uncertainty,
        incorrect
    )
)


# ============================================================
# High-confidence incorrect pixels
# ============================================================

high_confidence_wrong = (
    incorrect
    & (
        max_probability >= 0.90
    )
)

num_high_conf_wrong = int(
    high_confidence_wrong.sum()
)

high_conf_wrong_fraction = (
    num_high_conf_wrong
    / num_incorrect
    if num_incorrect > 0
    else 0.0
)


# ============================================================
# Print results
# ============================================================

print("\n")
print("=" * 72)
print("SEGMENTATION")
print("=" * 72)

print(
    f"Valid pixels:       "
    f"{num_valid:,}"
)

print(
    f"Correct pixels:     "
    f"{num_correct:,}"
)

print(
    f"Incorrect pixels:   "
    f"{num_incorrect:,}"
)

print(
    f"Pixel accuracy:     "
    f"{pixel_accuracy:.4f}"
)


print("\n")
print("=" * 72)
print("MSP UNCERTAINTY")
print("=" * 72)

print(
    "Correct pixels:"
)

print(
    f"  mean   = "
    f"{msp_correct['mean']:.4f}"
)

print(
    f"  median = "
    f"{msp_correct['median']:.4f}"
)

print(
    f"  p90    = "
    f"{msp_correct['p90']:.4f}"
)

print(
    "Incorrect pixels:"
)

print(
    f"  mean   = "
    f"{msp_incorrect['mean']:.4f}"
)

print(
    f"  median = "
    f"{msp_incorrect['median']:.4f}"
)

print(
    f"  p90    = "
    f"{msp_incorrect['p90']:.4f}"
)


print("\n")
print("=" * 72)
print("NORMALIZED ENTROPY")
print("=" * 72)

print(
    "Correct pixels:"
)

print(
    f"  mean   = "
    f"{entropy_correct['mean']:.4f}"
)

print(
    f"  median = "
    f"{entropy_correct['median']:.4f}"
)

print(
    f"  p90    = "
    f"{entropy_correct['p90']:.4f}"
)

print(
    "Incorrect pixels:"
)

print(
    f"  mean   = "
    f"{entropy_incorrect['mean']:.4f}"
)

print(
    f"  median = "
    f"{entropy_incorrect['median']:.4f}"
)

print(
    f"  p90    = "
    f"{entropy_incorrect['p90']:.4f}"
)


print("\n")
print("=" * 72)
print("HIGH-CONFIDENCE FAILURES")
print("=" * 72)

print(
    "Wrong pixels with "
    "softmax confidence >= 0.90:"
)

print(
    f"{num_high_conf_wrong:,} / "
    f"{num_incorrect:,}"
)

print(
    f"Fraction: "
    f"{high_conf_wrong_fraction*100:.2f}%"
)


# ============================================================
# Visualization
# ============================================================

pred_color = colorize(
    prediction
)

gt_color = colorize(
    gt
)

error_map = make_error_map(
    prediction,
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
    "ACDC Input — Night"
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


# Error map
axes[1, 0].imshow(
    error_map
)

axes[1, 0].set_title(
    "Prediction Error Map\n"
    "Red = Incorrect"
)

axes[1, 0].axis(
    "off"
)


# MSP uncertainty
msp_plot = axes[1, 1].imshow(
    msp_uncertainty,
    vmin=0,
    vmax=1,
    cmap="inferno"
)

axes[1, 1].set_title(
    "MSP Uncertainty\n"
    "1 - max softmax probability"
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


# Entropy uncertainty
entropy_plot = axes[1, 2].imshow(
    entropy_uncertainty,
    vmin=0,
    vmax=1,
    cmap="inferno"
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
    "Stage 2A — Uncertainty on a "
    "Known Nighttime Failure",
    fontsize=18
)

fig.tight_layout()


# ============================================================
# Save
# ============================================================

from pathlib import Path

output_path = Path(
    OUTPUT_PATH
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

fig.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print(
    f"\nSaved figure to: "
    f"{output_path}"
)