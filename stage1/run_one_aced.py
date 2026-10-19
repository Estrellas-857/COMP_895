import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from stage1.dataset import RGB_ROOT, GT_ROOT


# ============================================================
# Configuration
# ============================================================


MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

IMAGE_PATH = RGB_ROOT / "fog/val/GOPR0476/GOPR0476_frame_001019_rgb_anon.png"

GT_PATH = GT_ROOT / "fog/val/GOPR0476/GOPR0476_frame_001019_gt_labelTrainIds.png"

OUTPUT_PATH = "first_acdc_result.png"


# Cityscapes 19-class palette
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


def colorize(mask):
    """
    Convert trainId mask (0-18, 255 ignore) into RGB visualization.
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(19):
        color[mask == class_id] = CITYSCAPES_COLORS[class_id]

    # Ignore pixels = black
    color[mask == 255] = [0, 0, 0]

    return color


def compute_iou(pred, gt, num_classes=19, ignore_index=255):
    """
    Compute per-class IoU and mean IoU for one image.
    """
    valid = gt != ignore_index

    pred = pred[valid]
    gt = gt[valid]

    ious = []

    for class_id in range(num_classes):
        pred_class = pred == class_id
        gt_class = gt == class_id

        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    miou = np.nanmean(ious)

    return ious, miou


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load model
# ============================================================

print("\nLoading model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

print("Model loaded.")


# ============================================================
# Load ACDC image + GT
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")
gt = np.array(Image.open(GT_PATH))

print("\nOriginal image size:", image.size)
print("GT shape:", gt.shape)

print("GT unique IDs:")
print(np.unique(gt))


# ============================================================
# Preprocess
# ============================================================

inputs = processor(
    images=image,
    return_tensors="pt",
    size={
        "height": 512,
        "width": 1024
    }
)

print("Network input shape:", inputs["pixel_values"].shape)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ============================================================
# Inference
# ============================================================

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

print("Raw logits shape:", logits.shape)


# ============================================================
# Resize logits to original GT resolution
# ============================================================

target_height, target_width = gt.shape

upsampled_logits = F.interpolate(
    logits,
    size=(target_height, target_width),
    mode="bilinear",
    align_corners=False
)

prediction = upsampled_logits.argmax(dim=1)[0]

prediction = prediction.cpu().numpy().astype(np.uint8)

print("Final prediction shape:", prediction.shape)
print("Prediction unique IDs:", np.unique(prediction))


# ============================================================
# IoU
# ============================================================

ious, miou = compute_iou(prediction, gt)

print("\nPer-class IoU:")

for class_id, iou in enumerate(ious):
    class_name = model.config.id2label[class_id]

    if np.isnan(iou):
        print(f"{class_id:2d} {class_name:15s}: N/A")
    else:
        print(f"{class_id:2d} {class_name:15s}: {iou:.4f}")

print(f"\nSingle-image mIoU: {miou:.4f}")


# ============================================================
# Visualization
# ============================================================

pred_color = colorize(prediction)
gt_color = colorize(gt)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("ACDC Input — Fog")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pred_color)
plt.title("SegFormer-B0 Prediction")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gt_color)
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

print(f"\nSaved visualization to: {OUTPUT_PATH}")
