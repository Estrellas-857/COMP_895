import torch
from PIL import Image

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

# --------------------------------------------------
# 1. Check GPU
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. Load SegFormer
# --------------------------------------------------
print("\nLoading SegFormer...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")
print("Number of classes:", model.config.num_labels)


# --------------------------------------------------
# 3. Create a dummy image
# --------------------------------------------------
image = Image.new(
    "RGB",
    (1024, 512),
    color=(128, 128, 128)
)

inputs = processor(
    images=image,
    return_tensors="pt",
    size={"height": 512, "width": 1024}
)

print("Processed input shape:", inputs["pixel_values"].shape)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# --------------------------------------------------
# 4. Run inference
# --------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

print("\nInference successful.")
print("Logits shape:", logits.shape)

print("\nCityscapes labels:")
for idx, label in model.config.id2label.items():
    print(idx, label)