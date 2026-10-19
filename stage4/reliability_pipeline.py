"""Ground-truth-free SegFormer inference with a user-supplied reliability threshold.

Canonical segmentation always uses unscaled full-resolution logit argmax.
Accepted means uncertainty <= threshold; it is not a guarantee of correctness.
No dataset discovery, label loading, fitting, or operating-point selection occurs here.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
MODEL_REVISION = "7a91500a7086b805eeb868719ea5542a3e0bdeb3"
EXPECTED_TEMPERATURE = 2.124186626456894  # Integrity check only; JSON supplies T.
TEMPERATURE_PATH = (Path(__file__).resolve().parents[1] /
                    "results/stage3_calibration/stage3b/fitted_temperature.json")
METHODS = ("raw_msp", "calibrated_msp", "raw_entropy", "calibrated_entropy")
NUM_CLASSES = 19


def load_frozen_temperature(path=TEMPERATURE_PATH):
    """Read and validate the existing Stage 3B artifact; never fit a value."""
    path = Path(path).resolve()
    contents = path.read_bytes()
    artifact = json.loads(contents)
    temperature = artifact["temperature"]
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0
            or temperature != EXPECTED_TEMPERATURE):
        raise ValueError("Temperature does not match the frozen Stage 3B value")
    if (artifact.get("model") != MODEL_NAME
            or artifact.get("model_revision") != MODEL_REVISION
            or artifact.get("validation_used") is not False
            or artifact.get("converged") is not True):
        raise ValueError("Frozen temperature provenance does not match this pipeline")
    return float(temperature), hashlib.sha256(contents).hexdigest()


def validate_threshold(threshold):
    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError("Threshold must be a finite number in [0, 1]")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number in [0, 1]")
    return threshold


def apply_reliability_threshold(uncertainty, threshold):
    """Return a boolean acceptance mask; equality at the boundary is accepted."""
    threshold = validate_threshold(threshold)
    values = np.asarray(uncertainty)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Uncertainty must be a finite two-dimensional map")
    if values.min() < -1e-7 or values.max() > 1 + 1e-7:
        raise ValueError("Uncertainty is outside [0, 1] beyond numerical tolerance")
    return values <= threshold


def load_rgb(image):
    """Accept a PIL image or an arbitrary RGB-image path, without GT conventions."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as source:
        return source.convert("RGB")


class ReliabilityPipeline:
    """Reusable single-image API; threshold is mandatory, with no chosen default.

    predict() returns CPU NumPy arrays of shape (height, width): segmentation
    uint8 class IDs 0..18; uncertainty/confidence float32 (raw) or float64
    (calibrated); masks bool. reliability_mask and accepted_mask are aliases.
    confidence always means maximum class probability, including entropy modes.
    infer_logits() + predict_from_logits() allow multiple modes without rerunning
    the network. The caller owns/release any returned full-resolution logits.
    """

    def __init__(self, *, threshold, uncertainty_method="calibrated_msp", device=None,
                 temperature_path=TEMPERATURE_PATH, local_files_only=False):
        self.threshold = validate_threshold(threshold)
        self._validate_method(uncertainty_method)
        self.uncertainty_method = uncertainty_method
        self.temperature_path = Path(temperature_path).resolve()
        self.temperature, self.temperature_sha256 = load_frozen_temperature(self.temperature_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Supported devices are cuda and cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        kwargs = dict(revision=MODEL_REVISION, local_files_only=local_files_only)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME, **kwargs)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, **kwargs)
        if (self.model.config._commit_hash != MODEL_REVISION
                or type(self.processor).__name__ != "SegformerImageProcessor"):
            raise RuntimeError("Model revision or processor differs from the frozen protocol")
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _validate_method(method):
        if method not in METHODS:
            raise ValueError(f"Uncertainty method must be one of {METHODS}")

    @staticmethod
    def _validate_logits(logits):
        if (logits.ndim != 4 or logits.shape[:2] != (1, NUM_CLASSES)
                or min(logits.shape[2:]) < 1 or logits.dtype != torch.float32):
            raise ValueError("Expected float32 full-resolution logits shaped (1, 19, H, W)")
        if not torch.isfinite(logits).all().item():
            raise ValueError("Non-finite logits")

    @torch.inference_mode()
    def infer_logits(self, image):
        """Return float32 (1,19,H,W) logits on the selected device, at input size."""
        rgb = load_rgb(image)
        width, height = rgb.size
        inputs = self.processor(images=rgb, return_tensors="pt",
                                size={"height": 512, "width": 1024})
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self.model.eval()
        native = self.model(**inputs).logits
        if native.shape != (1, NUM_CLASSES, 128, 256) or native.dtype != torch.float32:
            raise RuntimeError("Unexpected native SegFormer logits")
        return F.interpolate(native, size=(height, width), mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def compute_uncertainty(self, logits, method=None):
        """Return uncertainty and maximum probability as CPU NumPy maps.

        Frozen precision: raw float32; float64 scaling/softmax for calibrated
        modes after float32 interpolation. Entropy retains Stage 2/3's 1e-12
        probability clamp inside log and normalization by log(19).
        """
        method = self.uncertainty_method if method is None else method
        self._validate_method(method)
        self._validate_logits(logits)
        scaled = logits.to(torch.float64) / self.temperature if method.startswith("calibrated_") else logits
        probs = torch.softmax(scaled, dim=1)
        confidence = probs.max(dim=1).values
        if method.endswith("msp"):
            uncertainty = 1.0 - confidence
        else:
            uncertainty = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1) / math.log(NUM_CLASSES)
        return uncertainty[0].cpu().numpy(), confidence[0].cpu().numpy()

    @torch.inference_mode()
    def predict_from_logits(self, logits, *, uncertainty_method=None):
        """Apply uncertainty and the supplied threshold to infer_logits() output."""
        method = self.uncertainty_method if uncertainty_method is None else uncertainty_method
        uncertainty, confidence = self.compute_uncertainty(logits, method)
        segmentation = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        accepted = apply_reliability_threshold(uncertainty, self.threshold)
        rejected = ~accepted
        height, width = segmentation.shape
        accepted_pixels = int(accepted.sum())
        metadata = dict(
            model=MODEL_NAME, model_revision=MODEL_REVISION,
            temperature=self.temperature, applied_temperature=self.temperature if method.startswith("calibrated_") else 1.0,
            temperature_artifact=str(self.temperature_path), temperature_artifact_sha256=self.temperature_sha256,
            uncertainty_method=method, threshold=self.threshold,
            threshold_source="user supplied; no operating-point selection",
            original_image_size={"height": height, "width": width},
            total_pixels=height * width, accepted_pixels=accepted_pixels,
            rejected_pixels=int(rejected.sum()), coverage=accepted_pixels / (height * width),
            uncertainty_min=float(uncertainty.min()), uncertainty_max=float(uncertainty.max()),
            uncertainty_mean=float(uncertainty.mean(dtype=np.float64)),
            device=str(self.device), ground_truth_required=False,
            decision_rule="accepted iff uncertainty <= threshold",
            segmentation_definition="argmax of unscaled full-resolution logits",
            precision="float32 inference/interpolation; raw probabilities float32; calibrated probabilities float64",
        )
        return dict(segmentation=segmentation, uncertainty=uncertainty,
                    reliability_mask=accepted, accepted_mask=accepted, rejected_mask=rejected,
                    confidence=confidence, metadata=metadata)

    def predict(self, image):
        """Infer a new RGB image without ground truth or environmental labels."""
        return self.predict_from_logits(self.infer_logits(image))
