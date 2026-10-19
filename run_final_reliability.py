"""Frozen Stage 4C single-image interface. No threshold/method override or fitting.

    final_pipeline = FinalReliabilityPipeline()
    result = final_pipeline.predict(image_path_or_pil_image)

Inference and acceptance semantics are delegated to the unchanged Stage 4A
ReliabilityPipeline. Stage 3B/4B artifacts supply the parameters; Stage 4C
configuration records and verifies their hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from reliability_pipeline import (
    ReliabilityPipeline, load_rgb, load_frozen_temperature, MODEL_NAME, MODEL_REVISION,
)
from run_reliability_pipeline import PALETTE, check_result

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / "results/stage4_reliability/stage4c"
CONFIG_PATH = ROOT / "final_reliability_config.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_final_config():
    """Read-only verification of the frozen configuration and its source artifacts."""
    record = json.loads((ROOT / "final_config_freeze_record.json").read_text())
    if sha256(CONFIG_PATH) != record["config_sha256"]:
        raise ValueError("Final configuration differs from its frozen hash")
    config = json.loads(CONFIG_PATH.read_text())
    t_path = PROJECT / config["temperature_artifact"]
    tau_path = PROJECT / config["threshold_artifact"]
    temperature, t_hash = load_frozen_temperature(t_path)
    operating_point = json.loads(tau_path.read_text())
    threshold_freeze = json.loads((tau_path.parent / "threshold_freeze_record.json").read_text())
    tau_hash = sha256(tau_path)
    if (t_hash != config["temperature_artifact_hash"]
            or t_hash != operating_point["temperature_artifact_sha256"]
            or tau_hash != config["threshold_artifact_hash"]
            or tau_hash != threshold_freeze["artifact_sha256"]):
        raise ValueError("Frozen temperature or operating-point artifact hash mismatch")
    expected = dict(
        model=MODEL_NAME, model_revision=MODEL_REVISION,
        uncertainty_method="calibrated_msp", temperature=temperature,
        threshold=operating_point["threshold"], acceptance_rule="uncertainty <= threshold",
        selection_target_coverage=operating_point["target_coverage"],
        selection_dataset_description=operating_point["selection_dataset"],
        gt_required_at_inference=False,
    )
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Final configuration does not match the frozen primary system")
    if (operating_point["temperature"] != temperature
            or operating_point["uncertainty_method"] != "calibrated_msp"
            or operating_point["acceptance_rule"] != expected["acceptance_rule"]
            or operating_point["model"] != MODEL_NAME
            or operating_point["model_revision"] != MODEL_REVISION
            or operating_point["gt_used_for_selection"] is not False):
        raise ValueError("Operating-point provenance mismatch")
    for filename, expected_hash in config["implementation_dependencies"].items():
        if sha256(PROJECT / filename) != expected_hash:
            raise ValueError(f"Frozen implementation dependency changed: {filename}")
    if sha256(Path(__file__)) != record["wrapper_sha256"]:
        raise ValueError("Final wrapper differs from the frozen implementation")
    return config, record["config_sha256"]


class FinalReliabilityPipeline:
    """One fixed calibrated-MSP system; supports CPU/CUDA and path/PIL inputs."""

    def __init__(self, *, device=None, local_files_only=False):
        config, self.config_hash = load_final_config()
        self.config = MappingProxyType(config)
        self._pipeline = ReliabilityPipeline(
            threshold=config["threshold"], uncertainty_method=config["uncertainty_method"],
            temperature_path=PROJECT / config["temperature_artifact"],
            device=device, local_files_only=local_files_only,
        )
        if self._pipeline.temperature != config["temperature"]:
            raise ValueError("Stage 4A pipeline loaded a different temperature")

    @torch.inference_mode()
    def predict(self, image):
        config, config_hash = load_final_config()
        if config_hash != self.config_hash:
            raise ValueError("Configuration changed since pipeline creation")
        if (self._pipeline.threshold != config["threshold"]
                or self._pipeline.temperature != config["temperature"]
                or self._pipeline.uncertainty_method != config["uncertainty_method"]):
            raise ValueError("Frozen inference parameters were changed")
        rgb = load_rgb(image)
        logits = self._pipeline.infer_logits(rgb)
        canonical = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        result = self._pipeline.predict_from_logits(logits)
        check_result(result, (rgb.height, rgb.width))
        # Include rejected pixels in this check: masking must not replace class IDs.
        np.testing.assert_array_equal(result["segmentation"], canonical)
        result["metadata"].update(
            T=config["temperature"], tau=config["threshold"],
            threshold_source="Frozen Stage 4B operating-point artifact",
            temperature_artifact_hash=config["temperature_artifact_hash"],
            threshold_artifact=str(PROJECT / config["threshold_artifact"]),
            threshold_artifact_hash=config["threshold_artifact_hash"],
            final_configuration=str(CONFIG_PATH), final_configuration_hash=config_hash,
            threshold_purpose="Frozen shared operating point selected in Stage 4B",
            gt_used=False, ground_truth_loaded=False, canonical_prediction_changes=0,
            input_image=str(Path(image).resolve()) if not isinstance(image, Image.Image) else "PIL.Image",
        )
        return result


def save_final_outputs(output, rgb, result):
    """Serialize canonical arrays separately from display-only rejection colors."""
    output.mkdir(parents=True, exist_ok=False)
    colored = PALETTE[result["segmentation"]]
    flagged = colored.copy()
    flagged[result["rejected_mask"]] = [255, 0, 255]
    rgb.save(output / "input_rgb.png")
    for key in ("segmentation", "uncertainty", "confidence", "reliability_mask", "rejected_mask"):
        np.save(output / f"{key}.npy", result[key], allow_pickle=False)
    Image.fromarray(colored).save(output / "segmentation.png")
    Image.fromarray(result["segmentation"]).save(output / "segmentation_trainIds.png")
    Image.fromarray(result["reliability_mask"].astype(np.uint8)*255).save(output / "reliability_mask.png")
    Image.fromarray(flagged).save(output / "accepted_segmentation_visualization.png")
    plt.imsave(output / "uncertainty.png", result["uncertainty"], cmap="magma", vmin=0, vmax=1)
    metadata = result["metadata"]
    metadata["output_semantics"] = {
        "segmentation": "Original trainIds 0..18, preserved even at rejected pixels",
        "uncertainty": "Numerical float64 calibrated MSP; magma visualization fixed to [0,1]",
        "reliability_mask": "Boolean acceptance; PNG white=accepted, black=rejected",
        "rejected_mask": "Logical inverse of reliability_mask",
        "accepted_segmentation_visualization": "Display only: magenta marks rejected pixels",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    views = [(rgb, "Unlabeled RGB input"), (colored, "Original segmentation"),
             (result["uncertainty"], "Calibrated MSP uncertainty"),
             (result["reliability_mask"], "White = accepted; black = rejected"),
             (flagged, "Display only: magenta = rejected")]
    for ax, (values, title) in zip(axes.flat, views):
        if title == "Calibrated MSP uncertainty":
            im = ax.imshow(values, cmap="magma", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, shrink=.75, label="Uncertainty")
        elif title.startswith("White"):
            ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(values)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    axes[1, 2].axis("off")
    axes[1, 2].text(.02, .9,
        f"Frozen calibrated-MSP system\nT = {metadata['T']!r}\nτ = {metadata['tau']!r}\n"
        f"Coverage = {metadata['coverage']:.6%}\n"
        f"Accepted: {metadata['accepted_pixels']:,}\nRejected: {metadata['rejected_pixels']:,}\n\n"
        "GT-free integration test only.\nNo final performance evaluation.\n"
        "Rejected predictions retain their class IDs.", va="top", fontsize=10)
    fig.suptitle("Stage 4C · final frozen reliability implementation", fontsize=14)
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory; defaults to stage4c/inference_outputs/<image stem>")
    args = parser.parse_args()
    output = args.output_dir or ROOT / "inference_outputs" / args.image.stem
    if output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    pipeline = FinalReliabilityPipeline(device=args.device, local_files_only=args.local_files_only)
    result = pipeline.predict(args.image)
    save_final_outputs(output, load_rgb(args.image), result)
    print(json.dumps(result["metadata"], indent=2), flush=True)
    print(f"Outputs: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
