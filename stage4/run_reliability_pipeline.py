"""One-image CLI demo. Thresholds are supplied by the caller, never selected here."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from stage4.reliability_pipeline import ReliabilityPipeline, METHODS, load_rgb

# Same Cityscapes trainId palette as the frozen project visualizations.
PALETTE = np.array([
    [128,64,128], [244,35,232], [70,70,70], [102,102,156], [190,153,153],
    [153,153,153], [250,170,30], [220,220,0], [107,142,35], [152,251,152],
    [70,130,180], [220,20,60], [255,0,0], [0,0,142], [0,0,70],
    [0,60,100], [0,80,100], [0,0,230], [119,11,32],
], dtype=np.uint8)


def check_result(result, expected_shape):
    for key in ("segmentation", "uncertainty", "confidence", "reliability_mask", "accepted_mask", "rejected_mask"):
        assert result[key].shape == expected_shape, key
    assert result["segmentation"].dtype == np.uint8
    assert 0 <= result["segmentation"].min() <= result["segmentation"].max() < 19
    for key in ("uncertainty", "confidence"):
        values = result[key]
        assert np.isfinite(values).all()
        assert values.min() >= -1e-7 and values.max() <= 1 + 1e-7
    mask = result["reliability_mask"]
    assert mask.dtype == np.bool_
    assert result["accepted_mask"] is mask
    np.testing.assert_array_equal(result["rejected_mask"], ~mask)
    m = result["metadata"]
    np.testing.assert_array_equal(mask, result["uncertainty"] <= m["threshold"])
    assert m["accepted_pixels"] == int(mask.sum())
    assert m["accepted_pixels"] + m["rejected_pixels"] == mask.size
    assert m["coverage"] == m["accepted_pixels"] / mask.size


def save_outputs(output, rgb, result):
    """Keep canonical numeric outputs separate from display-only colors."""
    output.mkdir(parents=True, exist_ok=True)
    segmentation = result["segmentation"]
    colored = PALETTE[segmentation]
    mask = result["accepted_mask"]
    flagged = colored.copy()
    flagged[~mask] = [255, 0, 255]  # Display only: magenta means rejected.
    rgb.save(output / "input_rgb.png")
    np.save(output / "segmentation.npy", segmentation)
    Image.fromarray(segmentation).save(output / "segmentation_trainIds.png")
    Image.fromarray(colored).save(output / "segmentation.png")
    np.save(output / "uncertainty.npy", result["uncertainty"])
    np.save(output / "confidence.npy", result["confidence"])
    np.save(output / "reliability_mask.npy", mask)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "reliability_mask.png")
    Image.fromarray(flagged).save(output / "accepted_segmentation_visualization.png")
    plt.imsave(output / "uncertainty.png", result["uncertainty"], cmap="magma", vmin=0, vmax=1)
    metadata = result["metadata"]
    metadata["output_definitions"] = {
        "segmentation.npy": "Canonical uint8 trainIds, unchanged at rejected pixels",
        "segmentation_trainIds.png": "Lossless single-channel canonical trainIds 0..18",
        "segmentation.png": "Cityscapes color visualization of all original predictions",
        "uncertainty.npy": "Exact numerical uncertainty values; no display normalization",
        "uncertainty.png": "Magma visualization on fixed [0,1] scale",
        "reliability_mask.png": "White (255) accepted; black (0) rejected",
        "accepted_segmentation_visualization.png": "Display only; magenta flags rejected pixels",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.8), layout="constrained")
    panels = [(rgb, "RGB input"), (colored, "Original segmentation"),
              (result["uncertainty"], "Uncertainty: higher = less reliable"),
              (mask, "Acceptance mask: white accepted"),
              (flagged, "Display only: magenta rejected")]
    for ax, (values, title) in zip(axes.flat, panels):
        if title.startswith("Uncertainty"):
            im = ax.imshow(values, cmap="magma", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, shrink=.7, label="Uncertainty")
        elif title.startswith("Acceptance"):
            ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(values)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    axes[1, 2].axis("off")
    axes[1, 2].text(.03, .85,
        f"{metadata['uncertainty_method']}\nT = {metadata['temperature']:.9f}\n"
        f"User threshold = {metadata['threshold']}\nCoverage = {metadata['coverage']:.2%}\n\n"
        f"{metadata['threshold_purpose']}\nNo ground truth used.\nRejected predictions are not corrected.",
        va="top", fontsize=10, wrap=True)
    fig.suptitle("Stage 4A · ground-truth-free inference mechanism", fontsize=14)
    fig.savefig(output / "smoke_test_overview.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float, help="Explicit user input; no threshold selection")
    parser.add_argument("--uncertainty-method", choices=METHODS, default="calibrated_msp")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage4_reliability/stage4a/smoke_test"))
    parser.add_argument("--local-files-only", action="store_true", help="Load the pinned model from the local HF cache")
    parser.add_argument("--check-all-methods", action="store_true", help="Check four modes on the same logits; no GT")
    parser.add_argument("--threshold-purpose", default="User supplied; not a research-selected operating threshold")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty; existing results will not be overwritten")
    rgb = load_rgb(args.image)
    pipeline = ReliabilityPipeline(threshold=args.threshold, uncertainty_method=args.uncertainty_method,
                                   device=args.device, local_files_only=args.local_files_only)
    logits = pipeline.infer_logits(rgb)
    result = pipeline.predict_from_logits(logits)
    check_result(result, (rgb.height, rgb.width))
    method_checks = {}
    if args.check_all_methods:
        for method in METHODS:
            checked = pipeline.predict_from_logits(logits, uncertainty_method=method)
            check_result(checked, result["segmentation"].shape)
            np.testing.assert_array_equal(checked["segmentation"], result["segmentation"])
            method_checks[method] = dict(passed=True, identical_segmentation=True,
                                        uncertainty_min=checked["metadata"]["uncertainty_min"],
                                        uncertainty_max=checked["metadata"]["uncertainty_max"],
                                        uncertainty_mean=checked["metadata"]["uncertainty_mean"])
            del checked
    with torch.inference_mode():
        scaled_prediction = (logits.to(torch.float64) / pipeline.temperature).argmax(dim=1)[0].cpu().numpy()
    prediction_changes = int(np.count_nonzero(scaled_prediction != result["segmentation"]))
    assert prediction_changes == 0
    result["metadata"].update(input_image=str(args.image.resolve()),
        threshold_purpose=args.threshold_purpose, cross_method_checks=method_checks,
        prediction_changes=prediction_changes, ground_truth_loaded=False,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3 if pipeline.device.type == "cuda" else None)
    save_outputs(args.output_dir, rgb, result)
    print(json.dumps(result["metadata"], indent=2), flush=True)
    print(f"Outputs: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
