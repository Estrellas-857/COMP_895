"""Stage 5C: two preselected failure cases, frozen inference first, GT audit second.

No parameter overrides, fitting, threshold search, or performance benchmarking.
The final Stage 4C implementation supplies every prediction and acceptance mask.
"""
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch

from run_final_reliability import FinalReliabilityPipeline, load_final_config
from run_reliability_pipeline import PALETTE
from inspect_uncertainty_failures import CASES, CLASS_NAMES
from evaluate_condition import find_gt_path
from evaluate_frozen_operating_point import (
    InferenceBeforeGTAudit, image_counts, metric_row, read_csv, write_csv, sha256,
)

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / "results/stage5_final_evaluation/stage5c"
HISTORICAL = PROJECT / "results/stage2_uncertainty/failure_audit"
MAP_KEYS = ("segmentation", "uncertainty", "confidence", "reliability_mask", "rejected_mask")


def ratio(n, d):
    return float(n / d) if d else None


def percent(value):
    return "N/A (zero denominator)" if value is None else f"{value:.6%}"


def write_text(path, content):
    with path.open("x") as handle:
        handle.write(content)


def figure(out, rgb, gt, result, target, correct, missed, row):
    """Display-only colors; canonical numerical segmentation is never masked."""
    pred, accepted = result["segmentation"], result["reliability_mask"]
    gt_colors = np.full((*gt.shape, 3), 45, dtype=np.uint8)
    valid = gt < 19
    gt_colors[valid] = PALETTE[gt[valid]]
    background, green, red, blue = (35, 35, 35), (55, 190, 95), (230, 65, 65), (45, 155, 235)
    target_error = np.full((*gt.shape, 3), background, dtype=np.uint8)
    target_error[correct], target_error[missed] = green, red
    handling = np.full((*gt.shape, 3), background, dtype=np.uint8)
    handling[correct] = green
    handling[missed & ~accepted], handling[missed & accepted] = blue, red
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), layout="constrained")
    views = [(rgb, "RGB input"), (gt_colors, "Ground truth · gray = ignored"),
             (PALETTE[pred], "Canonical prediction · classes unchanged"),
             (result["uncertainty"], "Calibrated MSP · higher = more uncertain"),
             (accepted, "Final mask · white = accepted, black = rejected"),
             (target_error, "Target region · red = missed, green = correct"),
             (handling, "Target errors · blue = rejected, red = still accepted")]
    for ax, (values, title) in zip(axes.flat, views):
        if values is result["uncertainty"]:
            im = ax.imshow(values, cmap="magma", vmin=0, vmax=1)
            bar = fig.colorbar(im, ax=ax, shrink=.8)
            bar.set_label("Uncertainty")
            bar.ax.axhline(row["threshold"], color="cyan", linewidth=1.5)
            bar.ax.text(-.45, row["threshold"], "τ", ha="right", transform=bar.ax.get_yaxis_transform(), color="#126b8c")
        elif values is accepted:
            ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(values)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in (axes[2, 1], axes[2, 2]):
        ax.axis("off")
    axes[2, 1].text(0, .97,
        f"Frozen calibrated-MSP system\nT = {row['temperature']}\nτ = {row['threshold']}\n"
        f"Accept: uncertainty ≤ τ\n\nTarget: {row['target_class']} (trainId {row['target_trainId']})\n"
        f"Target pixels: {row['target_pixels']:,}\nCorrect target pixels: {row['target_correct']:,}\n"
        f"Target errors rejected: {row['target_error_recall']:.2%}\n"
        f"Target errors still accepted: {row['target_error_remaining_fraction']:.2%}\n"
        f"Whole-image error recall: {row['whole_image_error_recall']:.2%}",
        va="top", fontsize=11, linespacing=1.5)
    axes[2, 2].legend(handles=[Patch(color=np.array(c)/255, label=t) for c, t in (
        (background, "Outside GT target region"), (green, "Correct target prediction"),
        (blue, "Incorrect target prediction rejected"), (red, "Incorrect target prediction still accepted"))],
        loc="upper left", frameon=False, fontsize=10)
    axes[2, 2].text(0, .48,
        "Target-rejection map legend above.\n\n"
        "Rejection is abstention.\nThe original class prediction is preserved.\n"
        "No parameter or threshold was changed.\n\n"
        "These are two preselected case studies;\nno population-level claim is implied.",
        va="top", fontsize=11, linespacing=1.5)
    fig.suptitle(f"Stage 5C · {row['condition'].capitalize()} {row['target_class']} systematic-failure audit", fontsize=17)
    fig.savefig(out / "visualization.png", dpi=180)
    plt.close(fig)


def process(case, historical, reference, pipeline, config, audit):
    name, target_id = case["name"], case["target_id"]
    rgb_path, gt_path = PROJECT / reference["rgb_path"], PROJECT / reference["gt_path"]
    assert rgb_path.name == case["frame"] + "_rgb_anon.png"
    assert rgb_path.parent.name == case["sequence"]
    assert gt_path.resolve() == (PROJECT / find_gt_path(rgb_path, case["condition"])).resolve()
    assert gt_path.name == case["frame"] + "_gt_labelTrainIds.png"
    assert int(historical["target_id"]) == target_id
    out = ROOT / name
    out.mkdir()
    maps = out / "numerical_maps"
    maps.mkdir()
    audit.begin(rgb_path, gt_path)
    with Image.open(rgb_path) as source:
        rgb = source.convert("RGB")
    changes = []
    original_infer = pipeline._pipeline.infer_logits

    @torch.inference_mode()
    def observe_logits(image):
        # Read-only instrumentation of the existing call; return the same tensor.
        logits = original_infer(image)
        raw = logits.argmax(dim=1)
        scaled = (logits.to(torch.float64) / config["temperature"]).argmax(dim=1)
        changes.append(int(torch.count_nonzero(raw != scaled).item()))
        return logits

    with patch.object(pipeline._pipeline, "infer_logits", side_effect=observe_logits):
        result = pipeline.predict(rgb)
    assert changes == [0]
    result["metadata"].update(input_image=str(rgb_path), prediction_changes_due_to_temperature=0)
    # Freeze and serialize outputs while the audit hook still prohibits GT opens.
    for key in MAP_KEYS:
        result[key].flags.writeable = False
        np.save(maps / f"{key}.npy", result[key], allow_pickle=False)
    write_text(out / "inference_metadata.json", json.dumps(result["metadata"], indent=2) + "\n")
    frozen_hashes = {str(p.relative_to(out)): sha256(p) for p in sorted(maps.glob("*.npy"))}
    write_text(out / "inference_freeze_record.json", json.dumps(dict(
        outputs_saved_before_gt_open=True, sha256=frozen_hashes), indent=2) + "\n")
    print(f"{name}: final inference outputs saved and frozen; now allowing offline GT audit.", flush=True)
    audit.inference_completed()
    with Image.open(gt_path) as source:
        gt = np.array(source)
    audit.finish()
    pred, uncertainty = result["segmentation"], result["uncertainty"]
    confidence, accepted = result["confidence"], result["reliability_mask"]
    rejected = result["rejected_mask"]
    assert pred.shape == gt.shape == uncertainty.shape == accepted.shape == (rgb.height, rgb.width)
    assert np.isfinite(uncertainty).all() and uncertainty.min() >= 0 and uncertainty.max() <= 1
    np.testing.assert_array_equal(accepted, uncertainty <= config["threshold"])
    np.testing.assert_array_equal(rejected, uncertainty > config["threshold"])
    np.testing.assert_array_equal(uncertainty, 1 - confidence)
    counts = image_counts(pred, accepted, gt)
    for key, value in counts.items():
        assert value == int(reference[key]), f"Stage 5A count mismatch: {name}/{key}"
    whole = metric_row(case["condition"], counts, config, 0)
    for key in ("all_pixel_coverage", "valid_coverage", "baseline_risk", "accepted_risk", "error_recall", "rejection_precision"):
        assert whole[key] == float(reference[key]), f"Stage 5A metric mismatch: {key}"
    target = gt == target_id
    correct, missed = target & (pred == target_id), target & (pred != target_id)
    n, nc, nm = int(target.sum()), int(correct.sum()), int(missed.sum())
    na, nr = int((target & accepted).sum()), int((target & rejected).sum())
    mr, ma = int((missed & rejected).sum()), int((missed & accepted).sum())
    cr, ca = int((correct & rejected).sum()), int((correct & accepted).sum())
    assert nc + nm == n == na + nr and mr + ma == nm and cr + ca == nc
    values, conf = uncertainty[missed], confidence[missed]
    assert values.size == nm and nm > 0
    stats = dict(case=name, count=nm, minimum=float(values.min()), maximum=float(values.max()),
        mean=float(values.mean()), standard_deviation=float(values.std(ddof=0)),
        median=float(np.median(values)), percentile_25=float(np.percentile(values, 25, method="linear")),
        percentile_75=float(np.percentile(values, 75, method="linear")),
        percentile_90=float(np.percentile(values, 90, method="linear")),
        fraction_uncertainty_gt_tau=float(np.mean(values > config["threshold"])),
        fraction_uncertainty_le_tau=float(np.mean(values <= config["threshold"])),
        mean_calibrated_confidence=float(conf.mean()), median_calibrated_confidence=float(np.median(conf)),
        calibrated_high_conf_missed_fraction=float(np.mean(conf >= .90)))
    row = dict(case=name, condition=case["condition"], target_class=case["target_name"], target_trainId=target_id,
        temperature=config["temperature"], threshold=config["threshold"], uncertainty_method=config["uncertainty_method"],
        rgb_path=str(rgb_path.relative_to(PROJECT)), gt_path=str(gt_path.relative_to(PROJECT)),
        target_pixels=n, target_correct=nc, target_missed=nm, target_recall=ratio(nc, n),
        target_accepted_pixels=na, target_rejected_pixels=nr, target_coverage=ratio(na, n),
        target_rejection_fraction=ratio(nr, n), missed_target_rejected=mr, missed_target_accepted=ma,
        target_error_recall=ratio(mr, nm), target_error_remaining_fraction=ratio(ma, nm),
        accepted_target_risk=ratio(ma, na), correct_target_rejected=cr, correct_target_accepted=ca,
        correct_target_rejection_rate=ratio(cr, nc),
        whole_image_all_pixel_coverage=whole["all_pixel_coverage"], whole_image_valid_coverage=whole["valid_coverage"],
        whole_image_baseline_risk=whole["baseline_risk"], whole_image_accepted_risk=whole["accepted_risk"],
        whole_image_risk_reduction_absolute=whole["risk_reduction_absolute"],
        whole_image_error_recall=whole["error_recall"], whole_image_rejection_precision=whole["rejection_precision"],
        target_minus_whole_image_error_recall=ratio(mr, nm) - whole["error_recall"],
        historical_raw_high_conf_missed_fraction=float(historical["high_conf_missed_fraction"]),
        calibrated_high_conf_missed_fraction=stats["calibrated_high_conf_missed_fraction"],
        historical_target_pixels_difference=n-int(historical["target_pixels"]),
        historical_target_correct_difference=nc-int(historical["correct_target_pixels"]),
        historical_target_missed_difference=nm-int(historical["missed_target_pixels"]),
        prediction_changes=0, stage5a_counts_check="PASS")
    assert row["target_error_recall"] == stats["fraction_uncertainty_gt_tau"]
    assert row["target_error_remaining_fraction"] == stats["fraction_uncertainty_le_tau"]
    assert abs(row["target_error_recall"] + row["target_error_remaining_fraction"] - 1) < 1e-15
    old_dest = {int(r["predicted_class_id"]): r for r in read_csv(HISTORICAL / name / "misclassification_destinations.csv")}
    destinations = []
    for class_id, count in enumerate(np.bincount(pred[target], minlength=19)):
        old = old_dest.get(class_id, {"pixels": 0, "fraction": 0})
        destinations.append(dict(predicted_class_id=class_id, predicted_class=CLASS_NAMES[class_id],
            pixels=int(count), fraction=float(count/n), historical_pixels=int(old["pixels"]),
            historical_fraction=float(old["fraction"]), pixels_change=int(count)-int(old["pixels"])))
    assert sum(d["pixels"] for d in destinations) == n
    row["historical_destination_count_absolute_difference"] = sum(abs(d["pixels_change"]) for d in destinations)
    for key, mask in (("target_gt_mask", target), ("target_correct_mask", correct), ("target_missed_mask", missed),
                      ("missed_target_rejected_mask", missed & rejected), ("missed_target_accepted_mask", missed & accepted)):
        np.save(maps / f"{key}.npy", mask, allow_pickle=False)
    for path, digest in frozen_hashes.items():
        assert sha256(out / path) == digest, "Inference outputs changed during GT audit"
    write_csv(out / "stage5c_metrics.csv", [row])
    write_csv(out / "uncertainty_statistics.csv", [stats])
    write_csv(out / "misclassification_destinations.csv", destinations)
    write_csv(out / "whole_image_counts.csv", [counts])
    figure(out, rgb, gt, result, target, correct, missed, row)
    summary = (f"Stage 5C — {name}\nRGB: {rgb_path}\nGT (opened after inference outputs frozen): {gt_path}\n"
        f"calibrated_msp; T={config['temperature']}; tau={config['threshold']}; accept U <= tau.\n"
        f"Target pixels={n:,}; correct={nc:,}; missed={nm:,}; recall={percent(row['target_recall'])}.\n"
        f"Target errors rejected={mr:,} ({percent(row['target_error_recall'])}); still accepted={ma:,} ({percent(row['target_error_remaining_fraction'])}).\n"
        f"Target coverage={percent(row['target_coverage'])}; accepted target risk={percent(row['accepted_target_risk'])}.\n"
        f"Correct-target rejection rate={percent(row['correct_target_rejection_rate'])}.\n"
        f"Missed uncertainty mean={stats['mean']:.10f}; median={stats['median']:.10f}; population std={stats['standard_deviation']:.10f}.\n"
        f"Missed calibrated confidence mean={stats['mean_calibrated_confidence']:.10f}; median={stats['median_calibrated_confidence']:.10f}.\n"
        f"Historical RAW confidence >=0.90 missed fraction={percent(row['historical_raw_high_conf_missed_fraction'])}.\n"
        f"Current CALIBRATED confidence >=0.90 missed fraction={percent(row['calibrated_high_conf_missed_fraction'])}.\n"
        f"Whole-image all-pixel coverage={percent(row['whole_image_all_pixel_coverage'])}; valid coverage={percent(row['whole_image_valid_coverage'])}.\n"
        f"Whole-image baseline risk={percent(row['whole_image_baseline_risk'])}; accepted risk={percent(row['whole_image_accepted_risk'])}; reduction={100*row['whole_image_risk_reduction_absolute']:.6f} percentage points.\n"
        f"Whole-image error recall={percent(row['whole_image_error_recall'])}; rejection precision={percent(row['whole_image_rejection_precision'])}.\n"
        f"Target minus whole-image error recall={100*row['target_minus_whole_image_error_recall']:.6f} percentage points.\n"
        f"Historical target count differences: total={row['historical_target_pixels_difference']}, correct={row['historical_target_correct_difference']}, missed={row['historical_target_missed_difference']}.\n"
        f"Destination count changes from Stage 2E: {sum(abs(d['pixels_change']) for d in destinations)}.\n"
        "Stage 5A per-image counts and context metrics match exactly; temperature prediction changes=0 (all image pixels).\n"
        "Quantiles use NumPy linear interpolation; standard deviation uses ddof=0. Empty CSV field means undefined.\n"
        "Rejection mitigates by abstaining; it does not correct semantic predictions. No fitting, selection, or benchmarking.\n")
    write_text(out / "summary.txt", summary)
    print(summary, flush=True)
    return row


def main():
    assert (ROOT / "protected_files_before.json").is_file(), "Snapshot protected files before running"
    config, config_hash = load_final_config()
    assert config["uncertainty_method"] == "calibrated_msp"
    historical = {r["case"]: r for r in read_csv(HISTORICAL / "stage2e_summary.csv")}
    references = {r["image"]: r for r in read_csv(PROJECT / "results/stage5_final_evaluation/stage5a/per_image_metrics.csv")}
    print(f"Frozen final system: T={config['temperature']}; tau={config['threshold']}; calibrated_msp", flush=True)
    assert torch.cuda.is_available(), "Requested RTX GPU is unavailable"
    pipeline = FinalReliabilityPipeline(device="cuda", local_files_only=True)
    audit = InferenceBeforeGTAudit()
    rows = [process(c, historical[c["name"]], references[c["frame"] + "_rgb_anon.png"], pipeline, config, audit) for c in CASES]
    assert len(rows) == len(audit.events) == 2 and not audit.blocked
    write_csv(ROOT / "stage5c_combined_summary.csv", rows)
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    changed = [p for p, h in manifest.items() if not (PROJECT / p).is_file() or sha256(PROJECT / p) != h]
    assert not changed, f"Protected files changed: {changed}"
    assert load_final_config() == (config, config_hash)
    preservation = f"PASS: all {len(manifest)} pre-existing protected files retain their SHA-256 hashes.\nStages 1–5B results, root source scripts, frozen final wrapper, temperature, threshold, config and train splits unchanged.\nPrior caches hashed as opaque bytes only; no reserved train RGB or GT loaded.\n"
    write_text(ROOT / "preservation_check.txt", preservation)
    audit_record = dict(status="PASS", config_hash=config_hash,
        temperature_artifact_hash=config["temperature_artifact_hash"], threshold_artifact_hash=config["threshold_artifact_hash"],
        rgb_then_gt_events=audit.events, blocked_dataset_opens=audit.blocked,
        prediction_changes_all_pixels=0, frozen_output_hashes_rechecked_after_gt=True,
        stage5a_per_image_counts_match=True, no_fitting=True, no_threshold_selection=True,
        no_reserved_train_data=True, no_benchmarking=True,
        path_resolution="Stage 2E CASES and saved summary identify cases/classes; exact RGB/GT paths resolved from frozen Stage 5A per-image records and cross-checked against original Stage 2E mapping.",
        prediction_observation="Read-only observer checks argmax(z) vs argmax(float64(z)/T) during final wrapper inference; returns the original logits unchanged.")
    write_text(ROOT / "evaluation_audit.txt", json.dumps(audit_record, indent=2) + "\n")
    summary = ("Stage 5C — two preselected systematic failures\n"
        f"Frozen method=calibrated_msp; T={config['temperature']}; tau={config['threshold']}; accept U <= tau.\n\n")
    for r in rows:
        summary += (f"{r['case']}: rejected {percent(r['target_error_recall'])} of target errors; "
            f"{percent(r['target_error_remaining_fraction'])} remain accepted. "
            f"Whole-image error recall {percent(r['whole_image_error_recall'])}; "
            f"target difference {100*r['target_minus_whole_image_error_recall']:.6f} percentage points.\n")
    more = max(rows, key=lambda r: r["target_error_recall"])
    summary += (f"\n{more['case']} has the higher target-error rejection fraction at the same frozen threshold.\n"
        "Both cases retain systematic errors among accepted predictions. Rejection mitigates part of each failure without correcting the segmentation.\n"
        + "".join(f"{r['case']}: accepted target risk={percent(r['accepted_target_risk'])}; target recall={percent(r['target_recall'])}; correct-target rejection rate={percent(r['correct_target_rejection_rate'])}.\n" for r in rows) +
        "Historical raw high-confidence statistics and present calibrated high-confidence statistics use different probabilities; they are labeled separately.\n"
        + "".join(f"{r['case']}: historical target total/correct/missed differences = {r['historical_target_pixels_difference']}/{r['historical_target_correct_difference']}/{r['historical_target_missed_difference']}; destination absolute count difference={r['historical_destination_count_absolute_difference']}.\n" for r in rows) +
        "Whole-image metrics use Stage 5A definitions, valid trainIds 0–18, ignore 255; exact counts match frozen per-image records.\n"
        "This is a targeted two-case audit, not a generalization or safety claim. No fitting, threshold changes or Stage 5D benchmarking occurred.\n\n" + preservation)
    write_text(ROOT / "stage5c_summary.txt", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
