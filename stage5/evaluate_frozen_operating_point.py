"""Stage 5A: evaluation of ONE frozen operating point on ACDC validation.

No method/threshold overrides, fitting, threshold sweeps, targeted audits, or
runtime benchmarking. The final GT-free Stage 4C interface produces each mask
before GT becomes available to the offline evaluator.
"""

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from stage4.run_final_reliability import FinalReliabilityPipeline, load_final_config
from stage1.evaluate_condition import RGB_ROOT, find_gt_path, update_confusion_matrix
from stage4.select_operating_threshold import metrics_from_counts, safe_ratio

from stage1._layout import matches_frozen_hash

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "results/stage5_final_evaluation/stage5a"
EXPECTED = {"fog": 100, "rain": 100, "snow": 100, "night": 106}
COUNT_KEYS = ("num_images", "all_pixels", "all_accepted_pixels", "all_rejected_pixels",
              "valid_pixels", "ignored_pixels", "TP", "FP", "FN", "TN")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class InferenceBeforeGTAudit:
    """Restrict dataset reads to current RGB, then its GT after predict returns."""

    def __init__(self):
        self.dataset = (PROJECT / "data").resolve()
        self.allowed = None
        self.phase = None
        self.events = []
        self.blocked = []
        self.rgb_opened = self.gt_opened = False
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(self.dataset):
            if path != self.allowed:
                self.blocked.append(str(path))
                raise RuntimeError(f"Dataset open blocked during {self.phase}: {path}")
            if self.phase == "rgb_inference":
                self.rgb_opened = True
            elif self.phase == "offline_gt":
                self.gt_opened = True
            else:
                raise RuntimeError("Dataset file opened outside an evaluation step")

    def begin(self, rgb, gt):
        assert self.allowed is None
        self.rgb, self.gt = rgb.resolve(), gt.resolve()
        assert "val" in self.rgb.parts and "train" not in self.rgb.parts
        assert "val" in self.gt.parts and "train" not in self.gt.parts
        self.allowed, self.phase = self.rgb, "rgb_inference"
        self.rgb_opened = self.gt_opened = False

    def inference_completed(self):
        assert self.rgb_opened and not self.gt_opened and self.phase == "rgb_inference"
        self.allowed, self.phase = self.gt, "offline_gt"

    def finish(self):
        assert self.rgb_opened and self.gt_opened
        self.events.append(dict(rgb_path=str(self.rgb.relative_to(PROJECT)),
                                gt_path=str(self.gt.relative_to(PROJECT)),
                                gt_opened_only_after_inference=True))
        self.allowed, self.phase = None, None


def image_counts(prediction, accepted, gt):
    assert prediction.shape == accepted.shape == gt.shape
    assert accepted.dtype == np.bool_
    valid = (gt >= 0) & (gt < 19)
    assert np.all(valid | (gt == 255)), "Unexpected GT label outside 0..18 or 255"
    incorrect = prediction != gt
    c = dict(num_images=1, all_pixels=gt.size,
        all_accepted_pixels=int(accepted.sum()), all_rejected_pixels=int((~accepted).sum()),
        valid_pixels=int(valid.sum()), ignored_pixels=int((~valid).sum()),
        TP=int(np.count_nonzero(valid & incorrect & ~accepted)),
        FP=int(np.count_nonzero(valid & ~incorrect & ~accepted)),
        FN=int(np.count_nonzero(valid & incorrect & accepted)),
        TN=int(np.count_nonzero(valid & ~incorrect & accepted)))
    assert c["valid_pixels"] == c["TP"] + c["FP"] + c["FN"] + c["TN"]
    assert c["all_accepted_pixels"] + c["all_rejected_pixels"] == c["all_pixels"]
    assert c["valid_pixels"] + c["ignored_pixels"] == c["all_pixels"]
    return c


def metric_row(condition, counts, config, changes):
    # Reuse Stage 4B count-based formulas, including explicit zero-denominator handling.
    row = metrics_from_counts(condition, counts, config["threshold"])
    row.update(temperature=config["temperature"], valid_coverage=row["coverage"],
               accepted_risk=row["risk"], incorrect_pixels=row["incorrect_valid_pixels"],
               pixel_accuracy=safe_ratio(counts["TN"] + counts["FP"], counts["valid_pixels"]),
               prediction_changes=changes, baseline_counts_check="PASS")
    return row


def verify_baseline(condition, counts, confusion):
    stage1 = np.load(PROJECT / "results" / f"{condition}_val/confusion_matrix.npy")
    stage3c_root = PROJECT / "results/stage3_calibration/stage3c" / condition
    stage3c = np.load(stage3c_root / "confusion_matrix.npy")
    baseline = read_csv(stage3c_root / "calibrated_summary.csv")[0]
    np.testing.assert_array_equal(confusion, stage1, err_msg=f"{condition}: Stage 1 mismatch")
    np.testing.assert_array_equal(confusion, stage3c, err_msg=f"{condition}: Stage 3C mismatch")
    assert counts["num_images"] == EXPECTED[condition] == int(baseline["num_images"])
    assert counts["valid_pixels"] == int(baseline["num_valid_pixels"]) == int(confusion.sum())
    assert counts["TP"] + counts["FN"] == int(baseline["num_incorrect_pixels"])
    assert counts["FP"] + counts["TN"] == int(baseline["num_correct_pixels"]) == int(np.trace(confusion))
    assert (counts["FP"] + counts["TN"]) / counts["valid_pixels"] == float(baseline["pixel_accuracy"])


def generalization_comparison(validation):
    """Load the frozen train diagnostics only after validation metrics are computed."""
    stage4b = PROJECT / "results/stage4_reliability/stage4b"
    train = read_csv(stage4b / "condition_metrics.csv") + read_csv(stage4b / "stage4b_summary.csv")
    train = {row["condition"]: row for row in train}
    comparisons = []
    fields = ("all_pixel_coverage", "valid_coverage", "baseline_risk", "accepted_risk",
              "risk_reduction_absolute", "risk_reduction_relative", "error_recall", "rejection_precision")
    for val in validation:
        reference = train[val["condition"]]
        assert float(reference["threshold"]) == val["threshold"]
        row = dict(condition=val["condition"], threshold=val["threshold"],
                   comparison="reserved train reliability_selection vs ACDC validation; no retuning")
        for field in fields:
            train_field = {"valid_coverage": "coverage", "accepted_risk": "risk"}.get(field, field)
            old, new = float(reference[train_field]), val[field]
            row[f"selection_{field}"] = old
            row[f"validation_{field}"] = new
            row[f"validation_minus_selection_{field}"] = new - old if new is not None else None
        comparisons.append(row)
    return comparisons


def figures():
    # Plot the saved condition CSV, full percentage scale, separate risk/coverage axes.
    rows = read_csv(ROOT / "condition_metrics.csv")
    x = np.arange(len(rows))
    labels = [row["condition"].capitalize() for row in rows]
    for filename, fields, titles, colors, ylabel, title, note in (
        ("fig1_baseline_vs_accepted_risk.png", ("baseline_risk", "accepted_risk"),
         ("Baseline risk (all valid pixels)", "Accepted-pixel risk"), ("#6d7885", "#167d9a"),
         "Pixel error rate (%)", "ACDC validation · frozen operating-point risk",
         "Lower accepted risk is favorable; rejected predictions are abstentions, not corrected classes."),
        ("fig2_coverage_by_condition.png", ("all_pixel_coverage", "valid_coverage"),
         ("All image pixels", "Valid labeled pixels"), ("#4169a1", "#159b89"),
         "Accepted pixels (%)", "ACDC validation · frozen operating-point coverage",
         "Lower coverage means more abstention. All-pixel and valid-pixel denominators differ."),
    ):
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for offset, field, label, color in zip((-.19, .19), fields, titles, colors):
            values = [100*float(row[field]) for row in rows]
            bars = ax.bar(x+offset, values, width=.36, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 100), ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right" if fields[0] == "all_pixel_coverage" else "upper left")
        fig.text(.5, .035, note, ha="center", fontsize=9)
        fig.text(.5, .075,
                 f"calibrated MSP · frozen T = {float(rows[0]['temperature']):.6f} · frozen τ = {float(rows[0]['threshold']):.6f}",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, 1))
        fig.savefig(ROOT / filename, dpi=300)
        plt.close(fig)


def final_reports(rows, comparisons, audit, config, config_hash):
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    changed = [p for p, h in manifest.items() if not matches_frozen_hash(PROJECT / p, h)]
    assert not changed, f"Protected files changed: {changed}"
    after_config, after_hash = load_final_config()
    assert after_config == config and after_hash == config_hash
    preservation = (f"PASS: all {len(manifest)} pre-existing protected files pass SHA-256 checks with the documented source relocation.\n"
        "Stages 1–4 results, temperature, operating threshold, final config and split artifacts unchanged; sources match the documented relocation.\n"
        "Prior caches were hashed as opaque bytes only; no reserved train data were reevaluated.\n")
    (ROOT / "preservation_check.txt").write_text(preservation)
    assert len(audit.events) == 406 and not audit.blocked
    assert len({e["rgb_path"] for e in audit.events}) == len({e["gt_path"] for e in audit.events}) == 406
    (ROOT / "data_access_audit.json").write_text(json.dumps(dict(
        num_images=406, records=audit.events, blocked_attempts=audit.blocked,
        inference_before_gt_verified=True, training_images_or_gt_opened=False), indent=2) + "\n")
    evaluation_audit = ("Stage 5A: evaluation of the frozen operating point on ACDC validation.\n"
        f"Image counts: {EXPECTED}; total 406.\n"
        f"T={config['temperature']!r}; tau={config['threshold']!r}; method={config['uncertainty_method']}.\n"
        f"Final configuration SHA-256: {config_hash}.\n"
        "FinalReliabilityPipeline.predict called once per image, without GT; only after it returns may current GT be opened.\n"
        "Training-image/GT and out-of-order GT access blocked by dataset file-open guard.\n"
        "Canonical prediction changes = 0 across all image pixels, including rejected and ignored pixels.\n"
        "Exact 19x19 confusion agreement with frozen Stage 1 AND Stage 3C for each condition.\n"
        "Counts conserved for all/valid pixels; TP+FP+FN+TN=valid N. All denominators guarded against zero.\n"
        "Pooled statistics are ratios of pooled pixel counts, not image/condition averages.\n"
        "Threshold, temperature, model, method and preprocessing unchanged. No tuning or override.\n"
        "No threshold sweeps, risk-coverage curves, AURC, targeted failure audits, or latency measurements.\n"
        "Validation was previously analyzed in Stages 1–3; it is not a completely untouched test set.\n")
    (ROOT / "evaluation_audit.txt").write_text(evaluation_audit)
    (ROOT / "implementation_metadata.json").write_text(json.dumps(dict(
        stage="5A", evaluation="evaluation of the frozen operating point on ACDC validation",
        model=config["model"], model_revision=config["model_revision"], temperature=config["temperature"],
        threshold=config["threshold"], uncertainty_method=config["uncertainty_method"],
        config_hash=config_hash, temperature_hash=config["temperature_artifact_hash"],
        threshold_hash=config["threshold_artifact_hash"],
        evaluation_script_sha256=sha256(Path(__file__)), inference_before_gt=True,
        num_images=406, num_valid_pixels=rows[-1]["valid_pixels"],
        num_all_pixels=rows[-1]["all_pixels"], prediction_changes=0,
        exact_stage1_and_stage3c_confusion_match=True, protected_files=len(manifest),
        denominators_guarded=True, parameter_changes=False,
        stage5b_started=False, stage5c_started=False, stage5d_started=False,
    ), indent=2) + "\n")
    lines = ["Stage 5A — evaluation of the frozen operating point on ACDC validation", "",
        evaluation_audit, "Valid-pixel risk: rejected predictions are abstentions and never counted as corrected.",
        "All-pixel coverage uses every input pixel; valid coverage excludes GT trainId=255.",
        "Classes 0..18; GT labelTrainIds; all 406 frozen validation images included.", "",
        "Condition | All coverage | Valid coverage | Baseline risk | Accepted risk | Reduction (pp) | Relative reduction | Error recall | Rejection precision | Incorrect accepted fraction | Correct rejection rate"]
    for row in rows:
        fields = ["all_pixel_coverage", "valid_coverage", "baseline_risk", "accepted_risk",
                  "risk_reduction_absolute", "risk_reduction_relative", "error_recall",
                  "rejection_precision", "incorrect_accepted_fraction", "correct_rejection_rate"]
        lines.append(row["condition"] + " | " + " | ".join(
            f"{100*row[key]:.6f}" if row[key] is not None else "undefined" for key in fields))
    lines += ["", "All table values above are percentages, except absolute risk reduction in percentage points.",
        "Saved CSV proportions are fractions, not percentages. coverage=valid_coverage and risk=accepted_risk aliases.",
        "TP=incorrect rejected; FP=correct rejected; FN=incorrect accepted; TN=correct accepted.",
        "Baseline risk=(TP+FN)/N; accepted risk=FN/(TN+FN); valid coverage=(TN+FN)/N.",
        "Absolute reduction=baseline-risk; relative reduction=(baseline-risk)/baseline.",
        "Error recall=TP/(TP+FN); rejection precision=TP/(TP+FP); incorrect accepted fraction=FN/(TP+FN).",
        "Correct rejection rate=FP/(FP+TN); rejection rate=(TP+FP)/N.",
        "Undefined ratios are empty in CSV and listed in zero_denominators.", "",
        "Pooled reserved-train versus validation comparison (loaded Stage 4B reference; no retuning):",
        json.dumps(comparisons[-1], indent=2), "", preservation,
        "The ACDC validation sets were previously used for diagnostic analyses. No Stage 5A result revises the frozen system.",
        "Single-point results describe the coverage/risk tradeoff here; they do not establish corrected predictions or real-world safety.",
        "No methodological discrepancy encountered if all baseline/audit checks above pass.",
        "STOP: Stage 5B, Stage 5C, and Stage 5D were not begun."]
    (ROOT / "stage5a_summary.txt").write_text("\n".join(lines) + "\n")
    print(evaluation_audit, flush=True)
    print(preservation, flush=True)


def main():
    os.chdir(PROJECT)
    if (ROOT / "stage5a_combined_summary.csv").exists():
        raise RuntimeError("Refusing to overwrite completed Stage 5A results")
    config, config_hash = load_final_config()
    pipeline = FinalReliabilityPipeline(device="cuda", local_files_only=True)
    assert dict(pipeline.config) == config
    inventory = {c: sorted((RGB_ROOT / c / "val").rglob("*_rgb_anon.png")) for c in EXPECTED}
    assert {c: len(p) for c, p in inventory.items()} == EXPECTED
    assert all(find_gt_path(p, c).is_file() for c, paths in inventory.items() for p in paths)
    audit = InferenceBeforeGTAudit()
    aggregate = []
    per_image = []
    counts_by_condition = {}
    total_changes = 0
    done = 0
    confusion_dir = ROOT / "confusion_matrices"
    confusion_dir.mkdir(exist_ok=False)
    print(f"Stage 5A: T={config['temperature']!r}, tau={config['threshold']!r}; calibrated_msp; 406 images", flush=True)
    for condition, paths in inventory.items():
        counts = dict.fromkeys(COUNT_KEYS, 0)
        confusion = np.zeros((19, 19), dtype=np.int64)
        changes = 0
        for path in paths:
            gt_path = find_gt_path(path, condition)
            audit.begin(PROJECT / path, PROJECT / gt_path)
            result = pipeline.predict(path)
            meta = result["metadata"]
            assert meta["gt_used"] is False and meta["ground_truth_loaded"] is False
            assert meta["temperature"] == config["temperature"] and meta["threshold"] == config["threshold"]
            assert meta["uncertainty_method"] == "calibrated_msp" and meta["canonical_prediction_changes"] == 0
            pred, mask = result["segmentation"], result["reliability_mask"]
            assert np.isfinite(result["uncertainty"]).all()
            assert result["uncertainty"].min() >= 0 and result["uncertainty"].max() <= 1
            audit.inference_completed()
            with Image.open(gt_path) as source:
                gt = np.array(source)
            audit.finish()
            count = image_counts(pred, mask, gt)
            assert count["all_accepted_pixels"] == meta["accepted_pixels"]
            assert count["all_rejected_pixels"] == meta["rejected_pixels"]
            update_confusion_matrix(confusion, pred, gt)
            for key in COUNT_KEYS:
                counts[key] += count[key]
            changes += meta["canonical_prediction_changes"]
            row = metric_row(condition, count, config, meta["canonical_prediction_changes"])
            row.update(image=path.name, rgb_path=str(path), gt_path=str(gt_path))
            # Condition-level baseline check is performed before any metrics CSV is saved.
            row["baseline_counts_check"] = "PASS (condition aggregate)"
            per_image.append(row)
            done += 1
            if done == 1 or done % 25 == 0 or done == 406:
                print(f"[{done}/406] {condition}; frozen inference + offline GT complete", flush=True)
            del result, pred, mask, gt
        verify_baseline(condition, counts, confusion)
        assert changes == 0
        np.save(confusion_dir / f"{condition}_confusion_matrix.npy", confusion)
        counts_by_condition[condition] = counts
        aggregate.append(metric_row(condition, counts, config, changes))
        total_changes += changes
        print(f"{condition}: exact Stage 1/3C confusion PASS; accuracy={aggregate[-1]['pixel_accuracy']:.9%}", flush=True)
    assert done == len(per_image) == 406 and total_changes == 0
    pooled = {key: sum(c[key] for c in counts_by_condition.values()) for key in COUNT_KEYS}
    aggregate.append(metric_row("pooled", pooled, config, total_changes))
    assert pooled["valid_pixels"] == 790712012 and pooled["all_pixels"] == 406 * 1080 * 1920
    write_csv(ROOT / "condition_metrics.csv", aggregate[:-1])
    write_csv(ROOT / "stage5a_combined_summary.csv", aggregate)
    write_csv(ROOT / "per_image_metrics.csv", per_image)
    comparison = generalization_comparison(aggregate)
    write_csv(ROOT / "generalization_comparison.csv", comparison)
    figures()
    final_reports(aggregate, comparison, audit, config, config_hash)
    print(json.dumps(aggregate[-1], indent=2), flush=True)


if __name__ == "__main__":
    main()
