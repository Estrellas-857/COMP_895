"""Validate saved Stage 3D histograms and make descriptive before/after reports."""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_calibrated_failure_detection import (
    ROOT, METHOD, EXPECTED_IMAGES, load_temperature, load_baselines, rows_from,
    HIST_EDGES, compute_curves, sha256, write_csv,
)


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def validate(condition, temperature):
    output = ROOT / condition
    raw, _, baseline = load_baselines(condition)
    rows = rows_from(output / "calibrated_failure_detection_summary.csv")
    assert len(rows) == 2 and {r["method"] for r in rows} == {"msp", "entropy"}
    with np.load(output / "calibrated_uncertainty_histograms.npz") as hist:
        np.testing.assert_array_equal(hist["bin_edges"], HIST_EDGES)
        for row in rows:
            method = row["method"]
            assert row["condition"] == condition and row["metric_method"] == METHOD
            assert int(row["num_images"]) == EXPECTED_IMAGES[condition]
            assert float(row["temperature"]) == temperature and int(row["histogram_bins"]) == 1000
            correct, incorrect = hist[f"{method}_correct_hist"], hist[f"{method}_incorrect_hist"]
            for h in (correct, incorrect):
                assert h.shape == (1000,) and h.dtype == np.int64 and np.all(h >= 0)
            n, positives, negatives = (int(baseline[k]) for k in ("num_valid_pixels", "num_incorrect_pixels", "num_correct_pixels"))
            assert int(correct.sum()) == negatives and int(incorrect.sum()) == positives
            assert n == positives+negatives == int(row["num_valid_pixels"])
            assert int(row["num_correct_pixels"]) == negatives and int(row["num_incorrect_pixels"]) == positives
            assert int(row["prediction_changes"]) == 0 and int(row["prediction_pixels_checked"]) == n
            close(row["pixel_accuracy"], negatives/n)
            metric = compute_curves(correct, incorrect)
            for key in ("auroc", "auprc"):
                close(row[f"calibrated_{key}"], metric[key])
                assert float(row[f"raw_{key}"]) == float(raw[method][key])
                close(row[f"{key}_change"], metric[key]-float(raw[method][key]))
                assert 0 <= float(row[f"calibrated_{key}"]) <= 1
            close(row["error_prevalence"], positives/n)
            close(row["pr_random_baseline"], positives/n)
            close(row["raw_error_prevalence"], raw[method]["positive_prevalence"])
            close(row["raw_pr_random_baseline"], raw[method]["positive_prevalence"])
            close(row["raw_auprc_gain"], float(row["raw_auprc"])-float(row["raw_pr_random_baseline"]))
            close(row["calibrated_auprc_gain"], metric["auprc"]-positives/n)
            assert int(row["incorrect_count_change_vs_stage2d"]) == positives-int(raw[method]["positive_pixels"])
            assert row["raw_histogram_exact_reproduction"] == "True" and row["stage1_3c_confusion_exact_match"] == "True"
    np.testing.assert_array_equal(np.load(output / "confusion_matrix.npy"), np.load(f"results/{condition}_val/confusion_matrix.npy"))
    np.testing.assert_array_equal(np.load(output / "raw_softmax_confusion_recheck.npy"),
                                  np.load(f"results/stage2_uncertainty/{condition}/confusion_matrix.npy"))
    audit = rows_from(output / "raw_softmax_tie_audit.csv")
    assert len(audit) == int(rows[0]["historical_softmax_argmax_differences"])
    assert all(r["probability_tie"] == "True" and
               float(r["probability_at_logit_prediction"]) == float(r["probability_at_softmax_prediction"])
               for r in audit)
    changed_correct = sum(int(r["raw_softmax_prediction"] == r["gt"]) - int(r["logit_prediction"] == r["gt"]) for r in audit)
    # (raw correct - logit correct) equals (logit incorrect - raw incorrect).
    assert changed_correct == int(rows[0]["incorrect_count_change_vs_stage2d"])
    assert (output / "terminal_output.log").stat().st_size > 0
    print(f"PASS {condition}: complete histograms, original Stage 2D integration, unchanged Stage 1/3C predictions/counts; "
          f"historical Stage 2D incorrect-count offset={rows[0]['incorrect_count_change_vs_stage2d']}.")
    return rows


def wide_row(rows):
    first = rows[0]
    result = {k: first[k] for k in ("condition", "temperature", "metric_method", "histogram_bins",
              "num_images", "num_valid_pixels", "num_correct_pixels", "num_incorrect_pixels",
              "error_prevalence", "raw_error_prevalence", "prediction_changes", "pr_random_baseline",
              "raw_pr_random_baseline", "incorrect_count_change_vs_stage2d", "historical_softmax_argmax_differences")}
    for row in rows:
        method = row["method"]
        for metric in ("auroc", "auprc"):
            for branch in ("raw", "calibrated"):
                result[f"{branch}_{method}_{metric}"] = row[f"{branch}_{metric}"]
            result[f"{method}_{metric}_change"] = row[f"{metric}_change"]
        for branch in ("raw", "calibrated"):
            result[f"{branch}_{method}_auprc_gain"] = row[f"{branch}_auprc_gain"]
    return result


def plot_comparison(rows, metric, path):
    fig, ax = plt.subplots(figsize=(11.5, 6), layout="constrained")
    x = np.arange(len(rows))
    series = [("raw_msp", "Raw MSP", "#1764a0", ""),
              ("calibrated_msp", "Calibrated MSP", "#78b4d8", "//"),
              ("raw_entropy", "Raw entropy", "#b34a12", ""),
              ("calibrated_entropy", "Calibrated entropy", "#efa66b", "//")]
    for i, (key, label, color, hatch) in enumerate(series):
        values = [float(r[f"{key}_{metric}"]) for r in rows]
        bars = ax.bar(x+(i-1.5)*.20, values, width=.18, label=label, color=color,
                      hatch=hatch, edgecolor="white", linewidth=.4)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=8)
    if metric == "auroc":
        ax.axhline(.5, color="0.3", linestyle="--", linewidth=1, label="Random AUROC = 0.5")
        subtitle = "1000-bin histogram approximation; incorrect pixels are positive"
    else:
        for i, row in enumerate(rows):
            p = float(row["pr_random_baseline"])
            ax.hlines(p, i-.43, i+.43, color="black", linestyle="--", linewidth=1.4,
                      label="Random PR baseline (prevalence)" if i == 0 else None)
        subtitle = "1000-bin histogram approximation; dashed segments show each condition's prevalence"
    ax.set(xticks=x, xticklabels=[r["condition"].capitalize() for r in rows],
           ylim=(0, 1), ylabel=metric.upper(), title=f"Failure detection: raw vs calibrated {metric.upper()}\n{subtitle}")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=3, frameon=False, fontsize=9)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    temperature = load_temperature()
    grouped = [validate(c, temperature) for c in EXPECTED_IMAGES]
    rows = [wide_row(r) for r in grouped]
    output = ROOT / "stage3d_summary"
    if output.exists():
        raise RuntimeError("Combined summary already exists; refusing overwrite")
    output.mkdir()
    write_csv(output / "stage3d_combined_summary.csv", rows)
    plot_comparison(rows, "auroc", output / "fig1_auroc_before_after.png")
    plot_comparison(rows, "auprc", output / "fig2_auprc_before_after.png")
    lines = ["STAGE 3D - FAILURE DISCRIMINATION AFTER FROZEN TEMPERATURE SCALING", "",
             f"T={temperature!r}; metric_method={METHOD}",
             "Positive class=incorrect pixel. Higher uncertainty predicts failure.",
             "Unchanged functions: evaluate_uncertainty.UncertaintyAccumulator and evaluate_failure_detection.compute_curves.",
             "AUROC uses trapezoidal integration; AUPRC uses sum(delta_recall*precision[1:]).",
             "Empty initial prediction has precision=1. Histogram bins are accumulated from high to low uncertainty.",
             "PR baseline=error prevalence. AUPRC gain=AUPRC minus the corresponding raw/calibrated prevalence.",
             "All changes below are absolute calibrated-minus-raw differences, not relative percentages.", ""]
    for r in rows:
        lines.append(f"{r['condition'].upper()}: prevalence={float(r['error_prevalence']):.10f}; "
                     f"historical raw prevalence={float(r['raw_error_prevalence']):.10f}.")
        for method in ("msp", "entropy"):
            for metric in ("auroc", "auprc"):
                before, after, change = (float(r[k]) for k in (f"raw_{method}_{metric}", f"calibrated_{method}_{metric}", f"{method}_{metric}_change"))
                direction = "increased" if change > 0 else "decreased" if change < 0 else "unchanged"
                lines.append(f"  {method} {metric.upper()}: {before:.10f} -> {after:.10f}; delta={change:+.10f} ({direction}).")
            lines.append(f"  {method} AUPRC gain above prevalence: {float(r[f'raw_{method}_auprc_gain']):.10f} -> "
                         f"{float(r[f'calibrated_{method}_auprc_gain']):.10f}.")
    lines += ["", "Sanity checks and methodological qualification:",
              "All 406 validation images included; zero temperature-induced prediction changes over all valid pixels.",
              "Stage 1/3C counts and confusion matrices match exactly; all 1000-bin int64 histograms are complete.",
              "Raw Stage 2D metrics and raw Stage 2B histograms were reproduced exactly without altering their artifacts.",
              "Pre-existing Rain/Snow count discrepancy: Stage 2 uses float32 softmax argmax; Stage 1/3C/3D use logit argmax.",
              "One float32 probability tie changes correctness in each: calibrated-vs-historical incorrect-count difference is +1 for Rain, -1 for Snow.",
              "The inference audit identifies each differing pixel and verifies the tied probabilities; both frozen definitions are retained.",
              "Raw PR baselines/gains use the original Stage 2D prevalence; calibrated PR baselines/gains use the exact Stage 3C prevalence.",
              "This tiny label discrepancy qualifies strict before/after comparability; it is not caused by temperature scaling.",
              "Raw calculations retain float32; calibrated probabilities retain Stage 3B/3C float64 scaling after float32 interpolation.",
              "Both branches use the same entropy formula, log-probability clamp (1e-12), histogram edges and integration definitions.",
              "Histogram quantization can affect measured differences; these are descriptive approximations, not statistical significance results.",
              "Calibration and failure discrimination are distinct; no improvement is assumed from ECE/NLL changes.",
              "No segmentation, rejection-performance or reliability-mask claims are made.",
              "No parameter fitting, threshold selection, reserved reliability-selection data use, risk-coverage or Stage 4 work."]
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    changed = [p for p, digest in manifest.items() if sha256(p) != digest]
    assert not changed, changed
    preservation = (f"PASS: all {len(manifest)} pre-existing protected script/result files retain their SHA-256 hashes.\n"
                    "Stages 1, 2, 3A, 3B and 3C unchanged, including frozen temperature and split artifacts.\n"
                    "Training artifacts were accessed only as opaque bytes for preservation hashing; reserved train images/GT were not loaded.\n")
    (ROOT / "preservation_check.txt").write_text(preservation)
    lines += ["", preservation]
    (output / "stage3d_summary.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
