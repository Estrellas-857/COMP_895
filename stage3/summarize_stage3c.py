"""Check and summarize saved Stage 3C validation results; no inference or fitting."""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from stage3.evaluate_calibrated import (
    ROOT, RAW_ROOT, EXPECTED_IMAGES, load_temperature, load_raw, rows_from, sha256, write_csv,
)

from stage1._layout import matches_frozen_hash


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def check_condition(condition, expected_images, temperature):
    directory = ROOT / condition
    saved = rows_from(directory / "calibrated_summary.csv")
    assert len(saved) == 1
    row = saved[0]
    raw, _ = load_raw(condition)
    assert row["condition"] == condition and int(row["num_images"]) == expected_images
    assert float(row["temperature"]) == temperature
    n = int(row["num_valid_pixels"])
    correct, wrong = int(row["num_correct_pixels"]), int(row["num_incorrect_pixels"])
    assert correct + wrong == n == int(raw["num_valid_pixels"])
    assert int(row["prediction_changes"]) == 0 and int(row["prediction_pixels_checked"]) == n
    assert float(row["pixel_accuracy"]) == float(raw["pixel_accuracy"])
    close(row["pixel_accuracy"], correct/n)
    close(row["error_rate"], wrong/n)
    assert int(row["num_bins"]) == 15 and float(row["high_conf_threshold"]) == 0.90
    for name in ("baseline_accuracy_check", "raw_stage3a_reproduction_check", "frozen_temperature_check"):
        assert row[name] == "PASS"
    assert row["stage1_confusion_matrix_exact_match"] == "True"
    assert np.array_equal(np.load(directory / "confusion_matrix.npy"),
                          np.load(f"results/{condition}_val/confusion_matrix.npy"))
    for key in ("ece", "nll", "high_conf_coverage", "high_conf_error_rate",
                "fraction_incorrect_that_are_high_conf", "num_high_conf_pixels", "num_high_conf_incorrect"):
        assert float(row[f"raw_{key}"]) == float(raw[key])
    for metric in ("ece", "nll"):
        before, after = float(row[f"raw_{metric}"]), float(row[f"calibrated_{metric}"])
        assert math.isfinite(after) and after >= 0
        close(row[f"{metric}_absolute_change"], after-before)
        close(row[f"{metric}_relative_change"], (after-before)/before)
        assert row[f"{metric}_improved"] == str(after < before)
    for prefix in ("raw", "calibrated"):
        high, high_wrong = int(row[f"{prefix}_num_high_conf_pixels"]), int(row[f"{prefix}_num_high_conf_incorrect"])
        assert 0 <= high_wrong <= high <= n and high_wrong <= wrong
        close(row[f"{prefix}_high_conf_coverage"], high/n)
        close(row[f"{prefix}_high_conf_error_rate"], high_wrong/high)
        close(row[f"{prefix}_fraction_incorrect_that_are_high_conf"], high_wrong/wrong)
        for metric in ("ece", "high_conf_coverage", "high_conf_error_rate", "fraction_incorrect_that_are_high_conf"):
            assert 0 <= float(row[f"{prefix}_{metric}"]) <= 1
    bins = rows_from(directory / "calibrated_confidence_bins.csv")
    assert len(bins) == 15 and sum(int(b["count"]) for b in bins) == n
    ece, reconstructed_correct = 0., 0.
    for i, b in enumerate(bins):
        assert int(b["bin_index"]) == i
        close(b["bin_lower"], i/15)
        close(b["bin_upper"], (i+1)/15)
        count = int(b["count"])
        assert count >= 0
        close(b["fraction"], count/n)
        if count:
            confidence, accuracy = float(b["mean_confidence"]), float(b["accuracy"])
            assert float(b["bin_lower"]) <= confidence <= float(b["bin_upper"])
            assert 0 <= accuracy <= 1
            close(b["error_rate"], 1-accuracy)
            close(b["calibration_gap"], abs(accuracy-confidence))
            ece += count/n * abs(accuracy-confidence)
            reconstructed_correct += count*accuracy
        else:
            assert all(math.isnan(float(b[k])) for k in ("mean_confidence", "accuracy", "error_rate", "calibration_gap"))
    close(ece, row["calibrated_ece"])
    close(reconstructed_correct, correct)
    close(sum(float(b["fraction"]) for b in bins), 1)
    for name in ("raw_vs_calibrated_reliability.png", "summary.txt", "terminal_output.log"):
        assert (directory / name).stat().st_size > 0
    print(f"PASS {condition}: {expected_images} images, {n:,} valid pixels, exact accuracy/confusion agreement, zero prediction changes.")
    return row


def comparison_plot(rows, metric, ylabel, title, path):
    x = np.arange(len(rows))
    raw = [float(r[f"raw_{metric}"]) for r in rows]
    calibrated = [float(r[f"calibrated_{metric}"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    for positions, values, color, label in ((x-.18, raw, "#1764a0", "Raw"),
                                           (x+.18, calibrated, "#c45b16", "Calibrated")):
        bars = ax.bar(positions, values, width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4, fontsize=9)
    ax.set(xticks=x, xticklabels=[r["condition"].capitalize() for r in rows], ylabel=ylabel,
           ylim=(0, max(raw+calibrated)*1.22), title=title)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    temperature, _ = load_temperature()
    rows = [check_condition(c, n, temperature) for c, n in EXPECTED_IMAGES.items()]
    output = ROOT / "stage3c_summary"
    if output.exists():
        raise RuntimeError("Combined summary already exists; refuse to overwrite")
    output.mkdir()
    write_csv(output / "stage3c_combined_summary.csv", rows)
    comparison_plot(rows, "ece", "Expected calibration error (ECE)",
                    "ACDC validation: raw vs calibrated ECE", output / "fig1_ece_before_after.png")
    comparison_plot(rows, "nll", "Negative log-likelihood (nats)",
                    "ACDC validation: raw vs calibrated NLL", output / "fig2_nll_before_after.png")
    lines = ["STAGE 3C - INDEPENDENT VALIDATION OF FROZEN SHARED TEMPERATURE", "",
             f"Frozen train-fitted T = {temperature!r}; no validation parameter fitting.",
             "Accuracy, error, coverage and conditional probabilities in CSVs are fractions.",
             "absolute_change = calibrated - raw; relative_change = (calibrated - raw)/raw.",
             "Negative ECE/NLL changes indicate improvement; relative changes below are percentages.",
             "HC = confidence >= 0.90 (prespecified). high_conf_error_rate = P(incorrect | HC).",
             "fraction_incorrect_that_are_high_conf = P(HC | incorrect).", ""]
    for r in rows:
        lines.append(f"{r['condition'].upper()}: {r['num_images']} images, {int(r['num_valid_pixels']):,} valid pixels; "
                     f"accuracy {float(r['pixel_accuracy'])*100:.6f}%, prediction changes {r['prediction_changes']}.")
        for metric in ("ece", "nll"):
            lines.append(f"  {metric.upper()}: {float(r[f'raw_{metric}']):.10f} -> {float(r[f'calibrated_{metric}']):.10f}; "
                         f"change {float(r[f'{metric}_absolute_change']):+.10f} "
                         f"({float(r[f'{metric}_relative_change'])*100:+.6f}%); "
                         f"{'improved' if r[f'{metric}_improved']=='True' else 'did not improve'}.")
        for metric in ("high_conf_coverage", "high_conf_error_rate", "fraction_incorrect_that_are_high_conf"):
            lines.append(f"  {metric}: {float(r[f'raw_{metric}'])*100:.6f}% -> {float(r[f'calibrated_{metric}'])*100:.6f}%.")
    lines += ["", "Protocol and checks:",
              "15 equal-width confidence bins, [lower,upper), final bin includes 1.0; all valid pixels included exactly once.",
              "Model and preprocessing frozen; float32 bilinear full-resolution interpolation before temperature scaling.",
              "Raw Stage 3A values retained from disk and independently reproduced during inference.",
              "Scaled branch uses float64 temperature division, softmax and stable log-softmax as in Stage 3B.",
              "All 406 validation images evaluated; zero changed predictions across all valid pixels.",
              "Accuracy and full confusion matrices match frozen Stage 1 exactly.",
              "All bounds, bin coverage, conditional-probability denominators and frozen-temperature checks passed.",
              "Reliability figures read the saved raw/calibrated bins and use identical [0,1] axes.",
              "", "Interpretation:",
              "ECE and NLL measure different aspects of probability calibration and may move in different directions.",
              "Changed high-confidence coverage reflects changed probabilities, not corrected semantic predictions.",
              "Results describe these validation subsets; no claim about accuracy, mIoU or failure-detection performance.",
              "No temperature refitting, threshold selection, Stage 3D or Stage 4 work performed."]
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    changed = [path for path, digest in manifest.items() if not matches_frozen_hash(path, digest)]
    assert not changed, changed
    preservation = (f"PASS: all {len(manifest)} pre-existing protected files pass SHA-256 checks with the documented source relocation.\n"
                    "Stage 1, Stage 2, Stage 3A and Stage 3B artifacts unchanged, including the fitted temperature, splits and cache.\n"
                    "Stage 3B cache files were read only as opaque bytes for preservation hashes; no train data used in evaluation.\n")
    (ROOT / "preservation_check.txt").write_text(preservation)
    lines += ["", preservation]
    (output / "stage3c_summary.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
