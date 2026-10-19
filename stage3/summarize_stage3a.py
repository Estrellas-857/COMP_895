"""Validate and summarize saved Stage 3A outputs without running inference."""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/stage3_calibration")
CONDITIONS = {"fog": 100, "rain": 100, "snow": 100, "night": 106}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def validate(condition, num_images):
    directory = ROOT / condition
    rows = read_rows(directory / "raw_calibration_summary.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["condition"] == condition
    assert int(row["num_images"]) == num_images
    assert int(row["num_bins"]) == 15
    assert float(row["high_conf_threshold"]) == 0.90
    bins = read_rows(directory / "confidence_bins.csv")
    assert len(bins) == 15
    n = int(row["num_valid_pixels"])
    correct = int(row["num_correct_pixels"])
    incorrect = int(row["num_incorrect_pixels"])
    high = int(row["num_high_conf_pixels"])
    high_wrong = int(row["num_high_conf_incorrect"])
    assert correct + incorrect == n
    assert 0 <= high_wrong <= high <= n and high_wrong <= incorrect
    close(row["pixel_accuracy"], correct / n)
    close(row["error_rate"], incorrect / n)
    close(row["high_conf_coverage"], high / n)
    close(row["high_conf_error_rate"], high_wrong / high)
    close(row["fraction_incorrect_that_are_high_conf"], high_wrong / incorrect)
    assert math.isfinite(float(row["nll"])) and float(row["nll"]) >= 0
    assert sum(int(b["count"]) for b in bins) == n
    close(sum(float(b["fraction"]) for b in bins), 1)
    ece = 0.0
    reconstructed_correct = 0.0
    for i, b in enumerate(bins):
        assert int(b["bin_index"]) == i
        close(b["bin_lower"], i / 15)
        close(b["bin_upper"], (i + 1) / 15)
        count = int(b["count"])
        assert count >= 0
        close(b["fraction"], count / n)
        if count:
            confidence, accuracy = float(b["mean_confidence"]), float(b["accuracy"])
            assert float(b["bin_lower"]) <= confidence <= float(b["bin_upper"])
            assert 0 <= accuracy <= 1
            close(b["error_rate"], 1 - accuracy)
            close(b["calibration_gap"], abs(accuracy - confidence))
            ece += count / n * abs(accuracy - confidence)
            reconstructed_correct += count * accuracy
        else:
            assert all(math.isnan(float(b[k])) for k in
                       ("mean_confidence", "accuracy", "error_rate", "calibration_gap"))
    close(reconstructed_correct, correct)
    close(ece, row["ece"])
    assert 0 <= ece <= 1
    baseline = np.load(Path("results") / f"{condition}_val" / "confusion_matrix.npy")
    assert int(baseline.sum()) == n
    baseline_accuracy = np.trace(baseline) / baseline.sum()
    close(row["stage1_pixel_accuracy"], baseline_accuracy)
    close(row["pixel_accuracy_difference"], float(row["pixel_accuracy"]) - baseline_accuracy)
    assert float(row["pixel_accuracy_tolerance"]) == 1e-5
    assert abs(float(row["pixel_accuracy_difference"])) <= 1e-5
    assert row["baseline_accuracy_check"] == "PASS"
    assert row["stage1_valid_pixel_count_matches"] == "True"
    assert row["stage1_confusion_matrix_exact_match"] == "True"
    for name in ("summary.txt", "terminal_output.log", "reliability_diagram.png"):
        assert (directory / name).stat().st_size > 0
    print(f"PASS {condition}: {num_images} images; {n:,} valid pixels; "
          "bin coverage, ECE, conditional probabilities and Stage 1 agreement.")
    return row, bins


def main():
    validated = [validate(c, n) for c, n in CONDITIONS.items()]
    rows = [row for row, _ in validated]
    output = ROOT / "stage3a_summary"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage3a_combined_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    values = [float(r["ece"]) for r in rows]
    bars = ax.bar([r["condition"].capitalize() for r in rows], values,
                  color="#1764a0", width=0.6)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=5)
    ax.set(ylabel="Expected calibration error (ECE)",
           title="Raw confidence calibration on ACDC validation",
           ylim=(0, max(values) * 1.22))
    ax.text(0.02, 0.96, "15 equal-width bins; pixel-weighted within each condition",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(output / "stage3a_calibration_comparison.png", dpi=300)
    plt.close(fig)

    lines = [
        "STAGE 3A: RAW CONFIDENCE CALIBRATION ON ACDC VALIDATION",
        "", "All four conditions passed the saved-output sanity checks.",
        "All four runs report exact full confusion-matrix agreement with frozen Stage 1.",
        "Night is read from its existing validated outputs; no Night inference was run.",
        "",
        "condition | accuracy (%) | error (%) | ECE | NLL (nats) | high-confidence coverage (%) | P(incorrect | high confidence) (%) | P(high confidence | incorrect) (%)",
    ]
    for r in rows:
        lines.append(f"{r['condition']} | {float(r['pixel_accuracy'])*100:.6f} | "
                     f"{float(r['error_rate'])*100:.6f} | {float(r['ece']):.10f} | "
                     f"{float(r['nll']):.10f} | {float(r['high_conf_coverage'])*100:.6f} | "
                     f"{float(r['high_conf_error_rate'])*100:.6f} | "
                     f"{float(r['fraction_incorrect_that_are_high_conf'])*100:.6f}")
    lines += [
        "", "High confidence means confidence >= 0.90 (prespecified, not selected).",
        "high_conf_error_rate = P(incorrect | confidence >= 0.90).",
        "fraction_incorrect_that_are_high_conf = P(confidence >= 0.90 | incorrect).",
        "CSV accuracy, error, coverage and conditional probabilities are fractions, not percentages.",
        "ECE uses 15 [lower, upper) bins, with 1.0 included in the final bin.",
        "NLL uses the unchanged stable log-softmax implementation and natural logarithms.",
        "All statistics are over valid pixels; trainId 255 is ignored.",
        "Individual reliability diagrams read the exact saved confidence_bins.csv files.",
        "", "Descriptive bin-level confidence minus accuracy:",
    ]
    for row, bins in validated:
        nonempty = [b for b in bins if int(b["count"])]
        over = sum(float(b["mean_confidence"]) > float(b["accuracy"]) for b in nonempty)
        signed_gap = sum(float(b["fraction"]) * (float(b["mean_confidence"]) -
                         float(b["accuracy"])) for b in nonempty)
        lines.append(f"{row['condition']}: confidence exceeds accuracy in {over}/{len(nonempty)} "
                     f"non-empty bins; pixel-weighted signed gap = {signed_gap:+.6f}.")
    lines += [
        "", "These results describe these ACDC validation subsets only.",
        "Calibration and segmentation accuracy are different properties; ECE does not establish accuracy ranking.",
        "No strong conclusions are drawn from small between-condition differences.",
        "No fitting, model changes, threshold selection or Stage 3B work was performed.",
    ]
    (output / "stage3a_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nCombined outputs: {output}")


if __name__ == "__main__":
    main()
