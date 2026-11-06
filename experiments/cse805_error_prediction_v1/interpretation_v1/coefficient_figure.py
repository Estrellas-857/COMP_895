"""Plot existing weighted class diagnostics without fitting or resampling."""
import csv
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / "logs/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .shared import OUT, table, dump, verify_execution
from ..common import sha256


def main():
    verify_execution()
    sources = {name: list(csv.DictReader((OUT / "tables" / name).open())) for name in
               ("class_profile_high_confidence.csv", "class_profile_train_heldout.csv")}
    panels = [dict(group="gopro_0376", weather="Night", cutoff=.99, classes=("vegetation", "road"),
                   source="class_profile_high_confidence.csv", title="Night · gopro_0376 · MSP ≥ 0.99"),
              dict(group="gopro_0477", weather="Fog", cutoff=None, classes=("sky", "building"),
                   source="class_profile_train_heldout.csv", title="Fog · gopro_0477 · all confidence levels")]
    metrics = [("train_error_rate", "Outer-training error rate", "#999999"),
               ("heldout_error_rate", "Held-out error rate", "#D55E00"),
               ("heldout_mean_lr3_score", "Mean LR3 predicted risk", "#0072B2"),
               ("heldout_mean_lr3_class_score", "Mean LR+class predicted risk", "#009E73")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.1), sharey=True)
    output_rows = []
    for ax, panel in zip(axes, panels):
        selected = []
        for name in panel["classes"]:
            matches = [row for row in sources[panel["source"]] if row["group"] == panel["group"]
                       and row["predicted_class"] == name and
                       (panel["cutoff"] is None or float(row["msp_cutoff"]) == panel["cutoff"])]
            if len(matches) != 1:
                raise ValueError("Plot selection must identify exactly one existing diagnostic row")
            selected.append(matches[0])
        locations = np.arange(2)
        width = .18
        for i, (key, label, color) in enumerate(metrics):
            values = np.array([float(row[key]) for row in selected])
            if not np.all(np.isfinite(values) & (values >= 0) & (values <= 1)):
                raise ValueError("Plotted rates must be defined probabilities")
            bars = ax.bar(locations + (i-1.5)*width, 100*values, width=width, label=label,
                          color=color, edgecolor="white", linewidth=.5)
            ax.bar_label(bars, labels=[f"{100*v:.1f}" for v in values], padding=3, fontsize=10)
            for row, value in zip(selected, values):
                output_rows.append(dict(panel=panel["weather"], group=panel["group"],
                    predicted_class=row["predicted_class"], msp_minimum=panel["cutoff"],
                    metric=key, label=label, fraction=float(value), percent=float(100*value),
                    train_subset_share_of_valid=float(row["train_class_population_share"]),
                    heldout_subset_share_of_valid=float(row["heldout_class_population_share"]),
                    source_table=panel["source"]))
        ax.set_xticks(locations, [f"{r['predicted_class'].capitalize()}\n{100*float(r['heldout_class_population_share']):.2f}% of held-out valid pixels"
                                 for r in selected], fontsize=10)
        ax.set_title(panel["title"], fontsize=12, pad=13)
        ax.set_ylim(0, 85)
        ax.set_yticks(np.arange(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Error rate or mean predicted risk", fontsize=11)
    figure.suptitle("Predicted-class reliability and frozen-model scores", fontsize=16, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .13), ncol=2,
                  fontsize=11, frameon=False, columnspacing=2.3)
    figure.text(.5, .04,
                "Inverse-inclusion-weighted saved samples: outer train 512/image; held-out QA 8,192/image.\n"
                "Outer training includes other recording groups and weather conditions. Descriptive diagnostic; not a calibration validation.",
                ha="center", va="bottom", fontsize=9.5, color="#444444")
    figure.subplots_adjust(top=.85, bottom=.30, left=.075, right=.98, wspace=.15)
    output = OUT / "figures/class_reliability_shift.png"
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)
    values_path = table("coefficient_figure_values.csv", output_rows)
    assert len(output_rows) == 16
    verify_execution()
    dump("logs/coefficient_figure.json", dict(status="PASS", script_sha256=sha256(__file__),
        source_hashes={name: sha256(OUT / "tables" / name) for name in sources},
        output_hashes={str(p.relative_to(OUT)): sha256(p) for p in (output, values_path)},
        note="Reads existing weighted sample summaries only; no model, feature, threshold, or sample changes"))
    print(output)


if __name__ == "__main__":
    main()
