from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

RESULT_ROOT = Path(
    "results/stage2_uncertainty"
)

OUTPUT_DIR = (
    RESULT_ROOT
    / "stage2b_analysis"
)

CONDITIONS = [
    "fog",
    "rain",
    "snow",
    "night",
]

DISPLAY_NAMES = {
    "fog": "Fog",
    "rain": "Rain",
    "snow": "Snow",
    "night": "Night",
}


# ============================================================
# Load summaries
# ============================================================

def load_summary():

    rows = []

    for condition in CONDITIONS:

        path = (
            RESULT_ROOT
            / condition
            / "summary.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

        df = pd.read_csv(path)

        if len(df) != 1:
            raise ValueError(
                f"Expected one row in {path}"
            )

        row = df.iloc[0].to_dict()

        rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    summary["condition"] = (
        pd.Categorical(
            summary["condition"],
            categories=CONDITIONS,
            ordered=True
        )
    )

    summary = (
        summary
        .sort_values("condition")
        .reset_index(drop=True)
    )

    # Descriptive separation only.
    # Do NOT interpret these as AUROC.
    summary["msp_mean_gap"] = (
        summary["msp_incorrect_mean"]
        - summary["msp_correct_mean"]
    )

    summary["entropy_mean_gap"] = (
        summary["entropy_incorrect_mean"]
        - summary["entropy_correct_mean"]
    )

    return summary


# ============================================================
# Console table
# ============================================================

def print_summary(summary):

    print("\n")
    print("=" * 116)
    print("STAGE 2B — UNCERTAINTY PROFILING SUMMARY")
    print("=" * 116)

    print(
        f"{'Condition':<10}"
        f"{'Error %':>10}"
        f"{'MSP Corr':>11}"
        f"{'MSP Wrong':>12}"
        f"{'MSP Gap':>11}"
        f"{'Ent Corr':>11}"
        f"{'Ent Wrong':>12}"
        f"{'Ent Gap':>11}"
        f"{'HC Wrong %':>13}"
    )

    print("-" * 116)

    for _, row in summary.iterrows():

        condition = str(
            row["condition"]
        )

        print(
            f"{DISPLAY_NAMES[condition]:<10}"
            f"{row['error_rate']*100:>10.2f}"
            f"{row['msp_correct_mean']:>11.4f}"
            f"{row['msp_incorrect_mean']:>12.4f}"
            f"{row['msp_mean_gap']:>11.4f}"
            f"{row['entropy_correct_mean']:>11.4f}"
            f"{row['entropy_incorrect_mean']:>12.4f}"
            f"{row['entropy_mean_gap']:>11.4f}"
            f"{row['high_conf_wrong_fraction']*100:>13.2f}"
        )


# ============================================================
# Figure 1:
# MSP correct vs incorrect
# ============================================================

def plot_msp_means(summary):

    labels = [
        DISPLAY_NAMES[str(x)]
        for x in summary["condition"]
    ]

    correct = (
        summary[
            "msp_correct_mean"
        ].to_numpy()
    )

    incorrect = (
        summary[
            "msp_incorrect_mean"
        ].to_numpy()
    )

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars1 = ax.bar(
        x - width / 2,
        correct,
        width,
        label="Correct Pixels"
    )

    bars2 = ax.bar(
        x + width / 2,
        incorrect,
        width,
        label="Incorrect Pixels"
    )

    ax.set_ylabel(
        "Mean MSP Uncertainty"
    )

    ax.set_title(
        "MSP Uncertainty: Correct vs Incorrect Pixels"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0,
        max(incorrect) + 0.06
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bars in [
        bars1,
        bars2
    ]:

        for bar in bars:

            height = (
                bar.get_height()
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig1_msp_correct_vs_incorrect.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Figure 2:
# Entropy correct vs incorrect
# ============================================================

def plot_entropy_means(summary):

    labels = [
        DISPLAY_NAMES[str(x)]
        for x in summary["condition"]
    ]

    correct = (
        summary[
            "entropy_correct_mean"
        ].to_numpy()
    )

    incorrect = (
        summary[
            "entropy_incorrect_mean"
        ].to_numpy()
    )

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars1 = ax.bar(
        x - width / 2,
        correct,
        width,
        label="Correct Pixels"
    )

    bars2 = ax.bar(
        x + width / 2,
        incorrect,
        width,
        label="Incorrect Pixels"
    )

    ax.set_ylabel(
        "Mean Normalized Entropy"
    )

    ax.set_title(
        "Predictive Entropy: Correct vs Incorrect Pixels"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0,
        max(incorrect) + 0.06
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bars in [
        bars1,
        bars2
    ]:

        for bar in bars:

            height = (
                bar.get_height()
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig2_entropy_correct_vs_incorrect.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Figure 3:
# Error rate vs high-confidence errors
# ============================================================

def plot_failure_profile(summary):

    labels = [
        DISPLAY_NAMES[str(x)]
        for x in summary["condition"]
    ]

    # Fraction of ALL valid pixels that are wrong
    error_rate = (
        summary["error_rate"].to_numpy()
        * 100.0
    )

    # Fraction of ALL valid pixels that are BOTH
    # wrong and high-confidence.
    high_conf_error_rate = (
        summary["error_rate"].to_numpy()
        * summary["high_conf_wrong_fraction"].to_numpy()
        * 100.0
    )

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars1 = ax.bar(
        x - width / 2,
        error_rate,
        width,
        label="All Incorrect Pixels"
    )

    bars2 = ax.bar(
        x + width / 2,
        high_conf_error_rate,
        width,
        label=(
            "High-Confidence Incorrect Pixels "
            "(confidence >= 0.90)"
        )
    )

    ax.set_ylabel(
        "Percentage of All Valid Pixels (%)"
    )

    ax.set_title(
        "Overall Error vs High-Confidence Error Rate"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0,
        max(error_rate) + 8
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bars in [
        bars1,
        bars2
    ]:

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.6,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig3_high_confidence_failures.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Histogram loader
# ============================================================

def load_histograms(
    condition
):

    path = (
        RESULT_ROOT
        / condition
        / "uncertainty_histograms.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}"
        )

    return np.load(path)


# ============================================================
# Normalize histogram to probability mass
# ============================================================

def normalize_hist(hist):

    total = hist.sum()

    if total == 0:
        return hist.astype(
            np.float64
        )

    return (
        hist.astype(
            np.float64
        )
        / total
    )


# ============================================================
# Figure 4:
# MSP distributions
# ============================================================

def plot_msp_distributions():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for ax, condition in zip(
        axes,
        CONDITIONS
    ):

        data = load_histograms(
            condition
        )

        edges = data[
            "bin_edges"
        ]

        centers = (
            edges[:-1]
            + edges[1:]
        ) / 2.0

        correct = normalize_hist(
            data[
                "msp_correct"
            ]
        )

        incorrect = normalize_hist(
            data[
                "msp_incorrect"
            ]
        )

        ax.plot(
            centers,
            correct,
            label="Correct"
        )

        ax.plot(
            centers,
            incorrect,
            label="Incorrect"
        )

        ax.set_title(
            DISPLAY_NAMES[
                condition
            ]
        )

        ax.set_xlim(
            0,
            1
        )

        ax.set_yscale(
            "log"
        )

        ax.grid(
            alpha=0.2
        )

    axes[0].legend()

    fig.supxlabel(
        "MSP Uncertainty"
    )

    fig.supylabel(
        "Probability Mass (log scale)"
    )

    fig.suptitle(
        "MSP Uncertainty Distributions"
    )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig4_msp_distributions.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Figure 5:
# Entropy distributions
# ============================================================

def plot_entropy_distributions():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for ax, condition in zip(
        axes,
        CONDITIONS
    ):

        data = load_histograms(
            condition
        )

        edges = data[
            "bin_edges"
        ]

        centers = (
            edges[:-1]
            + edges[1:]
        ) / 2.0

        correct = normalize_hist(
            data[
                "entropy_correct"
            ]
        )

        incorrect = normalize_hist(
            data[
                "entropy_incorrect"
            ]
        )

        ax.plot(
            centers,
            correct,
            label="Correct"
        )

        ax.plot(
            centers,
            incorrect,
            label="Incorrect"
        )

        ax.set_title(
            DISPLAY_NAMES[
                condition
            ]
        )

        ax.set_xlim(
            0,
            1
        )

        ax.set_yscale(
            "log"
        )

        ax.grid(
            alpha=0.2
        )

    axes[0].legend()

    fig.supxlabel(
        "Normalized Predictive Entropy"
    )

    fig.supylabel(
        "Probability Mass (log scale)"
    )

    fig.suptitle(
        "Predictive Entropy Distributions"
    )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig5_entropy_distributions.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Text summary
# ============================================================

def write_summary(summary):

    path = (
        OUTPUT_DIR
        / "stage2b_summary.txt"
    )

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "STAGE 2B — DATASET UNCERTAINTY PROFILING\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        for _, row in summary.iterrows():

            condition = str(
                row["condition"]
            )

            f.write(
                f"{DISPLAY_NAMES[condition]}\n"
            )

            f.write(
                f"  Error rate: "
                f"{row['error_rate']*100:.2f}%\n"
            )

            f.write(
                f"  MSP mean: "
                f"correct={row['msp_correct_mean']:.4f}, "
                f"incorrect={row['msp_incorrect_mean']:.4f}, "
                f"gap={row['msp_mean_gap']:.4f}\n"
            )

            f.write(
                f"  Entropy mean: "
                f"correct={row['entropy_correct_mean']:.4f}, "
                f"incorrect={row['entropy_incorrect_mean']:.4f}, "
                f"gap={row['entropy_mean_gap']:.4f}\n"
            )

            f.write(
                f"  High-confidence wrong: "
                f"{row['high_conf_wrong_fraction']*100:.2f}%\n\n"
            )

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "- Incorrect pixels show higher mean uncertainty "
            "than correct pixels in every evaluated condition.\n"
        )

        f.write(
            "- A substantial fraction of incorrect pixels "
            "remain highly confident (softmax >= 0.90).\n"
        )

        f.write(
            "- Night has the highest segmentation error rate "
            "and the highest high-confidence-error fraction.\n"
        )

        f.write(
            "- Mean uncertainty gaps are descriptive only; "
            "formal failure-detection quality must be evaluated "
            "with AUROC/AUPRC in a later stage.\n"
        )

    return path


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = load_summary()

    print_summary(
        summary
    )

    summary_csv = (
        OUTPUT_DIR
        / "stage2b_summary.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False
    )

    fig1 = plot_msp_means(
        summary
    )

    fig2 = plot_entropy_means(
        summary
    )

    fig3 = plot_failure_profile(
        summary
    )

    fig4 = (
        plot_msp_distributions()
    )

    fig5 = (
        plot_entropy_distributions()
    )

    text_summary = (
        write_summary(
            summary
        )
    )

    print("\n")
    print("=" * 78)
    print("STAGE 2B ANALYSIS COMPLETE")
    print("=" * 78)

    print(
        "\nGenerated:"
    )

    for path in [
        summary_csv,
        text_summary,
        fig1,
        fig2,
        fig3,
        fig4,
        fig5,
    ]:

        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()