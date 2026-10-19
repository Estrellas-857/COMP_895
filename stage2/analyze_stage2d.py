from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = Path(
    "results/stage2_uncertainty/"
    "failure_detection"
)

OUTPUT_DIR = (
    ROOT
    / "stage2d_analysis"
)

CONDITIONS = [
    "fog",
    "rain",
    "snow",
    "night",
]

DISPLAY = {
    "fog": "Fog",
    "rain": "Rain",
    "snow": "Snow",
    "night": "Night",
}


# ============================================================
# Load summaries
# ============================================================

def load_data():

    rows = []

    for condition in CONDITIONS:

        path = (
            ROOT
            / condition
            / "summary.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing: {path}"
            )

        df = pd.read_csv(
            path
        )

        if len(df) != 2:

            raise ValueError(
                f"Expected two methods in {path}"
            )

        for _, row in df.iterrows():

            rows.append(
                row.to_dict()
            )

    df = pd.DataFrame(
        rows
    )

    df["condition"] = pd.Categorical(
        df["condition"],
        categories=CONDITIONS,
        ordered=True
    )

    df = (
        df
        .sort_values([
            "condition",
            "method"
        ])
        .reset_index(drop=True)
    )

    # Improvement above random PR baseline
    df["auprc_gain"] = (
        df["auprc"]
        - df["positive_prevalence"]
    )

    return df


# ============================================================
# Console summary
# ============================================================

def print_summary(df):

    print("\n")
    print("=" * 104)

    print(
        "STAGE 2D — PIXEL-LEVEL "
        "FAILURE DETECTION SUMMARY"
    )

    print("=" * 104)

    print(
        f"{'Condition':<10}"
        f"{'Method':<12}"
        f"{'Prevalence':>12}"
        f"{'AUROC':>10}"
        f"{'AUPRC':>10}"
        f"{'PR Gain':>10}"
    )

    print("-" * 104)

    for _, row in df.iterrows():

        condition = str(
            row["condition"]
        )

        method = (
            "MSP"
            if row["method"] == "msp"
            else "Entropy"
        )

        print(
            f"{DISPLAY[condition]:<10}"
            f"{method:<12}"
            f"{row['positive_prevalence']:>12.4f}"
            f"{row['auroc']:>10.4f}"
            f"{row['auprc']:>10.4f}"
            f"{row['auprc_gain']:>10.4f}"
        )


# ============================================================
# Figure 1 — AUROC
# ============================================================

def plot_auroc(df):

    pivot = df.pivot(
        index="condition",
        columns="method",
        values="auroc"
    )

    pivot = pivot.reindex(
        CONDITIONS
    )

    labels = [
        DISPLAY[c]
        for c in CONDITIONS
    ]

    msp = (
        pivot["msp"]
        .to_numpy()
    )

    entropy = (
        pivot["entropy"]
        .to_numpy()
    )

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    b1 = ax.bar(
        x - width / 2,
        msp,
        width,
        label="MSP"
    )

    b2 = ax.bar(
        x + width / 2,
        entropy,
        width,
        label="Predictive Entropy"
    )

    ax.axhline(
        0.5,
        linestyle="--",
        label="Random Baseline"
    )

    ax.set_ylabel(
        "AUROC"
    )

    ax.set_title(
        "Pixel-Level Failure Detection AUROC"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0.45,
        1.0
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bars in [
        b1,
        b2
    ]:

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                h + 0.008,
                f"{h:.3f}",
                ha="center",
                fontsize=9
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig1_failure_detection_auroc.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Figure 2 — AUPRC + baseline
# ============================================================

def plot_auprc(df):

    msp = (
        df[
            df["method"] == "msp"
        ]
        .set_index("condition")
        .reindex(CONDITIONS)
    )

    entropy = (
        df[
            df["method"] == "entropy"
        ]
        .set_index("condition")
        .reindex(CONDITIONS)
    )

    labels = [
        DISPLAY[c]
        for c in CONDITIONS
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.30

    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
    )

    b1 = ax.bar(
        x - width,
        msp["auprc"],
        width,
        label="MSP AUPRC"
    )

    b2 = ax.bar(
        x,
        entropy["auprc"],
        width,
        label="Entropy AUPRC"
    )

    b3 = ax.bar(
        x + width,
        msp[
            "positive_prevalence"
        ],
        width,
        label="Random PR Baseline"
    )

    ax.set_ylabel(
        "Precision–Recall Score"
    )

    ax.set_title(
        "Failure Detection AUPRC vs "
        "Condition-Specific Baseline"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0,
        0.75
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bars in [
        b1,
        b2,
        b3
    ]:

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                fontsize=8
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig2_failure_detection_auprc.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Figure 3 — PR gain over baseline
# ============================================================

def plot_pr_gain(df):

    pivot = df.pivot(
        index="condition",
        columns="method",
        values="auprc_gain"
    )

    pivot = pivot.reindex(
        CONDITIONS
    )

    labels = [
        DISPLAY[c]
        for c in CONDITIONS
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    b1 = ax.bar(
        x - width / 2,
        pivot["msp"],
        width,
        label="MSP"
    )

    b2 = ax.bar(
        x + width / 2,
        pivot["entropy"],
        width,
        label="Predictive Entropy"
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_ylabel(
        "AUPRC − Random Baseline"
    )

    ax.set_title(
        "Failure-Detection PR Improvement "
        "over Random Baseline"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bars in [
        b1,
        b2
    ]:

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                h + 0.008,
                f"{h:.3f}",
                ha="center",
                fontsize=9
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig3_auprc_gain_over_baseline.png"
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

def write_summary(df):

    path = (
        OUTPUT_DIR
        / "stage2d_summary.txt"
    )

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "STAGE 2D — PIXEL-LEVEL "
            "FAILURE DETECTION\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for condition in CONDITIONS:

            subset = df[
                df["condition"]
                == condition
            ]

            f.write(
                f"{DISPLAY[condition]}\n"
            )

            prevalence = float(
                subset[
                    "positive_prevalence"
                ].iloc[0]
            )

            f.write(
                f"  Error prevalence: "
                f"{prevalence:.4f}\n"
            )

            for method in [
                "msp",
                "entropy"
            ]:

                row = subset[
                    subset["method"]
                    == method
                ].iloc[0]

                name = (
                    "MSP"
                    if method == "msp"
                    else "Predictive Entropy"
                )

                f.write(
                    f"  {name}: "
                    f"AUROC={row['auroc']:.4f}, "
                    f"AUPRC={row['auprc']:.4f}, "
                    f"PR gain={row['auprc_gain']:.4f}\n"
                )

            f.write("\n")

        f.write(
            "Interpretation\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            "1. Both MSP uncertainty and predictive "
            "entropy detected incorrect pixels better "
            "than random ranking in all four adverse "
            "conditions.\n"
        )

        f.write(
            "2. Failure-detection AUROC was strongest "
            "for rain and fog and weaker for snow and "
            "night.\n"
        )

        f.write(
            "3. Night had the lowest AUROC despite "
            "showing the largest aggregate uncertainty "
            "increase in Stage 2C. This indicates that "
            "higher global uncertainty does not "
            "necessarily imply better pixel-level "
            "failure discrimination.\n"
        )

        f.write(
            "4. Predictive entropy had a small numerical "
            "AUROC advantage over MSP in all conditions, "
            "but the differences were small.\n"
        )

        f.write(
            "5. AUPRC must be interpreted relative to "
            "the condition-specific error prevalence "
            "because the positive-class rate differs "
            "substantially across conditions.\n"
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

    df = load_data()

    print_summary(
        df
    )

    csv_path = (
        OUTPUT_DIR
        / "stage2d_summary.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    fig1 = plot_auroc(
        df
    )

    fig2 = plot_auprc(
        df
    )

    fig3 = plot_pr_gain(
        df
    )

    text_path = write_summary(
        df
    )

    print("\n")
    print("=" * 82)

    print(
        "STAGE 2D ANALYSIS COMPLETE"
    )

    print("=" * 82)

    print(
        "\nGenerated:"
    )

    for path in [
        csv_path,
        text_path,
        fig1,
        fig2,
        fig3,
    ]:

        print(
            f"  {path}"
        )


if __name__ == "__main__":

    main()