from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = Path(
    "results/stage2_uncertainty/"
    "paired_normal_vs_adverse"
)

OUTPUT_DIR = (
    ROOT
    / "stage2c_analysis"
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
# Load data
# ============================================================

def load_summaries():

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

        if len(df) != 1:

            raise ValueError(
                f"Expected one row: {path}"
            )

        rows.append(
            df.iloc[0].to_dict()
        )

    result = pd.DataFrame(
        rows
    )

    result["condition"] = pd.Categorical(
        result["condition"],
        categories=CONDITIONS,
        ordered=True
    )

    result = (
        result
        .sort_values("condition")
        .reset_index(drop=True)
    )

    result["miou_drop_pp"] = (
        -result["miou_delta"]
        * 100.0
    )

    result["accuracy_drop_pp"] = (
        -result["accuracy_delta"]
        * 100.0
    )

    return result


# ============================================================
# Console summary
# ============================================================

def print_summary(df):

    print("\n")
    print("=" * 118)
    print(
        "STAGE 2C — PAIRED NORMAL VS "
        "ADVERSE UNCERTAINTY SUMMARY"
    )
    print("=" * 118)

    print(
        f"{'Condition':<10}"
        f"{'Pairs':>7}"
        f"{'mIoU Drop (pp)':>15}"
        f"{'Acc Drop (pp)':>15}"
        f"{'MSP Delta':>12}"
        f"{'Ent Delta':>12}"
        f"{'Acc↓':>10}"
        f"{'MSP↑':>10}"
        f"{'Ent↑':>10}"
    )

    print("-" * 118)

    for _, row in df.iterrows():

        condition = str(
            row["condition"]
        )

        pairs = int(
            row["pairs"]
        )

        print(
            f"{DISPLAY[condition]:<10}"
            f"{pairs:>7d}"
            f"{row['miou_drop_pp']:>12.2f} pp"
            f"{row['accuracy_drop_pp']:>12.2f} pp"
            f"{row['msp_global_delta']:>12.4f}"
            f"{row['entropy_global_delta']:>12.4f}"
            f"{int(row['accuracy_pairs_decreased']):>6d}/{pairs:<3d}"
            f"{int(row['msp_pairs_increased']):>6d}/{pairs:<3d}"
            f"{int(row['entropy_pairs_increased']):>6d}/{pairs:<3d}"
        )

# ============================================================
# Figure 1: paired mIoU
# ============================================================

def plot_miou(df):

    labels = [
        DISPLAY[str(x)]
        for x in df["condition"]
    ]

    normal = (
        df["normal_miou"]
        .to_numpy()
        * 100.0
    )

    adverse = (
        df["adverse_miou"]
        .to_numpy()
        * 100.0
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
        normal,
        width,
        label="Normal Reference"
    )

    b2 = ax.bar(
        x + width / 2,
        adverse,
        width,
        label="Adverse"
    )

    ax.set_ylabel(
        "mIoU (%)"
    )

    ax.set_title(
        "Paired Segmentation Performance"
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

    for bars in [b1, b2]:

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                h + 0.7,
                f"{h:.1f}",
                ha="center",
                fontsize=9
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig1_paired_miou.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Figure 2: paired MSP
# ============================================================

def plot_msp(df):

    labels = [
        DISPLAY[str(x)]
        for x in df["condition"]
    ]

    normal = (
        df["normal_msp_mean"]
        .to_numpy()
    )

    adverse = (
        df["adverse_msp_mean"]
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
        normal,
        width,
        label="Normal Reference"
    )

    b2 = ax.bar(
        x + width / 2,
        adverse,
        width,
        label="Adverse"
    )

    ax.set_ylabel(
        "Mean MSP Uncertainty"
    )

    ax.set_title(
        "Paired MSP Uncertainty"
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

    for bars in [b1, b2]:

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                h + 0.002,
                f"{h:.3f}",
                ha="center",
                fontsize=9
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig2_paired_msp_uncertainty.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Figure 3: paired entropy
# ============================================================

def plot_entropy(df):

    labels = [
        DISPLAY[str(x)]
        for x in df["condition"]
    ]

    normal = (
        df["normal_entropy_mean"]
        .to_numpy()
    )

    adverse = (
        df["adverse_entropy_mean"]
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
        normal,
        width,
        label="Normal Reference"
    )

    b2 = ax.bar(
        x + width / 2,
        adverse,
        width,
        label="Adverse"
    )

    ax.set_ylabel(
        "Mean Normalized Predictive Entropy"
    )

    ax.set_title(
        "Paired Predictive Entropy"
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

    for bars in [b1, b2]:

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                h + 0.002,
                f"{h:.3f}",
                ha="center",
                fontsize=9
            )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig3_paired_entropy.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# Figure 4: degradation vs uncertainty increase
# ============================================================

def plot_degradation_vs_uncertainty(df):

    degradation = (
        df["miou_drop_pp"]
        .to_numpy()
    )

    msp_delta = (
        df["msp_global_delta"]
        .to_numpy()
    )

    entropy_delta = (
        df["entropy_global_delta"]
        .to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        degradation,
        msp_delta,
        s=90,
        label="MSP"
    )

    ax.scatter(
        degradation,
        entropy_delta,
        s=90,
        marker="s",
        label="Predictive Entropy"
    )

    for i, condition in enumerate(
        CONDITIONS
    ):

        ax.annotate(
            DISPLAY[condition],
            (
                degradation[i],
                msp_delta[i]
            ),
            xytext=(6, 6),
            textcoords="offset points"
        )

        ax.annotate(
            DISPLAY[condition],
            (
                degradation[i],
                entropy_delta[i]
            ),
            xytext=(6, -13),
            textcoords="offset points"
        )

    ax.set_xlabel(
        "Paired mIoU Degradation "
        "(percentage points)"
    )

    ax.set_ylabel(
        "Increase in Mean Uncertainty"
    )

    ax.set_title(
        "Performance Degradation vs "
        "Uncertainty Increase"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "fig4_degradation_vs_uncertainty.png"
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
        / "stage2c_summary.txt"
    )

    miou_drop = (
        df["miou_drop_pp"]
        .to_numpy()
    )

    msp_delta = (
        df["msp_global_delta"]
        .to_numpy()
    )

    entropy_delta = (
        df["entropy_global_delta"]
        .to_numpy()
    )

    msp_corr = np.corrcoef(
        miou_drop,
        msp_delta
    )[0, 1]

    entropy_corr = np.corrcoef(
        miou_drop,
        entropy_delta
    )[0, 1]

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "STAGE 2C — PAIRED NORMAL VS "
            "ADVERSE UNCERTAINTY\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for _, row in df.iterrows():

            condition = str(
                row["condition"]
            )

            pairs = int(
                row["pairs"]
            )

            f.write(
                f"{DISPLAY[condition]}\n"
            )

            f.write(
                f"  Pairs: {pairs}\n"
            )

            f.write(
                f"  mIoU: "
                f"{row['normal_miou']:.4f} -> "
                f"{row['adverse_miou']:.4f} "
                f"(-{row['miou_drop_pp']:.2f} pp)\n"
            )

            f.write(
                f"  MSP uncertainty: "
                f"{row['normal_msp_mean']:.4f} -> "
                f"{row['adverse_msp_mean']:.4f} "
                f"(delta "
                f"{row['msp_global_delta']:+.4f})\n"
            )

            f.write(
                f"  Predictive entropy: "
                f"{row['normal_entropy_mean']:.4f} -> "
                f"{row['adverse_entropy_mean']:.4f} "
                f"(delta "
                f"{row['entropy_global_delta']:+.4f})\n"
            )

            f.write(
                f"  Accuracy decreased: "
                f"{int(row['accuracy_pairs_decreased'])}/"
                f"{pairs}\n"
            )

            f.write(
                f"  MSP increased: "
                f"{int(row['msp_pairs_increased'])}/"
                f"{pairs}\n"
            )

            f.write(
                f"  Entropy increased: "
                f"{int(row['entropy_pairs_increased'])}/"
                f"{pairs}\n\n"
            )

        f.write(
            "Interpretation\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            "1. All four adverse conditions showed "
            "higher aggregate MSP uncertainty and "
            "predictive entropy than their paired "
            "normal-reference subsets.\n"
        )

        f.write(
            "2. Larger aggregate performance degradation "
            "was accompanied by larger uncertainty "
            "increases across the evaluated conditions.\n"
        )

        f.write(
            "3. Night showed the strongest effect: all "
            "53 pairs decreased in accuracy and increased "
            "in both uncertainty measures.\n"
        )

        f.write(
            "4. The four-condition descriptive correlation "
            "between mIoU degradation and uncertainty "
            "increase was:\n"
        )

        f.write(
            f"   MSP:     r={msp_corr:.4f}\n"
        )

        f.write(
            f"   Entropy: r={entropy_corr:.4f}\n"
        )

        f.write(
            "   These correlations are descriptive only "
            "because there are only four condition-level "
            "observations.\n"
        )

        f.write(
            "5. These aggregate results do not imply that "
            "all pixel-level failures are detected; Stage "
            "2B showed substantial high-confidence errors.\n"
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

    df = load_summaries()

    print_summary(
        df
    )

    output_csv = (
        OUTPUT_DIR
        / "stage2c_summary.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    fig1 = plot_miou(df)
    fig2 = plot_msp(df)
    fig3 = plot_entropy(df)

    fig4 = (
        plot_degradation_vs_uncertainty(
            df
        )
    )

    text = write_summary(
        df
    )

    print("\n")
    print("=" * 82)
    print(
        "STAGE 2C ANALYSIS COMPLETE"
    )
    print("=" * 82)

    print(
        "\nGenerated:"
    )

    for path in [
        output_csv,
        text,
        fig1,
        fig2,
        fig3,
        fig4,
    ]:

        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()