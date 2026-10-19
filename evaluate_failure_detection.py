import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = Path(
    "results/stage2_uncertainty"
)

OUTPUT_ROOT = (
    ROOT
    / "failure_detection"
)

METHODS = {
    "msp": {
        "correct_key": "msp_correct",
        "incorrect_key": "msp_incorrect",
        "display": "MSP Uncertainty",
    },

    "entropy": {
        "correct_key": "entropy_correct",
        "incorrect_key": "entropy_incorrect",
        "display": "Predictive Entropy",
    },
}


# ============================================================
# Build ROC / PR from histogram
# ============================================================

def compute_curves(
    correct_hist,
    incorrect_hist
):

    """
    Positive class = incorrect pixel
    Negative class = correct pixel

    Higher uncertainty = more likely incorrect.

    Histogram bins run from low uncertainty -> high uncertainty.
    For thresholding we therefore accumulate from high -> low.
    """

    correct_hist = (
        correct_hist
        .astype(np.float64)
    )

    incorrect_hist = (
        incorrect_hist
        .astype(np.float64)
    )

    total_positive = (
        incorrect_hist.sum()
    )

    total_negative = (
        correct_hist.sum()
    )

    if total_positive == 0:
        raise ValueError(
            "No incorrect pixels."
        )

    if total_negative == 0:
        raise ValueError(
            "No correct pixels."
        )

    # Reverse:
    # highest uncertainty bin first
    pos_desc = (
        incorrect_hist[::-1]
    )

    neg_desc = (
        correct_hist[::-1]
    )

    # Threshold progressively moves downward
    tp = np.cumsum(
        pos_desc
    )

    fp = np.cumsum(
        neg_desc
    )

    # Start curve at threshold above maximum:
    # predict nothing as failure
    tp = np.concatenate(
        ([0.0], tp)
    )

    fp = np.concatenate(
        ([0.0], fp)
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    tpr = (
        tp
        / total_positive
    )

    fpr = (
        fp
        / total_negative
    )

    auroc = np.trapezoid(
        tpr,
        fpr
    )

    # --------------------------------------------------------
    # Precision–Recall
    # --------------------------------------------------------

    recall = tpr.copy()

    denominator = (
        tp + fp
    )

    precision = np.ones_like(
        denominator,
        dtype=np.float64
    )

    valid = (
        denominator > 0
    )

    precision[valid] = (
        tp[valid]
        / denominator[valid]
    )

    # Average-precision style area.
    #
    # This is more appropriate than simple trapezoid
    # interpolation for a discrete PR curve.
    delta_recall = np.diff(
        recall
    )

    auprc = np.sum(
        delta_recall
        * precision[1:]
    )

    prevalence = (
        total_positive
        / (
            total_positive
            + total_negative
        )
    )

    return {
        "fpr":
            fpr,

        "tpr":
            tpr,

        "recall":
            recall,

        "precision":
            precision,

        "auroc":
            float(auroc),

        "auprc":
            float(auprc),

        "prevalence":
            float(prevalence),

        "positive_pixels":
            int(total_positive),

        "negative_pixels":
            int(total_negative),
    }


# ============================================================
# Evaluate one condition
# ============================================================

def evaluate(
    condition
):

    print("=" * 82)
    print(
        "STAGE 2D — PIXEL-LEVEL FAILURE DETECTION"
    )
    print("=" * 82)

    print(
        f"Condition: {condition}"
    )

    histogram_path = (
        ROOT
        / condition
        / "uncertainty_histograms.npz"
    )

    if not histogram_path.exists():

        raise FileNotFoundError(
            f"Missing: {histogram_path}"
        )

    data = np.load(
        histogram_path
    )

    bin_edges = data[
        "bin_edges"
    ]

    output_dir = (
        OUTPUT_ROOT
        / condition
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    # ========================================================
    # Compute metrics
    # ========================================================

    for method, config in METHODS.items():

        correct_hist = data[
            config["correct_key"]
        ]

        incorrect_hist = data[
            config["incorrect_key"]
        ]

        result = compute_curves(
            correct_hist,
            incorrect_hist
        )

        results[
            method
        ] = result

    # ========================================================
    # Print summary
    # ========================================================

    prevalence = (
        results["msp"][
            "prevalence"
        ]
    )

    print("\n")
    print("=" * 82)
    print(
        "FAILURE DETECTION SUMMARY"
    )
    print("=" * 82)

    print(
        f"Incorrect-pixel prevalence: "
        f"{prevalence*100:.2f}%"
    )

    print(
        "\nPositive class: incorrect pixel"
    )

    print(
        "Higher uncertainty = more likely failure"
    )

    print("\n")
    print(
        f"{'Method':<24}"
        f"{'AUROC':>12}"
        f"{'AUPRC':>12}"
        f"{'PR Baseline':>14}"
    )

    print("-" * 62)

    for method in [
        "msp",
        "entropy"
    ]:

        result = results[
            method
        ]

        print(
            f"{METHODS[method]['display']:<24}"
            f"{result['auroc']:>12.4f}"
            f"{result['auprc']:>12.4f}"
            f"{result['prevalence']:>14.4f}"
        )

    # ========================================================
    # ROC figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.5, 6)
    )

    for method in [
        "msp",
        "entropy"
    ]:

        result = results[
            method
        ]

        ax.plot(
            result["fpr"],
            result["tpr"],
            label=(
                f"{METHODS[method]['display']} "
                f"(AUROC={result['auroc']:.3f})"
            )
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        f"{condition.capitalize()} — "
        f"Failure Detection ROC"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    roc_path = (
        output_dir
        / "roc_curve.png"
    )

    fig.savefig(
        roc_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # Precision–Recall figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.5, 6)
    )

    for method in [
        "msp",
        "entropy"
    ]:

        result = results[
            method
        ]

        ax.plot(
            result["recall"],
            result["precision"],
            label=(
                f"{METHODS[method]['display']} "
                f"(AUPRC={result['auprc']:.3f})"
            )
        )

    ax.axhline(
        prevalence,
        linestyle="--",
        label=(
            f"Random baseline "
            f"({prevalence:.3f})"
        )
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        f"{condition.capitalize()} — "
        f"Failure Detection Precision–Recall"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    pr_path = (
        output_dir
        / "precision_recall_curve.png"
    )

    fig.savefig(
        pr_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # Save summary
    # ========================================================

    summary_rows = []

    for method in [
        "msp",
        "entropy"
    ]:

        result = results[
            method
        ]

        summary_rows.append({
            "condition":
                condition,

            "method":
                method,

            "auroc":
                result["auroc"],

            "auprc":
                result["auprc"],

            "positive_prevalence":
                result["prevalence"],

            "positive_pixels":
                result["positive_pixels"],

            "negative_pixels":
                result["negative_pixels"],

            "histogram_bins":
                len(bin_edges) - 1,
        })

    summary_csv = (
        output_dir
        / "summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # ========================================================
    # Finish
    # ========================================================

    print("\n")
    print("=" * 82)
    print(
        "FILES SAVED"
    )
    print("=" * 82)

    print(
        summary_csv
    )

    print(
        roc_path
    )

    print(
        pr_path
    )

    print(
        "\nNote: metrics are computed from the "
        "1000-bin Stage 2B histograms."
    )

    print(
        "They are therefore extremely close "
        "histogram-based approximations to the "
        "raw pixel-level metrics."
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "fog",
            "rain",
            "snow",
            "night",
        ]
    )

    args = parser.parse_args()

    evaluate(
        args.condition
    )