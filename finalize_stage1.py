from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

RESULT_ROOT = Path("results")

BASELINE_DIR = (
    RESULT_ROOT
    / "baseline_analysis"
)

PAIRED_DIR = (
    RESULT_ROOT
    / "paired_normal_vs_adverse"
)

FAILURE_DIR = (
    RESULT_ROOT
    / "failure_cases"
)

OUTPUT_DIR = (
    RESULT_ROOT
    / "stage1_wrapup"
)

CONDITIONS = [
    "fog",
    "rain",
    "snow",
    "night"
]

DISPLAY_NAMES = {
    "fog": "Fog",
    "rain": "Rain",
    "snow": "Snow",
    "night": "Night"
}


# ============================================================
# Input paths
# ============================================================

BASELINE_SUMMARY_PATH = (
    BASELINE_DIR
    / "condition_summary.csv"
)

CLASS_IOU_PATH = (
    BASELINE_DIR
    / "class_iou_by_condition.csv"
)

CLASS_SUPPORT_PATH = (
    BASELINE_DIR
    / "class_support_by_condition.csv"
)

PAIRED_SUMMARY_PATH = (
    PAIRED_DIR
    / "paired_condition_summary.csv"
)


# ============================================================
# Load data
# ============================================================

def load_results():

    required_files = [
        BASELINE_SUMMARY_PATH,
        CLASS_IOU_PATH,
        CLASS_SUPPORT_PATH,
        PAIRED_SUMMARY_PATH,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    baseline = pd.read_csv(
        BASELINE_SUMMARY_PATH
    )

    class_iou = pd.read_csv(
        CLASS_IOU_PATH
    )

    class_support = pd.read_csv(
        CLASS_SUPPORT_PATH
    )

    paired = pd.read_csv(
        PAIRED_SUMMARY_PATH
    )

    return (
        baseline,
        class_iou,
        class_support,
        paired
    )


# ============================================================
# Validation
# ============================================================

def validate_results(
    baseline,
    class_iou,
    paired
):

    baseline_conditions = set(
        baseline["condition"]
    )

    paired_conditions = set(
        paired["condition"]
    )

    expected = set(
        CONDITIONS
    )

    if baseline_conditions != expected:

        raise ValueError(
            "Baseline conditions do not match "
            f"{expected}. Found: "
            f"{baseline_conditions}"
        )

    if paired_conditions != expected:

        raise ValueError(
            "Paired conditions do not match "
            f"{expected}. Found: "
            f"{paired_conditions}"
        )

    if len(class_iou) != 19:

        raise ValueError(
            "Expected 19 Cityscapes classes, "
            f"but found {len(class_iou)}."
        )


# ============================================================
# Stage 1 overview table
# ============================================================

def build_stage1_overview(
    baseline,
    paired
):

    baseline_small = baseline[
        [
            "condition",
            "images",
            "pixel_accuracy",
            "miou_union",
            "miou_gt",
            "latency_ms"
        ]
    ].copy()

    baseline_small = (
        baseline_small.rename(
            columns={
                "images":
                    "full_adverse_images",

                "pixel_accuracy":
                    "full_adverse_pixel_accuracy",

                "miou_union":
                    "full_adverse_miou",

                "miou_gt":
                    "full_adverse_miou_gt",

                "latency_ms":
                    "full_adverse_latency_ms",
            }
        )
    )

    paired_small = paired[
        [
            "condition",
            "pairs",

            "normal_pixel_accuracy",
            "adverse_pixel_accuracy",
            "pixel_accuracy_delta",

            "normal_miou_union",
            "adverse_miou_union",
            "miou_union_delta",

            "normal_miou_gt",
            "adverse_miou_gt",
            "miou_gt_delta",

            "normal_latency_ms",
            "adverse_latency_ms",
        ]
    ].copy()

    overview = pd.merge(
        baseline_small,
        paired_small,
        on="condition",
        how="inner"
    )

    order = {
        condition: index
        for index, condition
        in enumerate(CONDITIONS)
    }

    overview["_order"] = (
        overview["condition"]
        .map(order)
    )

    overview = (
        overview
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return overview


# ============================================================
# Figure 1:
# Full adverse baseline
# ============================================================

def plot_full_adverse_baseline(
    baseline
):

    df = (
        baseline
        .set_index("condition")
        .loc[CONDITIONS]
        .reset_index()
    )

    labels = [
        DISPLAY_NAMES[c]
        for c in df["condition"]
    ]

    miou = (
        df["miou_union"]
        .to_numpy()
        * 100.0
    )

    pixel_acc = (
        df["pixel_accuracy"]
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

    bars1 = ax.bar(
        x - width / 2,
        miou,
        width,
        label="mIoU"
    )

    bars2 = ax.bar(
        x + width / 2,
        pixel_acc,
        width,
        label="Pixel Accuracy"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_title(
        "SegFormer-B0 Performance "
        "under ACDC Adverse Conditions"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0,
        100
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
                height + 1.5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "fig1_full_adverse_baseline.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# Figure 2:
# Paired normal vs adverse mIoU
# ============================================================

def plot_paired_miou(
    paired
):

    df = (
        paired
        .set_index("condition")
        .loc[CONDITIONS]
        .reset_index()
    )

    labels = [
        DISPLAY_NAMES[c]
        for c in df["condition"]
    ]

    normal = (
        df["normal_miou_union"]
        .to_numpy()
        * 100.0
    )

    adverse = (
        df["adverse_miou_union"]
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

    bars1 = ax.bar(
        x - width / 2,
        normal,
        width,
        label="Normal Reference"
    )

    bars2 = ax.bar(
        x + width / 2,
        adverse,
        width,
        label="Adverse"
    )

    ax.set_ylabel(
        "mIoU (%)"
    )

    ax.set_title(
        "Paired Normal-Reference vs "
        "Adverse Performance"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0,
        max(
            normal.max(),
            adverse.max()
        ) + 12
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
                height + 1.0,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "fig2_paired_normal_vs_adverse.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# Figure 3:
# Paired degradation
# ============================================================

def plot_paired_degradation(
    paired
):

    df = (
        paired
        .set_index("condition")
        .loc[CONDITIONS]
        .reset_index()
    )

    labels = [
        DISPLAY_NAMES[c]
        for c in df["condition"]
    ]

    # Convert negative deltas into
    # positive "degradation magnitude"
    degradation = (
        -df["miou_union_delta"]
        .to_numpy()
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        labels,
        degradation
    )

    ax.set_ylabel(
        "mIoU Degradation "
        "(percentage points)"
    )

    ax.set_title(
        "Performance Degradation from "
        "Normal Reference to Adverse Condition"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        degradation
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.5,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "fig3_paired_degradation.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# Figure 4:
# Per-class IoU heatmap
# ============================================================

def plot_class_iou_heatmap(
    class_iou
):

    matrix = (
        class_iou[
            CONDITIONS
        ]
        .to_numpy(
            dtype=float
        )
    )

    class_names = (
        class_iou[
            "class_name"
        ]
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(9, 10)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1
    )

    ax.set_xticks(
        np.arange(
            len(CONDITIONS)
        )
    )

    ax.set_xticklabels(
        [
            DISPLAY_NAMES[c]
            for c in CONDITIONS
        ]
    )

    ax.set_yticks(
        np.arange(
            len(class_names)
        )
    )

    ax.set_yticklabels(
        class_names
    )

    ax.set_title(
        "Per-Class IoU across "
        "Adverse Conditions"
    )

    for row in range(
        matrix.shape[0]
    ):

        for col in range(
            matrix.shape[1]
        ):

            value = (
                matrix[row, col]
            )

            if np.isnan(value):
                text = "N/A"
            else:
                text = (
                    f"{value:.2f}"
                )

            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7
            )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "IoU"
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "fig4_class_iou_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# Representative failure manifest
# ============================================================

def find_rank_one_image(
    condition,
    class_name
):

    folder = (
        FAILURE_DIR
        / condition
        / class_name
    )

    if not folder.exists():

        return None

    candidates = sorted(
        folder.glob(
            "rank_01_*.png"
        )
    )

    if len(candidates) == 0:

        return None

    return candidates[0]


def write_failure_manifest():

    snow_terrain = (
        find_rank_one_image(
            "snow",
            "terrain"
        )
    )

    night_sky = (
        find_rank_one_image(
            "night",
            "sky"
        )
    )

    snow_bus = (
        find_rank_one_image(
            "snow",
            "bus"
        )
    )

    output_path = (
        OUTPUT_DIR
        / "representative_failures.txt"
    )

    with open(
        output_path,
        "w"
    ) as f:

        f.write(
            "Stage 1 Representative "
            "Failure Cases\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            "1. Snow terrain\n"
        )

        f.write(
            "   Interpretation: "
            "systematic terrain-to-"
            "sidewalk/road confusion "
            "under snow-covered ground "
            "appearance.\n"
        )

        f.write(
            f"   Figure: "
            f"{snow_terrain}\n\n"
        )

        f.write(
            "2. Night sky\n"
        )

        f.write(
            "   Interpretation: "
            "systematic nighttime sky "
            "confusion with building, "
            "vegetation, and road.\n"
        )

        f.write(
            f"   Figure: "
            f"{night_sky}\n\n"
        )

        f.write(
            "3. Snow bus\n"
        )

        f.write(
            "   Interpretation: "
            "object/sequence-dependent "
            "failure involving reflection, "
            "occlusion, distance, and "
            "vehicle-class confusion. "
            "Not treated as the primary "
            "systematic snow failure.\n"
        )

        f.write(
            f"   Figure: "
            f"{snow_bus}\n"
        )

    return output_path


# ============================================================
# Text summary
# ============================================================

def write_stage1_summary(
    baseline,
    paired
):

    baseline = (
        baseline
        .set_index("condition")
        .loc[CONDITIONS]
    )

    paired = (
        paired
        .set_index("condition")
        .loc[CONDITIONS]
    )

    output_path = (
        OUTPUT_DIR
        / "stage1_summary.txt"
    )

    with open(
        output_path,
        "w"
    ) as f:

        f.write(
            "STAGE 1 WRAP-UP\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        f.write(
            "Model\n"
        )

        f.write(
            "-----\n"
        )

        f.write(
            "SegFormer-B0 fine-tuned "
            "on Cityscapes\n"
        )

        f.write(
            "Inference input: "
            "512 x 1024\n"
        )

        f.write(
            "ACDC fine-tuning: none\n"
        )

        f.write(
            "Semantic classes: 19 "
            "Cityscapes trainIds\n\n"
        )

        # ----------------------------------------------------
        # Full adverse baseline
        # ----------------------------------------------------

        f.write(
            "Full Adverse Validation "
            "Baseline\n"
        )

        f.write(
            "------------------------"
            "---------\n"
        )

        for condition in CONDITIONS:

            row = baseline.loc[
                condition
            ]

            f.write(
                f"{DISPLAY_NAMES[condition]:<8} "
                f"images={int(row['images']):3d}, "
                f"mIoU={row['miou_union']*100:.2f}%, "
                f"pixel_acc="
                f"{row['pixel_accuracy']*100:.2f}%, "
                f"latency="
                f"{row['latency_ms']:.2f} ms\n"
            )

        # ----------------------------------------------------
        # Paired results
        # ----------------------------------------------------

        f.write(
            "\nPaired Normal Reference "
            "vs Adverse\n"
        )

        f.write(
            "--------------------------"
            "-------\n"
        )

        for condition in CONDITIONS:

            row = paired.loc[
                condition
            ]

            f.write(
                f"{DISPLAY_NAMES[condition]:<8} "
                f"pairs={int(row['pairs']):2d}, "
                f"normal_mIoU="
                f"{row['normal_miou_union']*100:.2f}%, "
                f"adverse_mIoU="
                f"{row['adverse_miou_union']*100:.2f}%, "
                f"delta="
                f"{row['miou_union_delta']*100:.2f} pp\n"
            )

        # ----------------------------------------------------
        # Key observations
        # ----------------------------------------------------

        f.write(
            "\nKey Stage 1 Observations\n"
        )

        f.write(
            "------------------------\n"
        )

        f.write(
            "1. Fog produced the strongest "
            "full adverse-condition baseline "
            "performance.\n"
        )

        f.write(
            "2. Rain and snow produced "
            "moderate degradation.\n"
        )

        f.write(
            "3. Nighttime produced the "
            "largest degradation by a wide "
            "margin.\n"
        )

        f.write(
            "4. In paired evaluation, "
            "night mIoU decreased from "
            f"{paired.loc['night', 'normal_miou_union']*100:.2f}% "
            "to "
            f"{paired.loc['night', 'adverse_miou_union']*100:.2f}%, "
            "a change of "
            f"{paired.loc['night', 'miou_union_delta']*100:.2f} "
            "percentage points.\n"
        )

        f.write(
            "5. Snow terrain showed a "
            "systematic confusion pattern "
            "toward sidewalk and road.\n"
        )

        f.write(
            "6. Night sky showed severe "
            "systematic confusion toward "
            "building and vegetation.\n"
        )

        f.write(
            "7. Snow bus failure was "
            "strongly influenced by a small "
            "number of difficult large-object "
            "cases and is therefore treated "
            "as object/sequence-dependent.\n"
        )

        f.write(
            "8. Model-forward latency stayed "
            "approximately constant across "
            "conditions at about 20-21 ms "
            "per image on the RTX 3060.\n"
        )

        f.write(
            "\nStage 1 Conclusion\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            "The baseline confirms that "
            "adverse-condition degradation "
            "is strongly condition- and "
            "class-dependent. Nighttime "
            "produces the most severe overall "
            "domain shift, while snow creates "
            "specific semantic confusions such "
            "as terrain versus sidewalk/road. "
            "These verified failure modes "
            "provide the basis for Stage 2, "
            "which will test whether uncertainty "
            "estimation can identify unreliable "
            "predictions automatically.\n"
        )

    return output_path


# ============================================================
# Save normalized tables
# ============================================================

def save_tables(
    overview,
    class_iou,
    class_support
):

    overview_path = (
        OUTPUT_DIR
        / "stage1_overview.csv"
    )

    class_iou_path = (
        OUTPUT_DIR
        / "stage1_class_iou.csv"
    )

    class_support_path = (
        OUTPUT_DIR
        / "stage1_class_support.csv"
    )

    overview.to_csv(
        overview_path,
        index=False
    )

    class_iou.to_csv(
        class_iou_path,
        index=False
    )

    class_support.to_csv(
        class_support_path,
        index=False
    )

    return (
        overview_path,
        class_iou_path,
        class_support_path
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        baseline,
        class_iou,
        class_support,
        paired
    ) = load_results()

    validate_results(
        baseline,
        class_iou,
        paired
    )

    # --------------------------------------------------------
    # Build merged overview
    # --------------------------------------------------------

    overview = (
        build_stage1_overview(
            baseline,
            paired
        )
    )

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    (
        overview_path,
        class_iou_path,
        class_support_path
    ) = save_tables(
        overview,
        class_iou,
        class_support
    )

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    fig1 = (
        plot_full_adverse_baseline(
            baseline
        )
    )

    fig2 = (
        plot_paired_miou(
            paired
        )
    )

    fig3 = (
        plot_paired_degradation(
            paired
        )
    )

    fig4 = (
        plot_class_iou_heatmap(
            class_iou
        )
    )

    # --------------------------------------------------------
    # Failure manifest
    # --------------------------------------------------------

    failure_manifest = (
        write_failure_manifest()
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    summary_path = (
        write_stage1_summary(
            baseline,
            paired
        )
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("STAGE 1 WRAP-UP COMPLETE")
    print("=" * 78)

    print(
        "\nFull adverse baseline:"
    )

    for condition in CONDITIONS:

        row = (
            baseline[
                baseline["condition"]
                == condition
            ]
            .iloc[0]
        )

        print(
            f"{DISPLAY_NAMES[condition]:<8} "
            f"mIoU="
            f"{row['miou_union']*100:.2f}% | "
            f"Acc="
            f"{row['pixel_accuracy']*100:.2f}% | "
            f"Latency="
            f"{row['latency_ms']:.2f} ms"
        )

    print(
        "\nPaired degradation:"
    )

    for condition in CONDITIONS:

        row = (
            paired[
                paired["condition"]
                == condition
            ]
            .iloc[0]
        )

        print(
            f"{DISPLAY_NAMES[condition]:<8} "
            f"{row['normal_miou_union']*100:.2f}% "
            f"-> "
            f"{row['adverse_miou_union']*100:.2f}% "
            f"("
            f"{row['miou_union_delta']*100:.2f} pp"
            f")"
        )

    print(
        "\nGenerated files:"
    )

    generated = [
        overview_path,
        class_iou_path,
        class_support_path,
        summary_path,
        fig1,
        fig2,
        fig3,
        fig4,
        failure_manifest,
    ]

    for path in generated:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()