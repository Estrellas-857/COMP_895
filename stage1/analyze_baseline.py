import csv
import re
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

RESULT_ROOT = Path("results")

CONDITIONS = [
    "fog",
    "rain",
    "snow",
    "night"
]

NUM_CLASSES = 19

CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

OUTPUT_DIR = RESULT_ROOT / "baseline_analysis"


# ============================================================
# Metric computation
# ============================================================

def compute_metrics(confusion):
    """
    Compute:
      - per-class IoU
      - mIoU_union:
            average over classes with union > 0
      - mIoU_gt:
            average only over classes present in GT
      - pixel accuracy
      - GT support
      - prediction support
    """

    tp = np.diag(confusion).astype(np.float64)

    gt_pixels = confusion.sum(axis=1).astype(np.float64)
    pred_pixels = confusion.sum(axis=0).astype(np.float64)

    union = gt_pixels + pred_pixels - tp

    iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)

    union_valid = union > 0

    iou[union_valid] = (
        tp[union_valid]
        / union[union_valid]
    )

    # Current evaluator-style mIoU:
    # include any class with union > 0
    miou_union = np.nanmean(iou[union_valid])

    # GT-supported mIoU:
    # only classes that actually exist in GT
    gt_valid = gt_pixels > 0

    miou_gt = np.mean(
        iou[gt_valid]
    )

    total_pixels = confusion.sum()

    if total_pixels > 0:
        pixel_accuracy = (
            tp.sum()
            / total_pixels
        )
    else:
        pixel_accuracy = np.nan

    return {
        "iou": iou,
        "miou_union": miou_union,
        "miou_gt": miou_gt,
        "pixel_accuracy": pixel_accuracy,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "union": union,
    }


# ============================================================
# Read summary.txt
# ============================================================

def read_summary_file(path):
    """
    Read number of images and recorded latency
    from the existing summary.txt.
    """

    data = {
        "images": None,
        "latency_ms": None
    }

    if not path.exists():
        return data

    text = path.read_text()

    image_match = re.search(
        r"Images evaluated:\s*(\d+)",
        text
    )

    latency_match = re.search(
        r"Average latency ms:\s*([0-9.]+)",
        text
    )

    if image_match:
        data["images"] = int(
            image_match.group(1)
        )

    if latency_match:
        data["latency_ms"] = float(
            latency_match.group(1)
        )

    return data


# ============================================================
# Load all conditions
# ============================================================

def load_all_results():

    results = {}

    for condition in CONDITIONS:

        condition_dir = (
            RESULT_ROOT
            / f"{condition}_val"
        )

        confusion_path = (
            condition_dir
            / "confusion_matrix.npy"
        )

        summary_path = (
            condition_dir
            / "summary.txt"
        )

        if not confusion_path.exists():
            raise FileNotFoundError(
                f"Missing confusion matrix: "
                f"{confusion_path}"
            )

        confusion = np.load(
            confusion_path
        )

        if confusion.shape != (
            NUM_CLASSES,
            NUM_CLASSES
        ):
            raise ValueError(
                f"{condition}: "
                f"unexpected confusion matrix "
                f"shape {confusion.shape}"
            )

        metrics = compute_metrics(
            confusion
        )

        summary = read_summary_file(
            summary_path
        )

        results[condition] = {
            "confusion": confusion,
            "metrics": metrics,
            "summary": summary
        }

    return results


# ============================================================
# Condition summary
# ============================================================

def print_condition_summary(results):

    print("\n")
    print("=" * 86)
    print("BASELINE CONDITION SUMMARY")
    print("=" * 86)

    print(
        f"{'Condition':<12}"
        f"{'Images':>10}"
        f"{'Pixel Acc':>14}"
        f"{'mIoU_union':>14}"
        f"{'mIoU_GT':>12}"
        f"{'Latency':>14}"
    )

    print("-" * 86)

    for condition in CONDITIONS:

        metrics = results[
            condition
        ]["metrics"]

        summary = results[
            condition
        ]["summary"]

        images = summary["images"]
        latency = summary["latency_ms"]

        image_string = (
            str(images)
            if images is not None
            else "N/A"
        )

        latency_string = (
            f"{latency:.2f} ms"
            if latency is not None
            else "N/A"
        )

        print(
            f"{condition:<12}"
            f"{image_string:>10}"
            f"{metrics['pixel_accuracy']:>14.4f}"
            f"{metrics['miou_union']:>14.4f}"
            f"{metrics['miou_gt']:>12.4f}"
            f"{latency_string:>14}"
        )


# ============================================================
# Per-class IoU table
# ============================================================

def print_class_iou_table(results):

    print("\n")
    print("=" * 86)
    print("PER-CLASS IoU ACROSS CONDITIONS")
    print("=" * 86)

    print(
        f"{'Class':<18}"
        f"{'Fog':>12}"
        f"{'Rain':>12}"
        f"{'Snow':>12}"
        f"{'Night':>12}"
    )

    print("-" * 66)

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        values = []

        for condition in CONDITIONS:

            value = results[
                condition
            ]["metrics"]["iou"][class_id]

            if np.isnan(value):
                values.append("N/A")
            else:
                values.append(
                    f"{value:.4f}"
                )

        print(
            f"{class_name:<18}"
            f"{values[0]:>12}"
            f"{values[1]:>12}"
            f"{values[2]:>12}"
            f"{values[3]:>12}"
        )


# ============================================================
# GT support table
# ============================================================

def print_support_table(results):

    print("\n")
    print("=" * 92)
    print("GT PIXEL SUPPORT ACROSS CONDITIONS")
    print("=" * 92)

    print(
        f"{'Class':<18}"
        f"{'Fog':>16}"
        f"{'Rain':>16}"
        f"{'Snow':>16}"
        f"{'Night':>16}"
    )

    print("-" * 82)

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        supports = []

        for condition in CONDITIONS:

            value = int(
                results[
                    condition
                ]["metrics"]["gt_pixels"][
                    class_id
                ]
            )

            supports.append(value)

        print(
            f"{class_name:<18}"
            f"{supports[0]:>16,d}"
            f"{supports[1]:>16,d}"
            f"{supports[2]:>16,d}"
            f"{supports[3]:>16,d}"
        )


# ============================================================
# Confusion analysis
# ============================================================

def analyze_class_confusion(
    results,
    condition,
    gt_class_name,
    top_k=6
):

    gt_class_id = CLASS_NAMES.index(
        gt_class_name
    )

    confusion = results[
        condition
    ]["confusion"]

    row = confusion[
        gt_class_id
    ].astype(np.float64)

    total = row.sum()

    print("\n")
    print("=" * 72)
    print(
        f"CONFUSION ANALYSIS: "
        f"{condition.upper()} / "
        f"GT = {gt_class_name}"
    )
    print("=" * 72)

    print(
        f"GT support: "
        f"{int(total):,} pixels"
    )

    if total == 0:

        print(
            "No GT pixels for this class."
        )

        return []

    percentages = (
        row / total
    )

    order = np.argsort(
        percentages
    )[::-1]

    records = []

    print(
        f"\n{'Predicted as':<20}"
        f"{'Pixels':>15}"
        f"{'% of GT class':>18}"
        f"{'Correct?':>12}"
    )

    print("-" * 65)

    for pred_id in order[:top_k]:

        pred_name = CLASS_NAMES[
            pred_id
        ]

        pixels = int(
            row[pred_id]
        )

        pct = (
            percentages[pred_id]
            * 100.0
        )

        correct = (
            "YES"
            if pred_id == gt_class_id
            else ""
        )

        print(
            f"{pred_name:<20}"
            f"{pixels:>15,d}"
            f"{pct:>17.2f}%"
            f"{correct:>12}"
        )

        records.append({
            "condition": condition,
            "gt_class": gt_class_name,
            "predicted_class": pred_name,
            "pixels": pixels,
            "percent_of_gt": pct,
            "correct": (
                pred_id == gt_class_id
            )
        })

    return records


# ============================================================
# Save CSV files
# ============================================================

def save_condition_summary(results):

    path = (
        OUTPUT_DIR
        / "condition_summary.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "condition",
            "images",
            "pixel_accuracy",
            "miou_union",
            "miou_gt",
            "latency_ms"
        ])

        for condition in CONDITIONS:

            metrics = results[
                condition
            ]["metrics"]

            summary = results[
                condition
            ]["summary"]

            writer.writerow([
                condition,
                summary["images"],
                metrics[
                    "pixel_accuracy"
                ],
                metrics[
                    "miou_union"
                ],
                metrics[
                    "miou_gt"
                ],
                summary[
                    "latency_ms"
                ]
            ])

    return path


def save_class_iou(results):

    path = (
        OUTPUT_DIR
        / "class_iou_by_condition.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "fog",
            "rain",
            "snow",
            "night"
        ])

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            row = [
                class_id,
                class_name
            ]

            for condition in CONDITIONS:

                row.append(
                    results[
                        condition
                    ]["metrics"]["iou"][
                        class_id
                    ]
                )

            writer.writerow(row)

    return path


def save_class_support(results):

    path = (
        OUTPUT_DIR
        / "class_support_by_condition.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "fog",
            "rain",
            "snow",
            "night"
        ])

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):

            row = [
                class_id,
                class_name
            ]

            for condition in CONDITIONS:

                row.append(
                    int(
                        results[
                            condition
                        ]["metrics"]["gt_pixels"][
                            class_id
                        ]
                    )
                )

            writer.writerow(row)

    return path


def save_confusion_analysis(records):

    path = (
        OUTPUT_DIR
        / "selected_confusions.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "condition",
            "gt_class",
            "predicted_class",
            "pixels",
            "percent_of_gt",
            "correct"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    return path


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = load_all_results()

    # --------------------------------------------------------
    # Main tables
    # --------------------------------------------------------

    print_condition_summary(
        results
    )

    print_class_iou_table(
        results
    )

    print_support_table(
        results
    )

    # --------------------------------------------------------
    # Selected failure cases
    # --------------------------------------------------------

    confusion_records = []

    confusion_records += (
        analyze_class_confusion(
            results,
            condition="snow",
            gt_class_name="terrain"
        )
    )

    confusion_records += (
        analyze_class_confusion(
            results,
            condition="snow",
            gt_class_name="bus"
        )
    )

    confusion_records += (
        analyze_class_confusion(
            results,
            condition="night",
            gt_class_name="sky"
        )
    )

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    summary_path = (
        save_condition_summary(
            results
        )
    )

    iou_path = (
        save_class_iou(
            results
        )
    )

    support_path = (
        save_class_support(
            results
        )
    )

    confusion_path = (
        save_confusion_analysis(
            confusion_records
        )
    )

    print("\n")
    print("=" * 72)
    print("FILES SAVED")
    print("=" * 72)

    print(summary_path)
    print(iou_path)
    print(support_path)
    print(confusion_path)


if __name__ == "__main__":
    main()