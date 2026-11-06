"""Post-hoc best/worst image examples from completed held-out predictions."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .common import ROOT, save_json, sha256


def select_cases(images):
    """Select the minimum/maximum LR+class minus MSP retrospective80 risk."""
    from .report import checkpoint80

    ranked, seen = [], set()
    for image in images:
        if image["image_id"] in seen:
            raise ValueError("Duplicate image in case-selection population")
        seen.add(image["image_id"])
        raw, learned = (image["methods"][m] for m in ("raw_msp", "lr3_class"))
        if (raw["num_pixels"] <= 0 or raw["num_pixels"] != learned["num_pixels"]
                or raw["num_errors"] != learned["num_errors"]):
            raise ValueError("Case methods must use identical nonempty evaluation populations")
        a, b = checkpoint80(learned), checkpoint80(raw)
        delta = float(a["risk"] - b["risk"])
        if not np.isfinite(delta):
            raise ValueError("Case-selection risk difference must be finite")
        ranked.append((delta, image["image_id"], image))
    if not ranked:
        raise ValueError("Case selection needs completed image results")
    ranked.sort(key=lambda item: (item[0], item[1]))
    extremes = [("minimum_delta", ranked[0])]
    if ranked[-1][1] != ranked[0][1]:
        extremes.append(("maximum_delta", ranked[-1]))
    return [dict(role=role, risk_delta=value[0], image=value[2]) for role, value in extremes]


def _render_case(case, row, selection, device, output):
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from PIL import Image

    from .evaluation import predict_scores
    from .features import IMAGE_SHAPE, iter_feature_blocks, load_target, native_cache
    from .report import checkpoint80

    started = time.perf_counter()
    group = row["recording_group"]
    model_path = ROOT / "data/models" / group / "lr3_class.joblib"
    model_digest = sha256(model_path)
    if model_digest != selection["model_hashes"]["lr3_class"]:
        raise RuntimeError("Selected fold model changed")
    model = joblib.load(model_path)
    feature_record = json.loads((ROOT / "data/features" / (row["image_id"] + ".json")).read_text())
    if feature_record["device"] != device or feature_record["feature_source_sha256"] != sha256(ROOT / "features.py"):
        raise RuntimeError("Visualizations require the original feature source and numerical backend")
    cache_path = Path(row["native_cache_path"]) if row.get("native_cache_path") else ROOT / "data/native_logits" / (row["image_id"] + ".npy")
    if not cache_path.is_file():
        raise RuntimeError("Visualization requires an existing cache; backbone inference is disabled")
    native, cache, _ = native_cache(row, device)
    target = load_target(row)
    prediction = np.empty(target.size, dtype=np.uint8)
    scores = {name: np.empty(target.size, dtype=np.float64) for name in ("raw_msp", "lr3_class")}
    for first, last, X, _ in iter_feature_blocks(native, device):
        prediction[first:last] = X[:, 3].astype(np.uint8)
        scores["raw_msp"][first:last] = predict_scores("raw_msp", X)
        scores["lr3_class"][first:last] = predict_scores("lr3_class", X, model)
    prediction = prediction.reshape(IMAGE_SHAPE)
    scores = {name: values.reshape(IMAGE_SHAPE) for name, values in scores.items()}
    valid = target < 19
    errors = (prediction != target) & valid
    outcomes, measured = {}, {}
    for name, values in scores.items():
        threshold = float(selection["thresholds"][name]["value"])
        accepted = values <= threshold
        counts = dict(num_evaluated=int(valid.sum()), accepted_pixels=int(np.count_nonzero(valid & accepted)),
                      accepted_errors=int(np.count_nonzero(errors & accepted)))
        expected = case["image"]["methods"][name]
        if counts["num_evaluated"] != expected["num_pixels"] or int(errors.sum()) != expected["num_errors"]:
            raise RuntimeError("Regenerated case differs from the evaluated valid/error population")
        if any(counts[key] != expected["fixed_threshold"][key] for key in counts):
            raise RuntimeError("Regenerated frozen-threshold counts differ from evaluation")
        if threshold != expected["fixed_threshold"]["threshold"]:
            raise RuntimeError("Case threshold differs from the fold training threshold")
        measured[name] = dict(**counts, threshold=threshold,
            coverage=counts["accepted_pixels"] / counts["num_evaluated"],
            risk=counts["accepted_errors"] / counts["accepted_pixels"] if counts["accepted_pixels"] else None)
        outcomes[name] = np.where(valid, errors.astype(np.uint8) + 2 * (~accepted), 4)
    if sha256(row["rgb_path"]) != row["rgb_sha256"]:
        raise RuntimeError("Case RGB bytes changed")
    with Image.open(row["rgb_path"]) as source:
        rgb = np.array(source.convert("RGB"))
    if rgb.shape[:2] != IMAGE_SHAPE:
        raise ValueError("Case RGB resolution mismatch")

    palette = ["#263b33", "#d1495b", "#607d8b", "#e6af2e", "#e5e5e5"]
    cmap = ListedColormap(palette)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    fig.subplots_adjust(left=.015, right=.985, top=.82, bottom=.13, wspace=.025, hspace=.27)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB input", fontsize=11)
    axes[0, 1].imshow(np.where(valid, errors.astype(np.uint8), 4), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[0, 1].set_title(f"GT error map: red = wrong, green = correct\nError rate {100 * errors.sum() / valid.sum():.2f}%", fontsize=10)
    for name, ax, title in (("raw_msp", axes[0, 2], "Raw MSP error score (1 − MSP)"),
                            ("lr3_class", axes[1, 0], "Learned error score (LR + predicted class)")):
        score_image = ax.imshow(np.ma.masked_where(~valid, scores[name]), cmap="viridis", vmin=0, vmax=1,
                               interpolation="nearest")
        ax.set_facecolor(palette[4])
        ax.set_title(title, fontsize=11)
    for name, ax, label in (("raw_msp", axes[1, 1], "MSP"), ("lr3_class", axes[1, 2], "LR + class")):
        metric = measured[name]
        risk_label = "undefined" if metric["risk"] is None else f"{100 * metric['risk']:.2f}%"
        ax.imshow(outcomes[name], cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(f"{label} · training cutoff τ={metric['threshold']:.5f}\n"
                     f"Coverage {100 * metric['coverage']:.2f}% · retained error {risk_label}", fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    raw80, learned80 = (checkpoint80(case["image"]["methods"][m]) for m in ("raw_msp", "lr3_class"))
    label = "Minimum" if case["role"] == "minimum_delta" else "Maximum"
    fig.suptitle(f"{label} retrospective risk difference (post-hoc example) · {row['image_id']}\n"
                 f"LR+class − MSP: {100 * case['risk_delta']:+.2f} percentage points; "
                 f"retrospective coverage MSP {100 * raw80['actual_coverage']:.2f}%, "
                 f"LR+class {100 * learned80['actual_coverage']:.2f}%\n"
                 "Lower-right panels use fold training cutoffs; retrospective ranking is used for case selection only.",
                 fontsize=11, y=.975)
    labels = ["Accepted correct", "Accepted error", "Rejected correct", "Rejected error", "Ignored GT"]
    fig.legend([Patch(color=color) for color in palette], labels, loc="lower center",
               bbox_to_anchor=(.40, .015), ncol=3, frameon=False, fontsize=9,
               title="Cutoff-panel outcomes", title_fontsize=9)
    colorbar = fig.colorbar(score_image, cax=fig.add_axes([.77, .065, .20, .018]), orientation="horizontal")
    colorbar.set_ticks([0, .5, 1])
    colorbar.set_label("Error score: low → high", fontsize=9)
    path = output / (case["role"] + ".png")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return dict(role=case["role"], image_id=row["image_id"], recording_group=group, condition=row["condition"],
        retrospective80_risk_delta=case["risk_delta"], retrospective80=dict(raw_msp=raw80, lr3_class=learned80),
        rendered_frozen_threshold_metrics=measured, png=str(path), png_sha256=sha256(path),
        native_cache=cache, model=str(model_path), model_sha256=model_digest,
        rgb_sha256=row["rgb_sha256"], gt_sha256=row["gt_sha256"], device=device,
        elapsed_seconds=time.perf_counter() - started)


def generate(device="cuda"):
    from .features import configure_torch
    from .protocol import read_csv
    from .report import collect

    started = time.perf_counter()
    complete = json.loads((ROOT / "results/evaluation_complete.json").read_text())
    if complete.get("status") != "PASS" or complete.get("num_groups") != 17 or complete.get("num_images") != 1600:
        raise RuntimeError("Case selection requires all 17 completed groups and 1600 images")
    groups, selections, _, sources = collect()
    metadata = {row["image_id"]: row for row in read_csv(ROOT / "data/metadata_train.csv")}
    images = []
    for group, result in groups.items():
        path = Path(result["image_records"])
        if not path.resolve().is_relative_to(ROOT):
            raise RuntimeError("Image results must be inside this extension")
        members = json.loads(path.read_text())
        for image in members:
            if image["recording_group"] != group or metadata[image["image_id"]]["recording_group"] != group:
                raise RuntimeError("Image result belongs to a different fold")
        images.extend(members)
        sources[str(path.relative_to(ROOT))] = sha256(path)
    if len(images) != 1600 or {image["image_id"] for image in images} != set(metadata):
        raise RuntimeError("Case selection requires every image exactly once")
    cases = select_cases(images)
    output = ROOT / "results/report/cases"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime/matplotlib"))
    configure_torch(device)
    records = [_render_case(case, metadata[case["image"]["image_id"]],
                           selections[case["image"]["recording_group"]], device, output) for case in cases]
    result = dict(selection="Post hoc minimum and maximum per-image LR+class minus raw MSP retrospective80 risk; image ID breaks ties",
        population_images=len(images), cases=records, source_hashes=sources,
        visualization_source_sha256=sha256(__file__), elapsed_seconds=time.perf_counter() - started,
        limitations=["Selected extreme examples are descriptive, not representative samples or independent evidence",
                     "Panel acceptance uses each outer fold's training-only cutoff; case ranking uses retrospective near-80% coverage",
                     "GT derives the error display and metric checks only; classifier inputs use RGB-derived frozen logits",
                     "Native caches are reused; no segmentation network forward pass or model fitting"])
    save_json(output / "case_selection.json", result)
    return result


def self_check():
    def example(name, delta):
        def metrics(risk):
            return dict(num_pixels=10, num_errors=4, checkpoints=[dict(requested_coverage=.8, risk=risk, actual_coverage=.8)])
        return dict(image_id=name, methods=dict(raw_msp=metrics(.2), lr3_class=metrics(.2 + delta)))
    images = [example("middle", 0), example("best", -.1), example("worst", .2)]
    cases = select_cases(images)
    assert [case["image"]["image_id"] for case in cases] == ["best", "worst"]
    assert len(select_cases([example("single", 0)])) == 1
    tied = select_cases([example("b", 0), example("a", 0)])
    assert [case["image"]["image_id"] for case in tied] == ["a", "b"]
    try:
        select_cases([images[0], images[0]])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate images were accepted")
    print("Case-selection synthetic self-check PASS; no images, models or GPU loaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        result = generate(args.device)
        print(json.dumps(dict(panels=[case["png"] for case in result["cases"]],
                              elapsed_seconds=result["elapsed_seconds"]), indent=2))
