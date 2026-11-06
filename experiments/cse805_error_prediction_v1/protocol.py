"""Audit ACDC inputs and freeze label-free grouped assignments for CSE 805."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
from PIL import Image

from stage1.dataset import ACDC_ROOT, RGB_ROOT, GT_ROOT, acdc_path, dataset_reference
from experiments.paired_calibration_v1.common import (
    PROJECT_ROOT, MODEL_NAME, MODEL_REVISION, sha256, read_csv, write_csv,
    read_json, write_json,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CONDITIONS = ("fog", "night", "rain", "snow")
FIT = PROJECT_ROOT / "results/stage3_calibration/stage3b"
SELECT = PROJECT_ROOT / "results/stage4_reliability/stage4b"
CORRECTED = PROJECT_ROOT / "results/research_extension/paired_calibration_v1/corrected_validation/cache_index_val.json"
SOURCE_MANIFESTS = (
    FIT / "cache_index.csv", FIT / "cache_metadata.json", FIT / "split_manifest.json",
    FIT / "splits/calibration_fit.csv", FIT / "splits/reliability_selection.csv",
    SELECT / "cache_index.csv", CORRECTED,
)


def recording_group(sequence_id):
    """Merge possible GoPro chapters; this is not a proven independent drive ID."""
    match = re.fullmatch(r"(?:GOPR|GP\d{2})(\d{4})", sequence_id)
    if match is None:
        raise ValueError(f"Unrecognized sequence prefix requires review: {sequence_id}")
    return "gopro_" + match[1]


def parse_image(path):
    path = Path(path)
    match = re.fullmatch(r"((?:GOPR|GP\d{2})\d{4})_frame_(\d{6})_rgb_anon\.png", path.name)
    if match is None or match[1] != path.parent.name:
        raise ValueError(f"Filename/parent sequence mismatch or unknown syntax: {path}")
    return match[1], recording_group(match[1])


def group_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["recording_group"]].append(row)
    result = []
    for group, members in sorted(groups.items()):
        conditions = Counter(r["condition"] for r in members)
        if len(conditions) != 1:
            raise ValueError(f"Numeric recording suffix collision across weather requires review: {group}")
        result.append({"recording_group": group, "num_images": len(members),
                       "conditions": ";".join(sorted(conditions)),
                       "num_conditions": len(conditions),
                       "sequence_ids": ";".join(sorted({r['sequence_id'] for r in members})),
                       **{f"{c}_images": conditions[c] for c in CONDITIONS}})
    return result


def build_assignments(rows):
    """Outer LOGO; inner three folds greedily balance weather/total image counts."""
    if any(r["dataset_split"] != "train" for r in rows):
        raise ValueError("Supplemental validation must not enter primary assignments")
    summaries = group_summary(rows)
    counts = {r["recording_group"]: r["num_images"] for r in summaries}
    weather = {r["recording_group"]: r["conditions"] for r in summaries}
    groups = sorted(counts)
    if len(groups) < 5:
        raise ValueError("Need at least five groups for this nested protocol")
    assignments = []
    for outer in groups:
        training = [g for g in groups if g != outer]
        folds, totals = [[], [], []], [Counter(), Counter(), Counter()]
        for group in sorted(training, key=lambda g: (-counts[g], g)):
            c = weather[group]
            destination = min(range(3), key=lambda f: (totals[f][c], sum(totals[f].values()), f))
            folds[destination].append(group)
            totals[destination][c] += counts[group]
        assignments.append({"outer_fold": outer, "eval_group": outer,
                            "eval_image_count": counts[outer], "train_groups": training,
                            "train_image_count": sum(counts[g] for g in training),
                            "inner_folds": [{"inner_fold": i,
                                "fit_groups": sorted(set(training) - set(dev)),
                                "dev_groups": sorted(dev),
                                "fit_image_count": sum(counts[g] for g in training if g not in dev),
                                "dev_image_count": sum(counts[g] for g in dev),
                                "dev_weather_image_counts": dict(sorted(totals[i].items()))}
                                for i, dev in enumerate(folds)]})
    validate_assignments(rows, assignments)
    return assignments


def validate_assignments(rows, assignments):
    groups = {r["recording_group"] for r in rows}
    assert all(r["dataset_split"] == "train" for r in rows)
    assert Counter(a["eval_group"] for a in assignments) == Counter({g: 1 for g in groups})
    for outer in assignments:
        evaluation = outer["eval_group"]
        training = set(outer["train_groups"])
        assert evaluation not in training and training | {evaluation} == groups
        assert len(outer["train_groups"]) == len(training)
        assert len(outer["inner_folds"]) == 3
        seen = Counter()
        for fold in outer["inner_folds"]:
            fit, dev = set(fold["fit_groups"]), set(fold["dev_groups"])
            assert fit and dev and not fit & dev
            assert fit | dev == training and evaluation not in fit | dev
            assert len(fit) == len(fold["fit_groups"]) and len(dev) == len(fold["dev_groups"])
            seen.update(dev)
        assert seen == Counter({g: 1 for g in training})


def _index(rows):
    result = {r["rgb_path"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate RGB in source inventory")
    return result


def _file_check(path, expected=""):
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if expected and digest != expected:
        raise ValueError(f"Frozen input SHA256 mismatch: {path}")
    return digest


def _metadata_row(job):
    path, split, old, native, prediction = job
    sequence, group = parse_image(path)
    condition = path.parents[2].name
    gt = GT_ROOT / condition / split / sequence / path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
    if old and (old["sequence_id"] != sequence or old["condition"] != condition or
                acdc_path(old["gt_path"]) != gt):
        raise ValueError(f"Source inventory/path disagreement: {path}")
    rgb_hash = _file_check(path, native.get("rgb_sha256", ""))
    gt_hash = _file_check(gt, native.get("gt_sha256", ""))
    with Image.open(path) as image, Image.open(gt) as label:
        if image.size != (1920, 1080) or label.size != image.size:
            raise ValueError(f"Image/label dimension mismatch: {path}")
        width, height = image.size
    cache = Path(native["cache_path"]) if native else None
    if cache and not cache.is_absolute():
        cache = PROJECT_ROOT / cache
    cache_hash = _file_check(cache, native["cache_sha256"]) if cache else ""
    if cache:
        array = np.load(cache, mmap_mode="r", allow_pickle=False)
        if array.shape != (1, 19, 128, 256) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid native logits: {cache}")
    pred = PROJECT_ROOT / prediction["prediction_path"] if prediction else None
    pred_hash = _file_check(pred, prediction["prediction_sha256"]) if pred else ""
    if pred:
        array = np.load(pred, mmap_mode="r", allow_pickle=False)
        if array.shape != (height, width):
            raise ValueError(f"Prediction-cache dimensions disagree: {pred}")
    uncertainty = PROJECT_ROOT / prediction["uncertainty_path"] if prediction else None
    if uncertainty and not uncertainty.is_file():
        raise FileNotFoundError(uncertainty)
    return {"image_id": f"{condition}/{path.stem}", "condition": condition,
            "dataset_split": split, "sequence_id": sequence, "recording_group": group,
            "rgb_path": str(path), "gt_path": str(gt),
            "rgb_reference": dataset_reference(path), "gt_reference": dataset_reference(gt),
            "rgb_sha256": rgb_hash, "gt_sha256": gt_hash,
            "height": height, "width": width, "legacy_subset": old.get("subset", "supplemental_val"),
            "native_cache_available": bool(cache), "native_cache_path": str(cache) if cache else "",
            "native_cache_sha256": cache_hash,
            "native_cache_origin": ("stage3b" if split == "train" else "paired_corrected_validation") if cache else "",
            "known_valid_pixels": native.get("num_valid_pixels", ""),
            "prediction_cache_available": bool(pred), "prediction_cache_path": str(pred) if pred else "",
            "prediction_cache_sha256": pred_hash,
            "calibrated_msp_cache_available": bool(uncertainty),
            "calibrated_msp_cache_path": str(uncertainty) if uncertainty else "",
            "calibrated_msp_cache_sha256_expected": prediction.get("uncertainty_sha256", "")}


def audit_metadata(train, val):
    assert len(train) == 1600 and len(val) == 406
    assert Counter(r["condition"] for r in train) == Counter(dict.fromkeys(CONDITIONS, 400))
    assert Counter(r["condition"] for r in val) == Counter(fog=100, night=106, rain=100, snow=100)
    for rows in (train, val):
        assert len({r["image_id"] for r in rows}) == len(rows)
        assert len({r["rgb_path"] for r in rows}) == len(rows)
        assert len({r["gt_path"] for r in rows}) == len(rows)
        group_summary(rows)
    assert len(group_summary(train)) == 17
    assert not {r["rgb_path"] for r in train} & {r["rgb_path"] for r in val}
    assert not {r["image_id"] for r in train} & {r["image_id"] for r in val}
    by_hash = defaultdict(list)
    for row in train:
        by_hash[row["rgb_sha256"]].append(row)
    duplicates = []
    for digest, members in by_hash.items():
        if len({r["recording_group"] for r in members}) > 1:
            raise ValueError("Identical training RGB across proposed recording groups")
        if len(members) > 1:
            duplicates.append({"rgb_sha256": digest, "image_ids": [r["image_id"] for r in members]})
    fit = [r for r in train if r["legacy_subset"] == "calibration_fit"]
    selection = [r for r in train if r["legacy_subset"] == "reliability_selection"]
    assert (len(fit), len(selection)) == (802, 798)
    overlaps = {}
    for name, left, right in (("train_vs_val", train, val), ("legacy_fit_vs_selection", fit, selection)):
        overlaps[name] = {key: sorted({r[key] for r in left} & {r[key] for r in right})
                          for key in ("sequence_id", "recording_group", "rgb_sha256")}
    return {"status": "PASS", "train_images": len(train), "supplemental_val_images": len(val),
            "recording_proxy_groups": 17,
            "train_native_logits_present": sum(r["native_cache_available"] for r in train),
            "train_native_logits_missing": sum(not r["native_cache_available"] for r in train),
            "supplemental_val_native_logits_present": sum(r["native_cache_available"] for r in val),
            "within_group_exact_rgb_duplicates": duplicates, "overlaps": overlaps,
            "source_rgb_gt_hashes_verified": True, "native_cache_hashes_shapes_finiteness_verified": True,
            "prediction_cache_hashes_shapes_verified": True,
            "calibrated_msp_cache_note": "Existence and historical expected hash recorded; not reused as raw features",
            "semantic_gt_values_note": "Full label-value checks occur during extraction; this audit checks headers and hashes",
            "group_independence_proven": False,
            "ambiguities": [
                "ACDC README defines filename prefixes as image sequences, not independent recording sessions.",
                "Merging GOPRnnnn and GPxxnnnn is a conservative chapter-family proxy inferred from filenames.",
                "Same numeric suffix may recur on another camera/date; no camera/time/drive metadata resolves this locally.",
                "Different numeric suffixes can still belong to one drive, route, location, or correlated weather period.",
                "All groups currently contain one weather; weather and recording identity are partly confounded.",
                "Rain and night each have only three primary groups; inner dev folds cannot always contain every weather.",
                "Historical train/validation and calibration/selection are not disjoint under the conservative proxy.",
                "Previously explored training/validation data do not constitute untouched confirmatory test data."]}


def build_metadata():
    for path, digest in read_json(FIT / "split_manifest.json").items():
        _file_check(PROJECT_ROOT / path, digest)
    fit = _index(read_csv(FIT / "splits/calibration_fit.csv"))
    selection = _index(read_csv(FIT / "splits/reliability_selection.csv"))
    native = _index(read_csv(FIT / "cache_index.csv"))
    prediction = _index(read_csv(SELECT / "cache_index.csv"))
    corrected = read_json(CORRECTED)
    old_meta = read_json(FIT / "cache_metadata.json")
    for record in (old_meta, corrected):
        if record["model"] != MODEL_NAME or record["model_revision"] != MODEL_REVISION:
            raise ValueError("Native caches do not match frozen model/revision")
    if old_meta["cache_index_sha256"] != sha256(FIT / "cache_index.csv"):
        raise ValueError("Frozen native cache index SHA256 mismatch")
    assert not fit.keys() & selection.keys()
    assert native.keys() == fit.keys() and prediction.keys() == selection.keys()
    corrected_native = _index([r for r in corrected["rows"] if r["domain"] == "night"])
    jobs = []
    for split in ("train", "val"):
        for condition in CONDITIONS:
            paths = sorted((RGB_ROOT / condition / split).glob("*/*_rgb_anon.png"))
            expected = 400 if split == "train" else 106 if condition == "night" else 100
            if len(paths) != expected:
                raise ValueError(f"Unexpected {condition}/{split} population: {len(paths)}")
            for path in paths:
                ref = dataset_reference(path)
                old = (fit.get(ref) or selection.get(ref)) if split == "train" else {}
                if split == "train" and old is None:
                    raise ValueError(f"Filesystem training image absent from old inventories: {path}")
                jobs.append((path, split, old, native.get(ref, corrected_native.get(ref, {})), prediction.get(ref, {})))
    if {dataset_reference(job[0]) for job in jobs if job[1] == "train"} != fit.keys() | selection.keys():
        raise ValueError("Filesystem/inventory populations differ")
    rows = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, row in enumerate(executor.map(_metadata_row, jobs), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(jobs):
                print(f"Audited image/header/cache hashes: {i}/{len(jobs)}", flush=True)
    train, val = ([r for r in rows if r["dataset_split"] == split] for split in ("train", "val"))
    return train, val, audit_metadata(train, val)


def prepare(data_dir=DATA):
    """Read-only audit of legacy/data inputs; write only new immutable metadata."""
    data_dir = Path(data_dir)
    if (data_dir / "leakage_audit.json").exists():
        raise FileExistsError("Phase 1 artifacts already exist; validate them instead of replacing")
    train, val, audit = build_metadata()
    assignments = build_assignments(train)
    write_csv(data_dir / "metadata_train.csv", train)
    write_csv(data_dir / "metadata_supplemental_val.csv", val)
    write_csv(data_dir / "groups_train.csv", group_summary(train))
    write_csv(data_dir / "groups_supplemental_val.csv", group_summary(val))
    write_json(data_dir / "assignments.json", assignments)
    write_json(data_dir / "leakage_audit.json", audit)
    write_json(data_dir / "phase1_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {p.name: sha256(p) for p in sorted(data_dir.iterdir()) if p.is_file()},
        "source_manifests": {str(p): sha256(p) for p in SOURCE_MANIFESTS},
        "source_code": {str(Path(__file__).resolve()): sha256(__file__)},
        "readme_evidence": {str(p): sha256(p) for p in
             (ACDC_ROOT / "rgb_anon_trainvaltest/README.md", ACDC_ROOT / "gt_trainval/README.md")}})
    return audit


def verify_phase1(data_dir=DATA):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "phase1_manifest.json")
    for name, digest in manifest["artifacts"].items():
        _file_check(data_dir / name, digest)
    for name, digest in manifest["source_manifests"].items():
        _file_check(Path(name), digest)
    rows = read_csv(data_dir / "metadata_train.csv")
    validate_assignments(rows, read_json(data_dir / "assignments.json"))
    return manifest


def freeze(data_dir=DATA):
    """Freeze planned choices before any extension model training/evaluation."""
    data_dir = Path(data_dir)
    verify_phase1(data_dir)
    if (data_dir / "protocol_freeze.json").exists():
        return load_protocol(data_dir)
    if any((ROOT / name).exists() for name in ("evaluation_complete.json", "full_evaluation")):
        raise RuntimeError("Cannot freeze after extension evaluation has begun")
    protocol = {
        "version": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "CSE805 supervised segmentation error prediction v1",
        "output_directory": str(ROOT), "dataset_root": str(ACDC_ROOT),
        "model": MODEL_NAME, "model_revision": MODEL_REVISION,
        "backbone_frozen": True, "baseline_temperature": 1.0,
        "primary_population": "All 1600 official adverse train images; 400 per weather",
        "supplemental_validation": "406 official val images inventoried separately; excluded from all primary fits, tuning, and evaluation",
        "group_rule": "fullmatch (?:GOPR|GP[0-9]{2})([0-9]{4}); gopro_ + four-digit suffix; exact prefix must equal parent directory",
        "group_independence_claimed": False,
        "group_ambiguity": "Filename chapter-family proxy, not verified independent camera/date/session/drive/route",
        "outer": {"method": "leave one conservative recording proxy group out", "folds": 17,
                  "assignments": "assignments.json", "every_image_evaluated_once": True},
        "inner": {"folds": 3, "assignment": "Within outer train only: groups sorted by decreasing image count then group ID; greedily assign to fold with fewest same-weather images, then total images, then fold index",
                  "labels_used_for_assignment": False,
                  "missing_weather": "Some inner dev folds omit a weather because rain/night each have only three groups before outer holdout",
                  "C_grid": [0.01, 0.1, 1.0], "regularization": "L2",
                  "selection": "Arithmetic mean of three inner-dev AP values, each computed with N_valid_image/sample_count weights; ties within 1e-12 select smaller C",
                  "outer_labels_used_for_selection": False,
                  "both_error_classes_required": "Require both labels in each training/development/evaluation population; fail and disclose unsupported metrics instead of fabricating scores"},
        "sampling": {"seed": 805, "train_samples_per_image": 512,
                     "optional_exact_metric_qa_samples_per_image": 8192,
                     "method": "Uniform semantic-valid pixels without replacement; deterministic image-ID SHA256 and seed; no error/class stratification",
                     "weight": "N_valid_image / sampled_count; normalize weights to mean 1 for fitting only",
                     "evaluation": "Primary metrics use every valid full-resolution pixel; QA sample is not the primary estimate"},
        "features": {"continuous": ["msp", "normalized_entropy", "margin"],
                     "categorical": "predicted semantic class; 19 Cityscapes train IDs, 0 through 18",
                     "native_logits": {"shape": [1, 19, 128, 256], "dtype": "float32"},
                     "input_size": [512, 1024], "evaluation_size": [1080, 1920],
                     "resize": "float32 bilinear native logits to label/image resolution, align_corners=False, before raw softmax",
                     "probability": "raw float32 softmax(z), T=1; finite, positive-normalized probabilities verified",
                     "msp": "max_c p_c",
                     "normalized_entropy": "-sum_c p_c*log(max(p_c,1e-12))/log(19)",
                     "margin": "largest probability minus second largest probability",
                     "prediction": "argmax of unscaled float32 interpolated logits; lowest class index for exact ties",
                     "label": "1[predicted_trainId != semantic_GT_trainId]",
                     "validity": "GT in 0..18; exclude 255; other GT IDs are a hard data error",
                     "invIds": "Perceptual ambiguity mask, not primary semantic-validity mask and never a predictor feature",
                     "ground_truth_as_feature": False,
                     "alignment": "Own image and own semantic GT only, no normal-reference pixel correspondence"},
        "methods": ["raw_msp", "raw_entropy", "lr3", "lr3_class"],
        "method_definition": {"raw_msp": "1-msp; greater means more likely error",
                              "raw_entropy": "normalized predictive entropy; greater means more likely error",
                              "lr3": "StandardScaler plus LogisticRegression on three continuous features",
                              "lr3_class": "StandardScaler continuous features plus predicted-class one-hot in sklearn preprocessing pipeline, then LogisticRegression",
                              "learned_score": "predict_proba for segmentation error label 1",
                              "preprocessing": "Fit scaler/encoder inside each inner-training or final outer-training sklearn pipeline only",
                              "class_balance": "No class-balanced sampling, no class weights; preserve pixel prevalence through inclusion weights"},
        "threshold": {"target_training_coverage": 0.8,
                      "selection": "For each method and outer fold, weighted 80% quantile of final outer-training sample scores; accept score <= threshold",
                      "record": "threshold, achieved weighted training-sample coverage, exact outer accepted counts/errors/coverage",
                      "outer_labels_used": False,
                      "limitation": "Training-score quantile is optimistic in-sample and does not guarantee 80% unseen-group coverage; ties may overshoot"},
        "metrics": {"primary_pixels": "Identical complete semantic-valid pixels for all four methods",
                    "streaming": "10,000 fixed equal-width score histogram bins over [0,1] per method; exact population/error and threshold counts; scores regenerated from native logits in image chunks",
                    "bin_assignment": "min(floor(clipped_score*10000),9999); zero in first bin and one in last bin",
                    "auroc": "Histogram approximation, tied-bin positive/negative pairs receive half credit",
                    "ap": "Histogram grouped-threshold average precision approximation",
                    "aurc": "Histogram grouped-bin risk-coverage approximation; same rule for every method",
                    "retrospective80": "Nearest attainable whole-bin cumulative coverage to 0.8, ties choose lower coverage; report actual coverage and error; diagnostic only, never deployment threshold",
                    "prevalence": "Exact error pixels / exact valid pixels",
                    "training_threshold": "Exact held-out accepted and error counts using training-only threshold",
                    "outputs": ["AUROC", "AP", "error_prevalence", "risk_coverage_curve", "AURC", "retrospective80_actual_coverage", "retrospective80_accepted_error", "training_threshold_actual_coverage", "training_threshold_accepted_error"],
                    "qa": "Compare histogram metrics against exact sklearn/sorted metrics on synthetic cases and retained shared per-image QA sample; disclose discrepancies"},
        "interpretation": {"primary_endpoint": "Mean paired outer-family difference in retrospective80 accepted error: each learned method minus raw MSP; negative is better; also compare raw entropy",
                           "class_ablation": "lr3_class minus lr3 for same held-out family and pixels",
                           "reporting": "Every family, each weather, pooled out-of-fold histograms, unsuccessful methods, and selection/runtime costs",
                           "pooled_caution": "Pooled cross-fitted scores combine different fold-specific models; diagnostic, not performance of one deployed model",
                           "group_uncertainty": "Descriptive paired family deltas, wins/ties/losses, range and leave-one-family-out sensitivity of mean; 17 conservative proxies, not proven independent sessions",
                           "significance": "No pixel-IID significance tests or confidence intervals; no unsupported population-wide significance claim",
                           "claim_limits": "No new segmentation backbone, no changed original segmentation mIoU, no superiority before measured evidence",
                           "historical_exploration": "Existing data were used in earlier COMP895 experiments; not pristine confirmatory test data"},
        "preservation": {"legacy_files": "Read-only; no writes to original data, stage1-stage5, or results",
                         "new_artifacts": "All new code, metadata, features, models, logs, and reports inside this extension directory"},
        "phase1_manifest_sha256": sha256(data_dir / "phase1_manifest.json"),
        "protocol_source_sha256": sha256(__file__),
    }
    write_json(data_dir / "protocol.json", protocol)
    write_json(data_dir / "protocol_freeze.json", {
        "protocol_sha256": sha256(data_dir / "protocol.json"), "frozen_utc": protocol["frozen_utc"],
        "extension_training_started": False, "primary_evaluation_started": False})
    return protocol


def load_protocol(data_dir=DATA):
    data_dir = Path(data_dir)
    record = read_json(data_dir / "protocol_freeze.json")
    _file_check(data_dir / "protocol.json", record["protocol_sha256"])
    protocol = read_json(data_dir / "protocol.json")
    _file_check(data_dir / "phase1_manifest.json", protocol["phase1_manifest_sha256"])
    verify_phase1(data_dir)
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "freeze"))
    arguments = parser.parse_args()
    print({"prepare": prepare, "verify": verify_phase1, "freeze": freeze}[arguments.action]())
