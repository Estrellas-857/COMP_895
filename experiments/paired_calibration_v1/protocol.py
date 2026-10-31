"""Freeze exact pair populations and analysis rules before validation evaluation."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

from stage1.dataset import RGB_ROOT, GT_ROOT, REF_GT_ROOT, acdc_path, dataset_reference
from .common import (PROJECT_ROOT, OUTPUT_ROOT, SOURCE_ROOT, GLOBAL_T, GLOBAL_TAU,
                     MODEL_NAME, MODEL_REVISION, sha256, read_csv, write_csv,
                     read_json, write_json, load_protocol)

FIT_ROOT = PROJECT_ROOT / "results/stage3_calibration/stage3b"


def build_manifests():
    fit = read_csv(FIT_ROOT / "splits/calibration_fit.csv")
    selection = read_csv(FIT_ROOT / "splits/reliability_selection.csv")
    original_cache = {r["rgb_path"]: r for r in read_csv(FIT_ROOT / "cache_index.csv")}
    fit_rgb = {r["rgb_path"] for r in fit}
    selection_rgb = {r["rgb_path"] for r in selection}
    if fit_rgb & selection_rgb:
        raise RuntimeError("Original fit/selection image overlap")
    pair_rows = {"fit": [], "val": []}
    image_rows = {"fit": [], "val": []}
    excluded = []
    for split, dataset_split in (("fit", "train"), ("val", "val")):
        for normal_gt in sorted((REF_GT_ROOT / "night" / f"{dataset_split}_ref").glob("*/*_gt_ref_labelTrainIds.png")):
            key = normal_gt.name.removesuffix("_gt_ref_labelTrainIds.png")
            sequence = normal_gt.parent.name
            night_rgb = RGB_ROOT / "night" / dataset_split / sequence / f"{key}_rgb_anon.png"
            normal_rgb = RGB_ROOT / "night" / f"{dataset_split}_ref" / sequence / f"{key}_rgb_ref_anon.png"
            night_gt = GT_ROOT / "night" / dataset_split / sequence / f"{key}_gt_labelTrainIds.png"
            invalid = GT_ROOT / "night" / dataset_split / sequence / f"{key}_gt_invIds.png"
            night_id = dataset_reference(night_rgb)
            if split == "fit" and night_id not in fit_rgb:
                if night_id not in selection_rgb:
                    raise RuntimeError(f"Unassigned training pair: {key}")
                excluded.append({"pair_id": key, "sequence_id": sequence,
                                 "reason": "adverse counterpart belongs to reserved reliability_selection"})
                continue
            files = (night_rgb, normal_rgb, night_gt, normal_gt, invalid)
            if not all(p.is_file() for p in files):
                raise FileNotFoundError(f"Incomplete pair: {key}")
            pair = dict(pair_id=key, sequence_id=sequence, split=split,
                        night_rgb=night_id, night_gt=dataset_reference(night_gt),
                        normal_rgb=dataset_reference(normal_rgb), normal_gt=dataset_reference(normal_gt),
                        invalid_path=dataset_reference(invalid))
            pair_rows[split].append(pair)
            for domain in ("night", "normal"):
                rgb = pair[f"{domain}_rgb"]
                gt = pair[f"{domain}_gt"]
                row = dict(pair_id=key, sequence_id=sequence, split=split, domain=domain,
                           rgb_path=rgb, gt_path=gt,
                           invalid_path=pair["invalid_path"] if domain == "night" else "",
                           rgb_sha256=sha256(acdc_path(rgb)), gt_sha256=sha256(acdc_path(gt)))
                if domain == "night":
                    row["invalid_sha256"] = sha256(invalid)
                if split == "fit" and domain == "night":
                    cached = original_cache[rgb]
                    for name in ("rgb_sha256", "gt_sha256"):
                        if row[name] != cached[name]:
                            raise RuntimeError(f"Original training input changed: {rgb}")
                    row.update(cache_path=str(PROJECT_ROOT / cached["cache_path"]),
                               cache_sha256=cached["cache_sha256"],
                               num_valid_pixels=int(cached["num_valid_pixels"]))
                image_rows[split].append(row)
    if (len(pair_rows["fit"]), len(pair_rows["val"]), len(excluded)) != (101, 53, 99):
        raise RuntimeError("Unexpected matched populations; do not silently change experiment")
    fit_keys = {r["pair_id"] for r in pair_rows["fit"]}
    val_keys = {r["pair_id"] for r in pair_rows["val"]}
    if fit_keys & val_keys:
        raise RuntimeError("Fit/validation pair overlap")
    # Exact byte duplicates across experimental fitting/evaluation populations are forbidden.
    for domain in ("night", "normal"):
        hashes = [{r["rgb_sha256"] for r in image_rows[s] if r["domain"] == domain} for s in ("fit", "val")]
        if hashes[0] & hashes[1]:
            raise RuntimeError(f"Byte-identical {domain} RGB across fitting/validation")
    return pair_rows, image_rows, excluded


def freeze(out_dir=OUTPUT_ROOT):
    out_dir = Path(out_dir)
    if (out_dir / "protocol_freeze.json").exists():
        return load_protocol(out_dir)
    if (out_dir / "evaluation_complete.json").exists():
        raise RuntimeError("Cannot preregister after evaluation")
    fitted = read_json(FIT_ROOT / "fitted_temperature.json")
    operating = read_json(PROJECT_ROOT / "results/stage4_reliability/stage4b/frozen_operating_point.json")
    final = read_json(PROJECT_ROOT / "results/stage4_reliability/stage4c/final_reliability_config.json")
    if (fitted["temperature"] != GLOBAL_T or operating["temperature"] != GLOBAL_T
            or operating["threshold"] != GLOBAL_TAU or final["temperature"] != GLOBAL_T
            or final["threshold"] != GLOBAL_TAU or final["model"] != MODEL_NAME
            or final["model_revision"] != MODEL_REVISION or final["uncertainty_method"] != "calibrated_msp"):
        raise RuntimeError("Baseline constants disagree with frozen artifact contents")
    pairs, images, excluded = build_manifests()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_names = []
    for split in ("fit", "val"):
        for suffix, rows in (("pairs", pairs[split]), ("images", images[split])):
            name = f"{split}_{suffix}.csv"
            write_csv(out_dir / name, rows)
            manifest_names.append(name)
    write_csv(out_dir / "excluded_training_pairs.csv", excluded)
    manifest_names.append("excluded_training_pairs.csv")
    artifacts = [FIT_ROOT / "fitted_temperature.json", FIT_ROOT / "cache_index.csv",
                 FIT_ROOT / "cache_metadata.json", FIT_ROOT / "split_manifest.json",
                 FIT_ROOT / "splits/calibration_fit.csv", FIT_ROOT / "splits/reliability_selection.csv",
                 PROJECT_ROOT / "results/stage4_reliability/stage4b/frozen_operating_point.json",
                 PROJECT_ROOT / "results/stage4_reliability/stage4c/final_reliability_config.json",
                 PROJECT_ROOT / "results/stage5_final_evaluation/stage5a/per_image_metrics.csv",
                 PROJECT_ROOT / "results/paired_normal_vs_adverse/night_normal_confusion.npy",
                 PROJECT_ROOT / "results/paired_normal_vs_adverse/night_adverse_confusion.npy"]
    source_paths = sorted(SOURCE_ROOT.glob("*.py"))
    source_paths += sorted(p for i in range(1, 6) for p in (PROJECT_ROOT / f"stage{i}").glob("*.py"))
    source_paths += [PROJECT_ROOT / "stage1/source_layout.json", PROJECT_ROOT / "requirements.txt"]
    protocol = dict(
        version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
        output_directory=str(out_dir.resolve()), dataset_root=str(RGB_ROOT.parents[1]),
        model=MODEL_NAME, model_revision=MODEL_REVISION, global_temperature=GLOBAL_T,
        global_threshold=GLOBAL_TAU, acceptance_rule="1 - max softmax(z/T) <= global_threshold",
        populations={s: {"pairs": len(pairs[s]), "images": len(images[s]),
                         "pair_ids": [r["pair_id"] for r in pairs[s]],
                         "sequences": dict(Counter(r["sequence_id"] for r in pairs[s]))} for s in ("fit", "val")},
        temperatures={"raw": 1.0, "global": GLOBAL_T,
                      "NightPair": "minimize pooled all-valid-pixel NLL on the 101 paired fitting Night images",
                      "NormalPair": "minimize pooled all-valid-pixel NLL on the 101 matched fitting Normal images",
                      "Mix": "minimize 0.5 * domain-mean Normal NLL + 0.5 * domain-mean Night NLL; no weight search"},
        fitting={"optimizer": "existing safeguarded Newton in beta=1/T", "gradient_tolerance": 1e-7,
                 "beta_interval_tolerance": 1e-8, "maximum_iterations": 50,
                 "pixel_subsampling": False, "validation_used": False},
        numerical_conventions={"input_size": [512, 1024], "native_shape": [1, 19, 128, 256],
            "interpolation": "float32 bilinear to own GT size, align_corners=False",
            "prediction": "argmax of unscaled float32 full-resolution logits",
            "raw_probabilities": "float32 softmax and log_softmax",
            "calibrated_probabilities": "float64 division, softmax and log_softmax",
            "validity": "own semantic GT in [0,18]; 255 ignored; no cross-view pixel alignment",
            "ambiguity": "Night invIds==0/1 stratifies semantic-valid pixels; never replaces semantic validity"},
        metrics={"primary": "full-pixel pooled NLL in nats",
            "ece": "15 equal-width top-label confidence bins, right-side interior edges, confidence=1 in last bin",
            "signed_gap": "mean confidence minus pixel accuracy",
            "coverage_valid": "accepted semantic-valid / semantic-valid",
            "coverage_all": "accepted all image pixels / all image pixels",
            "accepted_risk": "incorrect accepted semantic-valid / accepted semantic-valid",
            "high_confidence": 0.9,
            "high_conf_error_rate": "incorrect high-confidence / high-confidence semantic-valid",
            "error_high_conf_fraction": "incorrect high-confidence / all incorrect semantic-valid",
            "class_statistics": "all 19 classes reported descriptively by predicted and GT IDs; no class selection or class thresholds",
            "undefined_ratios": "null / empty CSV cell, never zero",
            "fixed_tau_interpretation": "cutoff-transfer diagnostic, not evidence of ranking gains at unequal coverage"},
        risk_coverage={"seed": 42, "samples_per_image": 8192,
            "sampling": "uniform semantic-valid pixels without replacement; image-specific SHA256-derived RNG seed",
            "weights": "N_valid_image / sample_count_image; same indices and weights across all temperatures",
            "tie_handling": "ascending uncertainty then deterministic label-independent pixel tie keys",
            "checkpoints": [0.5, 0.6, 0.7, 0.8, 0.9],
            "checkpoint_estimator": "nearest cumulative sample-weight coverage, ties first; report actual coverage and mismatch",
            "aurc": "weighted discrete right-step sum delta_coverage * cumulative_risk; estimated from shared sample",
            "deployment": "evaluation cutoffs never used to choose new rejection thresholds"},
        aggregation={"pooled": "sum counts and ECE bin statistics before division, separately per domain",
                     "paired": "Night minus Normal per-image metric; report pair means/medians separately from pooled differences",
                     "sequence": ["GOPR0351", "GOPR0356"],
                     "uncertainty_estimates": "no pixel-IID tests or two-cluster population confidence intervals"},
        integrity={"validation_previously_explored": True, "new_thresholds": False,
                   "class_thresholds": False, "retraining": False,
                   "selection_pairs_excluded": 99, "normal_reference_needed_at_inference": False,
                   "limitations": ["two adverse validation source-sequence IDs", "possible chapter-family dependence",
                       "Normal filenames inherit adverse identifiers", "geographical independence not locally verified",
                       "global baseline fitting population differs from controlled paired fits"]},
        criteria={"calibration": "report NLL and ECE changes separately; consistent improvement requires same direction in both validation sequences",
                  "selective": "report matched-coverage deltas; 1 percentage point at 80% is a prespecified practical-interest threshold, not significance",
                  "stop": ["parity or source integrity failure", "nonfinite fit or unconverged fit",
                           "no consistent practically useful ranking improvement: recommend no added class-aware complexity"],
                  "no_adaptation": "no temperature objective, class subset, mixture weight or coverage checkpoint may change based on validation"},
        source_hashes={str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in source_paths},
        artifact_hashes={str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in artifacts},
        manifest_hashes={name: sha256(out_dir / name) for name in manifest_names})
    write_json(out_dir / "protocol.json", protocol)
    digest = sha256(out_dir / "protocol.json")
    write_json(out_dir / "protocol_freeze.json", {"protocol_sha256": digest,
               "frozen_utc": protocol["frozen_utc"], "validation_evaluations_started": False})
    write_json(out_dir / "manifest_verification.json", {
        "status": "PASS", "fit_pairs": 101, "validation_pairs": 53,
        "selection_pairs_excluded": 99, "all_required_files_present": True,
        "fit_val_pair_overlap": 0, "fit_val_exact_rgb_duplicate_overlap": 0,
        "source_sequence_independence_claimed": False})
    return load_protocol(out_dir)
