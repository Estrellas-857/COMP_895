"""Evaluate frozen policies on own-GT pairs, with full-pixel and sampled metrics."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from stage1.dataset import acdc_path
from .common import PROJECT_ROOT, GLOBAL_TAU, GLOBAL_T, read_csv, read_json, write_csv, write_json, sha256
from .metrics import MetricAccumulator, sample_indices, weighted_risk_coverage

TEMPERATURE_ORDER = ("raw", "global", "NightPair", "NormalPair", "Mix")
PAIR_METRICS = ("nll", "ece", "signed_gap", "coverage_valid", "coverage_all", "accepted_risk",
                "high_conf_error_rate", "error_high_conf_fraction", "high_conf_error_fraction_all_valid")


@torch.inference_mode()
def score_maps(logits32, gt, temperature, raw=False):
    """Preserve frozen raw/calibrated precision and stable GT log probability."""
    if logits32.dtype != torch.float32 or logits32.shape[:2] != (1, 19):
        raise ValueError("Expected float32 full-resolution 19-class logits")
    if tuple(logits32.shape[-2:]) != gt.shape or temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Bad shape or temperature")
    if raw and temperature != 1:
        raise ValueError("Raw configuration requires T=1")
    scaled = logits32 if raw else logits32.to(torch.float64) / temperature
    # Deliberately compute softmax separately: exp(log_softmax) need not match
    # the frozen softmax confidence at floating-point boundaries.
    confidence = torch.softmax(scaled, dim=1).max(dim=1).values[0].cpu().numpy()
    target = torch.as_tensor(gt.astype(np.int64), device=logits32.device)
    safe = target.masked_fill((target < 0) | (target >= 19), 0)
    logp = F.log_softmax(scaled, dim=1)
    nll_map = (-logp[0].gather(0, safe[None])[0]).cpu().numpy()
    canonical = logits32.argmax(dim=1)[0]
    scaled_prediction = scaled.argmax(dim=1)[0]
    changes = int(torch.count_nonzero(canonical != scaled_prediction).item())
    return confidence, nll_map, changes


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                raise RuntimeError(f"Existing array keys differ: {path}")
            for key, value in arrays.items():
                np.testing.assert_array_equal(existing[key], value)
        return
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def paired_statistics(per_image):
    index = defaultdict(dict)
    for row in per_image:
        index[(row["temperature_name"], row["pair_id"], row["sequence_id"])][row["domain"]] = row
    differences = []
    for (name, pair, sequence), domains in sorted(index.items()):
        if set(domains) != {"night", "normal"}:
            raise RuntimeError("Incomplete pair in evaluated statistics")
        row = dict(temperature_name=name, pair_id=pair, sequence_id=sequence,
                   direction="Night minus Normal; own GT; no pixel alignment")
        for metric in PAIR_METRICS:
            n, r = domains["night"][metric], domains["normal"][metric]
            row[metric + "_delta"] = n-r if n is not None and r is not None else None
        differences.append(row)
    summaries = []
    for name in TEMPERATURE_ORDER:
        for sequence in ("pooled", "GOPR0351", "GOPR0356"):
            selected = [r for r in differences if r["temperature_name"] == name
                        and (sequence == "pooled" or r["sequence_id"] == sequence)]
            if not selected:
                continue
            for metric in PAIR_METRICS:
                values = [r[metric+"_delta"] for r in selected if r[metric+"_delta"] is not None]
                summaries.append(dict(temperature_name=name, sequence_id=sequence, metric=metric,
                    num_pairs=len(selected), num_defined=len(values),
                    mean_delta=float(np.mean(values)) if values else None,
                    median_delta=float(np.median(values)) if values else None,
                    pairs_positive=sum(v > 0 for v in values), pairs_negative=sum(v < 0 for v in values),
                    aggregation="equal image-pair weights; not pooled pixel difference"))
    return differences, summaries


def _check_frozen_night(row, measured, old):
    reference = old[row["rgb_path"]]
    checks = {"num_all": "all_pixels", "num_accepted_all": "all_accepted_pixels",
              "num_valid": "valid_pixels", "num_accepted": "accepted_valid_pixels",
              "num_accepted_errors": "FN", "num_errors": "incorrect_valid_pixels"}
    for actual, expected in checks.items():
        if measured[actual] != int(reference[expected]):
            raise RuntimeError(f"Frozen Night parity failed for {row['pair_id']} {actual}: "
                               f"{measured[actual]} != {reference[expected]}")


@torch.inference_mode()
def evaluate(rows, temperatures, out_dir, device, protocol, protocol_sha256):
    """Compute all configurations once per cached validation image.

    Authoritative tables are published only after every frozen-baseline parity
    check passes. Nothing from these outcomes can feed temperature fitting.
    """
    out_dir, device = Path(out_dir), torch.device(device)
    expected = set(protocol["populations"]["val"]["pair_ids"])
    if len(rows) != 106 or any(r["split"] != "val" for r in rows):
        raise ValueError("Expected exactly the 106 registered validation images")
    for domain in ("night", "normal"):
        subset = [r for r in rows if r["domain"] == domain]
        if len(subset) != 53 or {r["pair_id"] for r in subset} != expected:
            raise ValueError("Validation image population differs from preregistration")
    if set(temperatures) != set(TEMPERATURE_ORDER) or temperatures["raw"] != 1 or temperatures["global"] != GLOBAL_T:
        raise ValueError("Unexpected temperature configurations")
    fit_hashes = {name: sha256(out_dir / "temperatures" / f"{name}.json")
                  for name in ("NightPair", "NormalPair", "Mix")}
    for name in fit_hashes:
        fit = read_json(out_dir / "temperatures" / f"{name}.json")
        if (fit["temperature"] != temperatures[name] or fit["protocol_sha256"] != protocol_sha256
                or not fit["converged"] or fit["validation_used"]):
            raise RuntimeError("Evaluation temperature differs from its frozen training artifact")
    start_path = out_dir / "evaluation_start.json"
    if not start_path.exists():
        write_json(start_path, dict(started_utc=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=protocol_sha256, temperature_artifact_hashes=fit_hashes,
                   population="53 registered Normal/Night validation pairs"))
    start_record = read_json(start_path)
    if start_record["protocol_sha256"] != protocol_sha256 or start_record["temperature_artifact_hashes"] != fit_hashes:
        raise RuntimeError("Evaluation start provenance mismatch")
    if (out_dir / "evaluation_complete.json").exists():
        completed = read_json(out_dir / "evaluation_complete.json")
        for relative, digest in completed["output_hashes"].items():
            if sha256(out_dir / relative) != digest:
                raise RuntimeError(f"Completed evaluation output changed: {relative}")
        return completed
    old = {r["rgb_path"]: r for r in read_csv(PROJECT_ROOT / "results/stage5_final_evaluation/stage5a/per_image_metrics.csv")}
    pooled = defaultdict(MetricAccumulator)
    ambiguous = defaultdict(MetricAccumulator)
    samples = defaultdict(list)
    per_image, image_resources, invariance = [], [], []
    begin = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for number, row in enumerate(rows, 1):
        started = time.perf_counter()
        if sha256(row["cache_path"]) != row["cache_sha256"] or sha256(acdc_path(row["gt_path"])) != row["gt_sha256"]:
            raise RuntimeError("Validation cache or GT hash mismatch")
        with Image.open(acdc_path(row["gt_path"])) as image:
            gt = np.array(image)
        if gt.shape != (1080, 1920) or not np.isin(gt, list(range(19)) + [255]).all():
            raise RuntimeError("Unexpected validation labels")
        native = np.load(row["cache_path"], allow_pickle=False)
        if native.shape != (1, 19, 128, 256) or native.dtype != np.float32:
            raise ValueError("Unexpected validation native logits")
        logits = F.interpolate(torch.from_numpy(native).to(device), size=gt.shape,
                               mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        sampling = sample_indices(gt, row["domain"]+"/"+row["pair_id"],
                                  seed=protocol["risk_coverage"]["seed"],
                                  n=protocol["risk_coverage"]["samples_per_image"])
        indices = sampling["indices"]
        sample = dict(pair_id=row["pair_id"], sequence_id=row["sequence_id"],
                      indices=indices, weights=sampling["weights"], tie_keys=sampling["tie_keys"],
                      errors=(prediction.ravel()[indices] != gt.ravel()[indices]))
        invalid = None
        if row["domain"] == "night":
            if sha256(acdc_path(row["invalid_path"])) != row["invalid_sha256"]:
                raise RuntimeError("Ambiguity annotation hash mismatch")
            with Image.open(acdc_path(row["invalid_path"])) as image:
                invalid = np.array(image)
            if invalid.shape != gt.shape or not np.isin(invalid, [0, 1]).all():
                raise RuntimeError("Unexpected invalid-region encoding")
        for name in TEMPERATURE_ORDER:
            temperature = float(temperatures[name])
            confidence, nll_map, changes = score_maps(logits, gt, temperature, raw=name == "raw")
            if changes:
                raise RuntimeError(f"Positive temperature changed canonical prediction: {row['pair_id']}, {name}")
            accumulator = MetricAccumulator().update(confidence, nll_map, prediction, gt)
            measured = accumulator.finalize()
            if row["domain"] == "night" and name == "global":
                _check_frozen_night(row, measured, old)
            per_image.append(dict(pair_id=row["pair_id"], sequence_id=row["sequence_id"],
                                  domain=row["domain"], temperature_name=name, temperature=temperature,
                                  policy_role="frozen_global" if name == "global" else "fixed_cutoff_transfer_diagnostic",
                                  **measured))
            for sequence in ("pooled", row["sequence_id"]):
                pooled[(row["domain"], name, sequence)].merge(accumulator)
                if invalid is not None:
                    for region, flag in (("clear", 0), ("ambiguous", 1)):
                        ambiguous[(name, sequence, region)].update(confidence, nll_map, prediction, gt, invalid == flag)
            sample[name] = (1.0-confidence.ravel()[indices]).astype(np.float64)
            invariance.append(dict(pair_id=row["pair_id"], domain=row["domain"], temperature_name=name,
                                   prediction_changes=changes, pixels_checked=int(gt.size)))
            del confidence, nll_map, accumulator
        samples[row["domain"]].append(sample)
        image_resources.append(dict(pair_id=row["pair_id"], domain=row["domain"],
                                    seconds=time.perf_counter()-started))
        del logits, native, prediction, gt, invalid
        if number == 1 or number % 10 == 0 or number == len(rows):
            print(f"Evaluation [{number}/{len(rows)}] complete; frozen Night parity checked; elapsed={time.perf_counter()-begin:.1f}s", flush=True)
    # Canonical whole-pair confusion must reproduce Stage 1 for both domains.
    for domain in ("night", "normal"):
        historical = PROJECT_ROOT / f"results/paired_normal_vs_adverse/night_{'adverse' if domain == 'night' else 'normal'}_confusion.npy"
        expected_confusion = np.load(historical, allow_pickle=False)
        for name in TEMPERATURE_ORDER:
            np.testing.assert_array_equal(pooled[(domain, name, "pooled")].confusion, expected_confusion)
    calibration, sequences, classes, bins, confusion_rows = [], [], [], [], []
    for (domain, name, sequence), accumulator in sorted(pooled.items()):
        identifiers = dict(domain=domain, temperature_name=name,
                           temperature=float(temperatures[name]), sequence_id=sequence)
        result = dict(identifiers, **accumulator.finalize())
        (calibration if sequence == "pooled" else sequences).append(result)
        classes.extend(dict(identifiers, **r) for r in accumulator.class_rows())
        bins.extend(dict(identifiers, **r) for r in accumulator.confidence_bin_rows())
        if name == "raw":
            confusion_rows.extend(dict(domain=domain, sequence_id=sequence, gt_class=g, predicted_class=p,
                                       count=int(accumulator.confusion[g, p])) for g in range(19) for p in range(19))
    region_rows = [dict(domain="night", temperature_name=name, temperature=float(temperatures[name]),
                        sequence_id=sequence, region=region, **acc.finalize())
                   for (name, sequence, region), acc in sorted(ambiguous.items())]
    checkpoints, areas = [], []
    for domain, records in samples.items():
        # Save the actual common sample, including indices and tie keys, for reproduction.
        common = dict(pair_ids=np.array([r["pair_id"] for r in records]),
                      sequence_ids=np.array([r["sequence_id"] for r in records]),
                      sample_sizes=np.array([len(r["indices"]) for r in records]),
                      **{k: np.concatenate([r[k] for r in records])
                         for k in ("indices", "weights", "tie_keys", "errors", *TEMPERATURE_ORDER)})
        _save_npz(out_dir / "samples" / f"{domain}.npz", **common)
        for sequence in ("pooled", "GOPR0351", "GOPR0356"):
            subset = [r for r in records if sequence == "pooled" or r["sequence_id"] == sequence]
            weights = np.concatenate([r["weights"] for r in subset])
            errors = np.concatenate([r["errors"] for r in subset])
            ties = np.concatenate([r["tie_keys"] for r in subset])
            for name in TEMPERATURE_ORDER:
                curve = weighted_risk_coverage(np.concatenate([r[name] for r in subset]), errors, weights, ties,
                                              targets=protocol["risk_coverage"]["checkpoints"])
                identifiers = dict(domain=domain, temperature_name=name, temperature=float(temperatures[name]),
                                   sequence_id=sequence, estimated=True, num_samples=curve["num_samples"])
                checkpoints.extend(dict(identifiers, **r) for r in curve["checkpoints"])
                areas.append(dict(identifiers, aurc=curve["aurc"], estimator=curve["method"]))
                _save_npz(out_dir / "curves" / f"{domain}_{name}_{sequence}.npz",
                          **{k: curve[k] for k in ("coverage", "risk", "accepted_weight", "error_weight")})
    differences, pair_summary = paired_statistics(per_image)
    outputs = {"calibration_comparison.csv": calibration,
               "frozen_policy_comparison.csv": [dict(r, policy_role="frozen_global" if r["temperature_name"] == "global"
                                                     else "fixed_cutoff_transfer_diagnostic") for r in calibration],
               "sequence_metrics.csv": sequences, "per_image_metrics.csv": per_image,
               "paired_differences.csv": differences, "paired_summary.csv": pair_summary,
               "class_metrics.csv": classes, "confidence_bins.csv": bins,
               "class_confusions.csv": confusion_rows, "ambiguity_metrics.csv": region_rows,
               "matched_coverage.csv": checkpoints, "aurc.csv": areas,
               "prediction_invariance.csv": invariance, "evaluation_per_image_runtime.csv": image_resources}
    for name, data in outputs.items():
        # A retry preserves the first completed timing observation; deterministic
        # scientific outputs must still reproduce exactly.
        if name != "evaluation_per_image_runtime.csv" or not (out_dir / name).exists():
            write_csv(out_dir / name, data)
    write_json(out_dir / "parity_checks.json", dict(status="PASS", night_images_exactly_matching_stage5a=53,
               both_domain_confusions_exactly_matching_stage1=True,
               temperature_prediction_changes=0, protocol_sha256=protocol_sha256))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    resources = dict(
        protocol_sha256=protocol_sha256, elapsed_seconds=time.perf_counter()-begin, num_images=106,
        num_temperature_image_evaluations=len(per_image),
        peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        timing_boundary="cached-logit loading, own GT, all five scores/metrics, sample curves and table serialization; no network forward",
        timing_caveat="single session with other GPU workloads; not an uncontended inference benchmark")
    if not (out_dir / "resources" / "evaluation.json").exists():
        write_json(out_dir / "resources" / "evaluation.json", resources)
    hashes = {str(p.relative_to(out_dir)): sha256(p) for p in out_dir.rglob("*")
              if p.is_file() and (p.name in outputs or p.parent.name in {"curves", "samples"})}
    completed = dict(status="PASS", completed_utc=datetime.now(timezone.utc).isoformat(),
                     protocol_sha256=protocol_sha256, temperature_artifact_hashes=fit_hashes,
                     output_hashes=hashes, full_pixel_metrics=True, sampled_ranking_metrics=True)
    write_json(out_dir / "evaluation_complete.json", completed)
    return completed
