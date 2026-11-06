"""Summarize all 17 completed outer groups without selecting favorable results."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .common import ROOT, save_json, sha256
from .evaluation import CONFIGS, HistogramAccumulator
from .protocol import load_protocol, read_csv

COMPARISONS = (("lr3", "raw_msp"), ("lr3", "raw_entropy"),
               ("lr3_class", "raw_msp"), ("lr3_class", "raw_entropy"),
               ("lr3_class", "lr3"))
LABELS = {"raw_msp": "Raw MSP", "raw_entropy": "Raw entropy",
          "lr3": "Logistic regression", "lr3_class": "Logistic regression + class"}
COLORS = {"raw_msp": "#414487", "raw_entropy": "#808080", "lr3": "#238a8d", "lr3_class": "#da8c28"}


def checkpoint80(metrics):
    matches = [p for p in metrics["checkpoints"] if abs(float(p["requested_coverage"]) - 0.8) < 1e-12]
    if len(matches) != 1 or matches[0]["risk"] is None:
        raise ValueError("Exactly one finite retrospective80 checkpoint is required")
    return matches[0]


def metric_row(scope, identity, method, metrics, weather="", num_groups=1):
    checkpoint, threshold = checkpoint80(metrics), metrics["fixed_threshold"]
    return dict(scope=scope, identity=identity, weather=weather, num_groups=num_groups, method=method,
                num_valid_pixels=metrics["num_pixels"], num_error_pixels=metrics["num_errors"],
                error_prevalence=metrics["error_prevalence"], auroc=metrics["auroc"],
                average_precision=metrics["average_precision"], aurc=metrics["aurc"],
                retrospective80_risk=checkpoint["risk"], retrospective80_actual_coverage=checkpoint["actual_coverage"],
                retrospective80_coverage_error=checkpoint["coverage_error"],
                training_threshold=threshold["threshold"], distinct_training_thresholds=threshold["distinct_thresholds"],
                training_threshold_actual_coverage=threshold["coverage"],
                training_threshold_accepted_error=threshold["risk"],
                threshold_evaluated_pixels=threshold["num_evaluated"],
                threshold_accepted_pixels=threshold["accepted_pixels"],
                threshold_accepted_error_pixels=threshold["accepted_errors"])


def paired_deltas(group_results):
    rows = []
    for group, result in sorted(group_results.items()):
        for method, baseline in COMPARISONS:
            a, b = result["methods"][method], result["methods"][baseline]
            ac, bc = checkpoint80(a), checkpoint80(b)
            rows.append(dict(group=group, weather=result["weather"], method=method, baseline=baseline,
                             risk_delta=ac["risk"] - bc["risk"],
                             method_risk=ac["risk"], baseline_risk=bc["risk"],
                             method_actual_coverage=ac["actual_coverage"],
                             baseline_actual_coverage=bc["actual_coverage"],
                             actual_coverage_difference=ac["actual_coverage"] - bc["actual_coverage"],
                             aurc_delta=a["aurc"] - b["aurc"],
                             auroc_delta=None if a["auroc"] is None or b["auroc"] is None else a["auroc"] - b["auroc"],
                             ap_delta=None if a["average_precision"] is None or b["average_precision"] is None
                                      else a["average_precision"] - b["average_precision"]))
    return rows


def summarize_deltas(rows):
    summaries = []
    scopes = ["all"] + sorted({r["weather"] for r in rows})
    for scope in scopes:
        for method, baseline in COMPARISONS:
            selected = [r for r in rows if r["method"] == method and r["baseline"] == baseline
                        and (scope == "all" or r["weather"] == scope)]
            if not selected:
                continue
            values = np.array([r["risk_delta"] for r in selected], dtype=np.float64)
            omitted = (values.sum() - values) / (len(values) - 1) if len(values) > 1 else np.array([])
            summary = dict(scope=scope, method=method, baseline=baseline, num_groups=len(values),
                                  mean_risk_delta=float(values.mean()), median_risk_delta=float(np.median(values)),
                                  min_risk_delta=float(values.min()), max_risk_delta=float(values.max()),
                                  improved_groups=int(np.count_nonzero(values < -1e-12)),
                                  tied_groups=int(np.count_nonzero(abs(values) <= 1e-12)),
                                  worsened_groups=int(np.count_nonzero(values > 1e-12)),
                                  leave_one_group_out_mean_min=float(omitted.min()) if omitted.size else None,
                                  leave_one_group_out_mean_max=float(omitted.max()) if omitted.size else None,
                                  max_abs_method_coverage_error=max(abs(r["method_actual_coverage"] - .8) for r in selected),
                                  max_abs_baseline_coverage_error=max(abs(r["baseline_actual_coverage"] - .8) for r in selected),
                                  max_abs_paired_coverage_difference=max(abs(r["actual_coverage_difference"]) for r in selected))
            directions = [-summary["mean_risk_delta"]]
            for metric, sign in (("auroc", 1), ("ap", 1), ("aurc", -1)):
                available = [r[f"{metric}_delta"] for r in selected if r.get(f"{metric}_delta") is not None]
                mean = float(np.mean(available)) if available else None
                summary[f"mean_{metric}_delta"] = mean
                summary[f"{metric}_num_groups"] = len(available)
                if mean is not None:
                    directions.append(sign * mean)
            summary["opposite_metric_directions"] = any(v > 1e-12 for v in directions) and any(v < -1e-12 for v in directions)
            summaries.append(summary)
    return summaries


def metric_delta_text(summary):
    fields = []
    for name in ("auroc", "ap", "aurc"):
        value = summary[f"mean_{name}_delta"]
        formatted = f"{value:+.6f}" if value is not None else "undefined"
        fields.append(f"mean paired {name.upper()} delta {formatted} (n={summary[f'{name}_num_groups']})")
    if summary["opposite_metric_directions"]:
        fields.append("opposite metric directions: a tradeoff, not overall superiority")
    return "; ".join(fields)


def extraction_summary(records):
    """Summarize canonical extraction and all-image historical prediction checks."""
    if not records or len({r["image_id"] for r in records}) != len(records):
        raise ValueError("Feature records must contain unique images")
    origins = Counter(r["native_cache"]["origin"] for r in records)
    if set(origins) - {"reused", "new_frozen_forward"}:
        raise ValueError("Unknown native-cache provenance")
    for record in records:
        if not 0 <= record["num_errors"] <= record["num_valid"] or record["num_ignored"] < 0:
            raise ValueError("Invalid extraction pixel counts")
        if record["num_valid"] + record["num_ignored"] != 1080 * 1920:
            raise ValueError("Extraction population does not cover the image")
    compared = [r for r in records if r["historical_prediction_changes"] is not None]
    if any(not 0 <= r["historical_prediction_changes"] <= 1080 * 1920 for r in compared):
        raise ValueError("Invalid historical prediction difference count")
    normalization = max(float(r["probability_normalization_max_error"]) for r in records)
    if not np.isfinite(normalization) or normalization < 0:
        raise ValueError("Invalid probability normalization diagnostic")
    return dict(num_images=len(records), reused_native_images=origins["reused"],
                new_forward_images=origins["new_frozen_forward"],
                num_valid_pixels=sum(r["num_valid"] for r in records),
                num_error_pixels=sum(r["num_errors"] for r in records),
                num_ignored_pixels=sum(r["num_ignored"] for r in records),
                probability_normalization_max_error=normalization,
                historical_prediction_images_compared=len(compared),
                historical_prediction_images_differing=sum(r["historical_prediction_changes"] > 0 for r in compared),
                historical_prediction_pixels_differing=sum(r["historical_prediction_changes"] for r in compared),
                historical_comparison_population="All image pixels, including semantic-ignored pixels",
                prediction_definition="All four methods share argmax of unscaled float32 full-resolution logits",
                historical_parity="No exact prediction or metric parity claim with historical Stage 4; original files remain unchanged")


def restore_accumulator(metrics, correct, incorrect):
    correct, incorrect = np.asarray(correct), np.asarray(incorrect)
    if (correct.ndim != 1 or correct.size < 1 or incorrect.shape != correct.shape or
            not np.issubdtype(correct.dtype, np.integer) or not np.issubdtype(incorrect.dtype, np.integer) or
            np.any(correct < 0) or np.any(incorrect < 0)):
        raise ValueError("Expected nonnegative full-population one-dimensional integer histogram counts")
    if int(correct.sum() + incorrect.sum()) != metrics["num_pixels"] or int(incorrect.sum()) != metrics["num_errors"]:
        raise ValueError("Histogram/result population mismatch")
    fixed = metrics["fixed_threshold"]
    if (not fixed["complete_population"] or fixed["num_evaluated"] != metrics["num_pixels"] or
            fixed["distinct_thresholds"] != 1 or fixed["threshold"] is None):
        raise ValueError("Each outer group must have one complete training-derived threshold evaluation")
    if not 0 <= fixed["accepted_errors"] <= fixed["accepted_pixels"] <= fixed["num_evaluated"]:
        raise ValueError("Invalid accepted counts")
    accumulator = HistogramAccumulator(bins=len(correct), threshold=fixed["threshold"])
    accumulator.correct[:] = correct
    accumulator.incorrect[:] = incorrect
    accumulator.threshold_evaluated = fixed["num_evaluated"]
    accumulator.accepted = fixed["accepted_pixels"]
    accumulator.accepted_errors = fixed["accepted_errors"]
    accumulator.thresholds.add(float(fixed["threshold"]))
    rebuilt = accumulator.finalize()
    for name in ("error_prevalence", "auroc", "average_precision", "aurc"):
        if metrics[name] is None and rebuilt[name] is None:
            continue
        if metrics[name] is None or rebuilt[name] is None or not np.isclose(metrics[name], rebuilt[name], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Histogram/result {name} disagreement")
    if abs(checkpoint80(metrics)["risk"] - checkpoint80(rebuilt)["risk"]) > 1e-12:
        raise ValueError("Histogram/result retrospective80 disagreement")
    return accumulator


def _write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path):
    return json.loads(path.read_text())


def collect(root=ROOT):
    root = Path(root)
    load_protocol(root / "data")
    assignments = _read_json(root / "data/assignments.json")
    expected = {a["eval_group"] for a in assignments}
    paths = {p.parent.name: p for p in (root / "results/folds").glob("*/result.json")}
    if len(expected) != 17 or set(paths) != expected:
        raise ValueError(f"Final report requires exactly all 17 outer groups; missing={sorted(expected-set(paths))}, extra={sorted(set(paths)-expected)}")
    metadata = read_csv(root / "data/metadata_train.csv")
    protocol_digest = sha256(root / "data/protocol.json")
    sources, groups, selections, accumulators = {}, {}, {}, {}
    for group in sorted(expected):
        path = paths[group]
        result = _read_json(path)
        members = [r for r in metadata if r["recording_group"] == group]
        if (result["group"] != group or result["num_images"] != len(members) or
                {r["condition"] for r in members} != {result["weather"]} or set(result["methods"]) != set(CONFIGS)):
            raise ValueError(f"Group result does not match protocol population: {group}")
        if len({(m["num_pixels"], m["num_errors"]) for m in result["methods"].values()}) != 1:
            raise ValueError(f"Methods did not evaluate identical valid/error pixel counts: {group}")
        histogram_path = path.parent / "histograms.npz"
        accumulators[group] = {}
        with np.load(histogram_path, allow_pickle=False) as histograms:
            for method in CONFIGS:
                accumulators[group][method] = restore_accumulator(result["methods"][method],
                    histograms[f"{method}_correct"], histograms[f"{method}_incorrect"])
        selection_path = root / "data/models" / group / "selection.json"
        selections[group] = _read_json(selection_path)
        if (selections[group]["protocol_sha256"] != protocol_digest or
                selections[group]["source_hashes"] != result["source_hashes"]):
            raise ValueError(f"Training/evaluation protocol or source mismatch: {group}")
        for name, digest in result["source_hashes"].items():
            if sha256(root / name) != digest:
                raise ValueError(f"Execution source changed since evaluation: {name}")
            sources[name] = digest
        if set(selections[group]["selected"]) != {"lr3", "lr3_class"}:
            raise ValueError(f"Missing learned-method selection: {group}")
        for method in CONFIGS:
            threshold = selections[group]["thresholds"][method]
            if abs(float(threshold["value"]) - float(result["methods"][method]["fixed_threshold"]["threshold"])) > 1e-12:
                raise ValueError(f"Training/evaluation threshold disagreement: {group}/{method}")
        for source in (path, histogram_path, selection_path):
            sources[str(source.relative_to(root))] = sha256(source)
        groups[group] = result
    return groups, selections, accumulators, sources


def _plots(output, groups, aggregated, deltas, bin_count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    names = ["fog", "night", "rain", "snow", "all"]
    for ax, name in zip(axes.flat, names):
        for method in CONFIGS:
            curve = aggregated[name][method]["risk_coverage"]
            ax.plot(curve["coverage"], curve["risk"], color=COLORS[method], label=LABELS[method], linewidth=1.6)
        ax.set(title=f"{name.capitalize()} (cross-fitted scores)", xlabel="Accepted pixel coverage", ylabel="Accepted error rate", xlim=(0, 1), ylim=(0, 1))
        ax.axvline(.8, color="#cccccc", linestyle=":", linewidth=.8)
        ax.grid(alpha=.18)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False)
    axes.flat[-1].text(.02, .45, f"All valid pixels; {bin_count:,} fixed transformed bins.\nEach recording group uses its own held-out model.\nPooled curves are cross-fitted diagnostics,\nnot estimates for one deployed model.", transform=axes.flat[-1].transAxes, fontsize=10, va="top")
    fig.suptitle("Risk–coverage diagnostics across all outer folds", fontsize=15)
    fig.savefig(output / "risk_coverage_crossfitted.png", dpi=180)
    plt.close(fig)

    ordered = sorted(groups, key=lambda group: (groups[group]["weather"], group))
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    for method, offset in (("lr3", -.18), ("lr3_class", .18)):
        lookup = {r["group"]: r for r in deltas if r["method"] == method and r["baseline"] == "raw_msp"}
        ax.barh(y + offset, [100 * lookup[g]["risk_delta"] for g in ordered], height=.34, color=COLORS[method], label=LABELS[method])
    ax.set_yticks(y, [f"{g} · {groups[g]['weather']} · {groups[g]['num_images']} images" for g in ordered])
    ax.axvline(0, color="#444444", linewidth=.8)
    ax.set(xlabel="Accepted-error difference vs raw MSP (percentage points; negative is better)",
           title="Each held-out recording proxy at nearest attainable 80% coverage")
    ax.grid(axis="x", alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(output / "paired_group_risk80_deltas.png", dpi=180)
    plt.close(fig)


def generate(root=ROOT):
    root = Path(root)
    if root.resolve() != ROOT.resolve():
        raise ValueError("Final artifacts must remain in the CSE805 extension")
    groups, selections, accumulators, sources = collect(root)
    bin_counts = {a.bins for methods in accumulators.values() for a in methods.values()}
    if len(bin_counts) != 1:
        raise ValueError("All methods/groups must use identical histogram binning")
    bin_count = bin_counts.pop()
    output = root / "results/report"
    output.mkdir(parents=True, exist_ok=True)
    group_rows = [{**metric_row("group", g, m, r["methods"][m], r["weather"]),
                   "num_images": r["num_images"],
                   "training_sample_coverage": selections[g]["thresholds"][m]["training_sample_coverage"]}
                  for g, r in sorted(groups.items()) for m in CONFIGS]
    aggregate, weather_rows, pooled_rows = {}, [], []
    for weather in ["fog", "night", "rain", "snow", "all"]:
        members = [g for g in groups if weather == "all" or groups[g]["weather"] == weather]
        aggregate[weather] = {}
        for method in CONFIGS:
            accumulator = HistogramAccumulator(bins=bin_count)
            for group in members:
                accumulator.merge(accumulators[group][method])
            metrics = accumulator.finalize()
            aggregate[weather][method] = metrics
            row = metric_row("pooled_crossfitted" if weather == "all" else "weather_crossfitted",
                             weather, method, metrics, weather, len(members))
            row["num_images"] = sum(groups[g]["num_images"] for g in members)
            (pooled_rows if weather == "all" else weather_rows).append(row)
    deltas = paired_deltas(groups)
    delta_summary = summarize_deltas(deltas)
    qa_rows, selection_rows, runtime_rows = [], [], []
    for group, result in sorted(groups.items()):
        for method in CONFIGS:
            qa = result["qa"][method]
            for metric in ("auroc", "average_precision", "aurc"):
                exact, quantized = qa["exact"][metric], qa["quantized"][metric]
                qa_rows.append(dict(group=group, method=method, metric=metric, exact=exact, quantized=quantized,
                                    difference=None if exact is None or quantized is None else quantized - exact,
                                    comparison=("Score quantization plus true-score tie-policy difference: secondary keys versus whole bins"
                                                if metric == "aurc" else "Score quantization; both estimators group true score ties")))
            exact, quantized = checkpoint80(qa["exact"]), checkpoint80(qa["quantized"])
            for metric in ("risk", "actual_coverage"):
                qa_rows.append(dict(group=group, method=method, metric="retrospective80_" + metric,
                                    exact=exact[metric], quantized=quantized[metric], difference=quantized[metric] - exact[metric],
                                    comparison="Score quantization plus true-score tie-policy difference: secondary keys versus whole bins"))
        selection = selections[group]
        for candidate in selection["candidates"]:
            row = {"group": group, **candidate}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, separators=(",", ":"))
            config = candidate["config"]
            row["selected_C"] = selection["selected"][config]
            selection_rows.append(row)
        for phase, runtime in (("training_and_selection", selection.get("runtime", {})), ("full_pixel_evaluation", result.get("runtime", {}))):
            for key, value in runtime.items():
                if key == "score_seconds":
                    runtime_rows.extend(dict(group=group, phase=phase, measure=f"score_seconds.{method}", value=seconds)
                                        for method, seconds in value.items())
                else:
                    runtime_rows.append(dict(group=group, phase=phase, measure=key,
                                             value=json.dumps(value) if isinstance(value, (dict, list)) else value))
    features_path = root / "data/features/manifest.json"
    extraction = _read_json(features_path)
    if len(extraction["records"]) != 1600:
        raise ValueError("Final report requires feature provenance for all 1600 images")
    extraction_qa = extraction_summary(extraction["records"])
    for method in CONFIGS:
        primary_counts = aggregate["all"][method]
        if (primary_counts["num_pixels"] != extraction_qa["num_valid_pixels"] or
                primary_counts["num_errors"] != extraction_qa["num_error_pixels"]):
            raise ValueError(f"Primary evaluation disagrees with canonical extraction counts: {method}")
    # Reused-cache runs refresh manifest elapsed time; per-image records retain original acquisition cost.
    feature_seconds = sum(float(r["elapsed_seconds"]) for r in extraction["records"])
    forward_seconds = sum(float(r["native_cache"].get("elapsed_seconds", 0)) for r in extraction["records"]
                          if r["native_cache"]["origin"] == "new_frozen_forward")
    runtime_rows += [dict(group="all", phase="feature_extraction", measure="summed_per_image_elapsed_seconds", value=feature_seconds),
                     dict(group="all", phase="feature_extraction", measure="new_native_forward_cache_elapsed_seconds_subset", value=forward_seconds)]
    sources[str(features_path.relative_to(root))] = sha256(features_path)
    tables = {"group_metrics.csv": group_rows, "weather_metrics.csv": weather_rows,
              "pooled_crossfitted_metrics.csv": pooled_rows, "paired_deltas.csv": deltas,
              "delta_summary.csv": delta_summary, "qa_discrepancies.csv": qa_rows,
              "model_selection.csv": selection_rows, "runtime.csv": runtime_rows}
    for filename, rows in tables.items():
        _write_csv(output / filename, rows)
    save_json(output / "extraction_qa.json", extraction_qa)
    _plots(output, groups, aggregate, deltas, bin_count)
    main = [r for r in delta_summary if r["scope"] == "all"]
    pooled = aggregate["all"]["raw_msp"]
    lines = ["CSE 805: Learning to Predict Semantic Segmentation Errors", "",
             "Complete nested group evaluation: 1,600 adverse training images, 17 conservative recording proxies.",
             "The original 406-image validation set is excluded. The segmentation backbone remains frozen.",
             "Each outer fold holds out one group; three grouped inner folds select C from 0.01, 0.1 and 1.0 by mean weighted AP (ties choose smaller C).",
             "Training samples 512 valid pixels per image uniformly without replacement, uses inverse inclusion-probability weights, and fits preprocessing only on each training partition.",
             "The shared 8,192-pixel-per-image QA sample checks numerical approximation; primary evaluation uses all valid pixels.",
             f"Evaluated {pooled['num_pixels']:,} valid pixels, {pooled['num_errors']:,} segmentation errors; exact prevalence {pooled['error_prevalence']:.8%}.",
             "", "Extraction QA:",
             f"{extraction_qa['num_images']:,} images; {extraction_qa['reused_native_images']:,} native caches reused and {extraction_qa['new_forward_images']:,} missing frozen forward passes computed.",
             f"Canonical extraction: {extraction_qa['num_valid_pixels']:,} valid pixels, {extraction_qa['num_error_pixels']:,} errors, {extraction_qa['num_ignored_pixels']:,} ignored pixels. These valid/error counts exactly match all four primary evaluations.",
             f"Maximum probability-sum error: {extraction_qa['probability_normalization_max_error']:.10g}.",
             f"Historical predictions compared for {extraction_qa['historical_prediction_images_compared']:,} images: {extraction_qa['historical_prediction_images_differing']:,} images differ in a total of {extraction_qa['historical_prediction_pixels_differing']:,} pixels. This comparison covers all image pixels, including semantic-ignored pixels.",
             "All four methods use the same new canonical raw-logit argmax predictions. No exact parity with historical Stage 4 predictions or metrics is claimed; the original files remain unchanged.",
             "", "Primary endpoint: equally weighted mean held-out-family accepted-error difference at retrospective nearest-bin 80% coverage.",
             "Negative differences favor the first method at the reported operating points. Both target 80%, but whole-bin/true-score ties can cause departures; these are not exactly matched-coverage comparisons. Actual values and discrepancies are in every table.",
             "Secondary group-macro paired deltas are also reported: higher AUROC/AP is better; lower AURC is better. An improved risk80 alongside degraded ranking metrics is a tradeoff, not overall superiority."]
    for item in main:
        lines.append(f"{item['method']} minus {item['baseline']}: mean {100*item['mean_risk_delta']:+.6f} percentage points; "
                     f"improved/tied/worsened families {item['improved_groups']}/{item['tied_groups']}/{item['worsened_groups']}; "
                     f"leave-one-family-out mean range [{100*item['leave_one_group_out_mean_min']:+.6f}, {100*item['leave_one_group_out_mean_max']:+.6f}] pp; "
                     f"maximum paired coverage difference {100*item['max_abs_paired_coverage_difference']:.6f} pp.")
        lines.append(metric_delta_text(item) + ".")
    lines += ["", "Weather-specific equally weighted family means (same endpoint; limited groups):"]
    for item in delta_summary:
        if item["scope"] != "all":
            lines.append(f"{item['scope']}: {item['method']} minus {item['baseline']} = {100*item['mean_risk_delta']:+.6f} pp across {item['num_groups']} groups.")
            if item["baseline"] == "raw_msp":
                lines.append(metric_delta_text(item) + ".")
    worse = sorted([r for r in deltas if r["baseline"] == "raw_msp" and r["risk_delta"] > 1e-12], key=lambda r: -r["risk_delta"])
    lines += ["", "Largest adverse changes against raw MSP (all family results remain in paired_deltas.csv):"]
    lines += [f"{r['group']} ({r['weather']}), {r['method']}: {100*r['risk_delta']:+.6f} pp." for r in worse[:8]] or ["No positive risk differences against raw MSP were observed."]
    statuses = Counter(str(r.get("status", "unrecorded")) for r in selection_rows)
    lines += ["", f"All model-search candidate statuses retained: {dict(statuses)}."]
    for config in ("lr3", "lr3_class"):
        lines.append(f"Selected regularization C for {config}: {dict(sorted(Counter(str(s['selected'][config]) for s in selections.values()).items()))}.")
    lines += ["", "Numerical/tie-policy QA: maximum absolute binned-minus-exact difference on identical weighted QA samples:",
              "AUROC/AP differences measure score quantization with true-score ties grouped in both estimators.",
              "AURC/risk/coverage differences combine quantization with different treatment of true-score ties: the exact sample estimator uses label-independent secondary keys, while the histogram accepts whole bins. These differences are not pure binning errors, and refining bins cannot remove exact-score ties."]
    for metric in ("auroc", "average_precision", "aurc", "retrospective80_risk", "retrospective80_actual_coverage"):
        values = [abs(r["difference"]) for r in qa_rows if r["metric"] == metric and r["difference"] is not None]
        lines.append(f"{metric}: {max(values):.10f}" if values else f"{metric}: undefined")
    primary = [r for r in main if r["baseline"] == "raw_msp"]
    training_seconds = sum(float(s["runtime"]["total_seconds"]) for s in selections.values())
    evaluation_seconds = sum(float(r["runtime"]["total_seconds"]) for r in groups.values())
    evaluation_feature_seconds = sum(float(r["runtime"]["feature_seconds"]) for r in groups.values())
    scoring_seconds = {m: sum(float(r["runtime"]["score_seconds"][m]) for r in groups.values()) for m in CONFIGS}
    lines += ["", "Cost and interpretation:",
              f"Original summed per-image feature extraction records: {feature_seconds:.3f} s; new frozen-forward/cache subset: {forward_seconds:.3f} s. The latest manifest-refresh elapsed time is not used as extraction cost.",
              "These extraction timings include I/O, hash checks, preprocessing, sampling and cache writes; the forward subset also includes cache handling. They are not pure neural-network latency.",
              f"All nested model searches/refits/threshold selection: {training_seconds:.3f} s. All full-pixel evaluation: {evaluation_seconds:.3f} s, including {evaluation_feature_seconds:.3f} s feature regeneration.",
              "Full-pixel score calls (exclude histogram updates; include input validation and model prediction): " + "; ".join(f"{m}={scoring_seconds[m]:.3f} s" for m in CONFIGS) + ".",
              "Per-fold timings retain their recorded boundaries in runtime.csv. Timings overlap by inclusion and must not all be added. This experiment does not establish production throughput.",
              "Runtime totals describe the completed formal experiment, not total project wall-clock time. They exclude smoke runs, package installation, the protocol audit, and the first failed empty-chunk attempt (approximately one outer-fold training plus partial evaluation), retained under logs/failed_empty_chunk_attempt.",
              ("At least one learned method has a lower mean primary error rate than MSP. This does not establish overall superiority: costs must be judged with group-macro AUROC/AP/AURC, family consistency, entropy comparisons and negative cases above."
               if any(r["mean_risk_delta"] < 0 for r in primary) else
               "Neither learned method improves the mean primary endpoint over MSP; added training and feature processing are not supported by this endpoint."),
              "", "Methodological limits:",
              "Full-population metric definition: " + pooled["method"] + ".",
              "A documented smoke-test metric amendment refined score binning before full training; the original frozen protocol is preserved unchanged.",
              "Retrospective80 accepts a complete transformed-score bin, including all exactly tied scores. Actual method coverages can differ; its threshold is not deployable.",
              "Training-derived thresholds use final outer-training sample scores and may be optimistic. Their held-out coverage is measured exactly and is not guaranteed to be 80%.",
              "Pooled/weather histograms mix different fold-specific classifiers: cross-fitted diagnostics, not a single deployed classifier. Primary inference uses paired family summaries.",
              "Filename-derived recording groups are not proven independent sessions; route/camera/date dependence can remain. Night and rain have only three groups each.",
              "No pixel-IID significance tests or population confidence intervals are reported. Leave-one-family-out ranges are descriptive sensitivity, not confidence intervals.",
              "Standardization and fitting use corresponding training groups only. No ground-truth feature is used. Existing data were previously explored; this is not an untouched confirmatory test.",
              "This is a supervised error predictor, not a new segmentation network. Original segmentation labels and mIoU are not improved by scoring or rejection."]
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    save_json(output / "report_manifest.json", dict(status="complete", outer_groups=17, images=1600,
        source_hashes=sources, report_source_sha256=sha256(__file__),
        output_hashes={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "report_manifest.json"}))
    return output


if __name__ == "__main__":
    print(generate())
