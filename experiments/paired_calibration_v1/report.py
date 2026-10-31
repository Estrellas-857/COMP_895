"""Summarize completed paired-calibration outputs without reading source data.

This module never fits parameters, selects thresholds, or opens dataset images.
Derived reports are written only when absent or byte-for-byte identical.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path


TEMPERATURE_ORDER = ("raw", "global", "NightPair", "NormalPair", "Mix")
DOMAINS = ("normal", "night")
SEQUENCES = ("GOPR0351", "GOPR0356")
COLORS = {"raw": "#606a73", "global": "#0072b2", "NightPair": "#d55e00",
          "NormalPair": "#009e73", "Mix": "#8b58a5"}
CLASS_NAMES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
               "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
               "truck", "bus", "train", "motorcycle", "bicycle")


def _rows(directory, name):
    path = directory / name
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _value(row, key):
    return _number(row.get(key)) if row else None


def _format(value, *, percent=False, digits=4):
    value = _number(value)
    if value is None:
        return "NA"
    return f"{100 * value if percent else value:.{digits}f}"


def _lookup(rows, domain, temperature_name, sequence_id="pooled"):
    matches = [r for r in rows if r.get("domain") == domain
               and r.get("temperature_name") == temperature_name
               and r.get("sequence_id", "pooled") in
               (("pooled", "all", "") if sequence_id == "pooled" else (sequence_id,))]
    if len(matches) > 1:
        raise ValueError(f"Duplicate metric rows: {domain}/{temperature_name}/{sequence_id}")
    return matches[0] if matches else None


def _difference(rows, domain, candidate, reference, metric, sequence_id="pooled"):
    a = _value(_lookup(rows, domain, candidate, sequence_id), metric)
    b = _value(_lookup(rows, domain, reference, sequence_id), metric)
    return a - b if a is not None and b is not None else None


def _change(value, *, percent=False):
    value = _number(value)
    if value is None:
        return "not available"
    unit = " percentage points" if percent else " nats"
    direction = "decreases" if value < 0 else "increases" if value > 0 else "is unchanged by"
    return f"{direction} {_format(abs(value), percent=percent, digits=6)}{unit}"


def _write_once_or_identical(path, content):
    data = content.encode("utf-8") if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite a different report: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def _table(headers, rows):
    strings = [[str(value) for value in row] for row in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in strings)) if strings else len(str(h))
              for i, h in enumerate(headers)]
    render = lambda row: " | ".join(str(v).ljust(w) for v, w in zip(row, widths))
    return "\n".join([render(headers), "-+-".join("-" * w for w in widths),
                      *(render(row) for row in strings)])


def _sequence_comparison(rows, domain, candidate, reference):
    values = {sequence: _difference(rows, domain, candidate, reference, "nll", sequence)
              for sequence in SEQUENCES}
    available = [value for value in values.values() if value is not None]
    if len(available) != len(SEQUENCES):
        conclusion = "Both sequence comparisons are not available."
    elif all(value < 0 for value in available):
        conclusion = "NLL is lower on both available sequences."
    elif all(value > 0 for value in available):
        conclusion = "NLL is higher on both available sequences."
    elif all(value == 0 for value in available):
        conclusion = "NLL is unchanged on both available sequences."
    else:
        conclusion = "The NLL direction is not consistent across the two sequences."
    text = "; ".join(f"{key}: delta NLL={_format(value, digits=6)}" for key, value in values.items())
    return values, f"{text}. {conclusion}"


def _checkpoint(rows, domain, temperature_name, requested=0.8):
    eligible = [r for r in rows if r.get("domain") == domain
                and r.get("temperature_name") == temperature_name
                and r.get("sequence_id", "pooled") in ("pooled", "all", "")
                and _value(r, "requested_coverage") is not None
                and abs(float(r["requested_coverage"]) - requested) < 1e-8]
    if len(eligible) > 1:
        raise ValueError(f"Duplicate matched-coverage rows: {domain}/{temperature_name}/{requested}")
    return eligible[0] if eligible else None


def _matched_statement(rows, domain, candidate, reference):
    a, b = _checkpoint(rows, domain, candidate), _checkpoint(rows, domain, reference)
    if a is None or b is None:
        return "The requested 80% coverage comparison is unavailable."
    av, bv, ar, br = (_value(a, "actual_coverage"), _value(b, "actual_coverage"),
                       _value(a, "risk"), _value(b, "risk"))
    if None in (av, bv, ar, br):
        return "The requested 80% coverage comparison contains undefined metrics."
    return (f"At requested 80% coverage on {domain}, {candidate} versus {reference}: "
            f"actual coverage {_format(av, percent=True)}% versus {_format(bv, percent=True)}%; "
            f"estimated accepted risk {_format(ar, percent=True)}% versus {_format(br, percent=True)}% "
            f"(delta {(ar-br)*100:+.4f} percentage points). "
            "Interpret any actual-coverage mismatch alongside this risk difference.")


def _load_status(directory, blocked_reason):
    if blocked_reason:
        return str(blocked_reason)
    for name in ("run_status.json", "status.json", "blocked.json", "resource_summary.json"):
        path = directory / name
        if path.exists():
            record = json.loads(path.read_text())
            for key in ("blocked_reason", "reason", "error"):
                if record.get(key):
                    return str(record[key])
    return "Required evaluation tables are absent; no numerical conclusion can be drawn."


def _paired_image_summary(rows, names):
    """Equal-image differences, deliberately distinct from pooled pixel metrics."""
    grouped = {}
    for row in rows:
        pair = row.get("pair_id")
        domain, name = row.get("domain"), row.get("temperature_name")
        if not pair or domain not in DOMAINS:
            continue
        key = (name, pair)
        entry = grouped.setdefault(key, {})
        if domain in entry:
            raise ValueError(f"Duplicate per-image metric: {key}/{domain}")
        entry[domain] = row
    results = []
    for name in names:
        for sequence in ("pooled", *SEQUENCES):
            pairs = [value for (temp, _), value in grouped.items() if temp == name and set(value) == set(DOMAINS)
                     and (sequence == "pooled" or all(v.get("sequence_id") == sequence for v in value.values()))]
            if not pairs:
                continue
            result = {"temperature_name": name, "sequence_id": sequence, "pairs": len(pairs)}
            for metric in ("nll", "ece", "signed_gap", "coverage_valid", "accepted_risk"):
                differences = [float(p["night"][metric]) - float(p["normal"][metric]) for p in pairs
                               if all(_value(p[domain], metric) is not None for domain in DOMAINS)]
                result["mean_night_minus_normal_" + metric] = sum(differences) / len(differences) if differences else None
                result[metric + "_defined_pairs"] = len(differences)
            results.append(result)
    return results


def _class_failure_table(rows):
    table = []
    for domain in DOMAINS:
        candidates = [r for r in rows if r.get("domain") == domain
                      and r.get("temperature_name") == "global"
                      and r.get("sequence_id", "pooled") in ("pooled", "all", "")
                      and _value(r, "pred_num_high_conf_errors") is not None]
        for row in sorted(candidates, key=lambda r: -float(r["pred_num_high_conf_errors"]))[:3]:
            class_id = int(row["class_id"])
            table.append((domain, CLASS_NAMES[class_id], row["pred_num_high_conf_errors"],
                          row.get("pred_num_high_conf", "NA"),
                          _format(row.get("pred_high_conf_error_rate"), percent=True)))
    return table


def _plot_calibration(directory, rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")
    x = np.arange(len(names))
    for ax, metric, label, multiplier in ((axes[0], "nll", "NLL (nats)", 1),
                                          (axes[1], "ece", "15-bin ECE (%)", 100)):
        for offset, domain, color in ((-.18, "normal", "#0072b2"), (.18, "night", "#d55e00")):
            values = [_value(_lookup(rows, domain, name), metric) for name in names]
            ax.bar(x + offset, [v * multiplier if v is not None else np.nan for v in values],
                   width=.35, label=domain.title(), color=color)
        ax.set_xticks(x, names, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    fig.suptitle("Paired validation: each image evaluated against its own labels")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    _write_once_or_identical(directory / "calibration_comparison.png", buffer.getvalue())


def _plot_curves(directory, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths = list(directory.rglob("*.npz"))
    found = False
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")
    for ax, domain in zip(axes, DOMAINS):
        for name in names:
            candidates = [p for p in paths if p.name in
                          (f"{domain}_{name}_pooled.npz", f"{domain}_{name}_all.npz")]
            if len(candidates) > 1:
                raise ValueError(f"Multiple pooled curve files for {domain}/{name}")
            if not candidates:
                continue
            with np.load(candidates[0], allow_pickle=False) as curve:
                coverage, risk = curve["coverage"], curve["risk"]
                if coverage.ndim != 1 or risk.shape != coverage.shape:
                    raise ValueError(f"Invalid risk-coverage shapes: {candidates[0]}")
                ax.plot(coverage * 100, risk * 100, label=name, color=COLORS.get(name), linewidth=1.5)
            found = True
        ax.set(title=domain.title(), xlabel="Valid-pixel coverage (%)", ylabel="Accepted-pixel error (%)",
               xlim=(0, 100), ylim=(0, None))
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Estimated risk–coverage: identical sampled pixels across temperatures")
    if found:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=180)
        _write_once_or_identical(directory / "risk_coverage.png", buffer.getvalue())
    plt.close(fig)
    return found


def generate_report(out_dir, blocked_reason=None, *, make_plots=True):
    """Create findings.txt, summary.json and plots from finalized experiment files.

    Missing evaluation results produce an explicit incomplete report, never
    substituted baseline values. Call once evaluation/provenance checks finish.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    calibration = _rows(directory, "calibration_comparison.csv")
    policy = _rows(directory, "frozen_policy_comparison.csv")
    sequences = _rows(directory, "sequence_metrics.csv")
    matched = _rows(directory, "matched_coverage.csv")
    aurc = _rows(directory, "aurc.csv")
    per_image = _rows(directory, "per_image_metrics.csv")
    class_rows = _rows(directory, "class_metrics.csv")
    ambiguity = _rows(directory, "ambiguity_metrics.csv")
    summary = {"status": "incomplete", "source": "finalized experiment outputs only",
               "limitations": ["Validation was previously explored.",
                               "Only two Night validation sequence IDs are available; pixels are not independent observations.",
                               "Corresponding Normal and Night images are not pixel aligned.",
                               "Risk-coverage/AURC are shared-sample estimates, not full-population exact ranks.",
                               "All 19 classes are descriptive; no class-specific policy was fitted."]}
    if blocked_reason or not calibration:
        reason = _load_status(directory, blocked_reason)
        summary["blocked_reason"] = reason
        text = ("PAIRED CALIBRATION V1 — EVALUATION NOT COMPLETED\n\n" + reason + "\n\n"
                "No new numerical calibration, transfer, or selective-risk conclusions are available.\n"
                "The seven research questions remain unanswered by this run. Existing frozen results do not substitute "
                "for the requested paired evaluation.\n")
        _write_once_or_identical(directory / "findings.txt", text)
        _write_once_or_identical(directory / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary

    present = {r.get("temperature_name") for r in calibration}
    names = [name for name in TEMPERATURE_ORDER if name in present]
    names += sorted(present - set(TEMPERATURE_ORDER) - {None})
    required = ("raw", "global", "NightPair", "NormalPair")
    missing = [f"{domain}/{name}" for domain in DOMAINS for name in required
               if _lookup(calibration, domain, name) is None]
    summary["status"] = "complete" if not missing and policy and matched and aurc and sequences else "partial"
    summary["missing_calibration_rows"] = missing
    questions = {}
    comparison_deltas = {}

    dn = _difference(calibration, "normal", "global", "raw", "nll")
    de = _difference(calibration, "normal", "global", "raw", "ece")
    comparison_deltas["normal_global_minus_raw"] = {"nll": dn, "ece": de}
    _, sequence_text = _sequence_comparison(sequences, "normal", "global", "raw")
    questions["1_global_to_normal"] = (f"On Normal, global versus raw NLL {_change(dn)}; ECE {_change(de, percent=True)}. "
                                        f"These two criteria are reported separately. {sequence_text}")

    dn = _difference(calibration, "night", "NightPair", "NormalPair", "nll")
    de = _difference(calibration, "night", "NightPair", "NormalPair", "ece")
    per_sequence, sequence_text = _sequence_comparison(sequences, "night", "NightPair", "NormalPair")
    comparison_deltas["night_NightPair_minus_NormalPair"] = {"nll": dn, "ece": de, "sequence_nll": per_sequence}
    questions["2_night_vs_normal_fit"] = (f"On Night, NightPair versus NormalPair NLL {_change(dn)}; ECE {_change(de, percent=True)}. "
                                           f"{sequence_text} This is descriptive consistency, not evidence from many independent sequences.")

    normal_delta = _difference(calibration, "normal", "NightPair", "NormalPair", "nll")
    night_delta = _difference(calibration, "night", "NightPair", "NormalPair", "nll")
    tradeoff = (normal_delta is not None and night_delta is not None and normal_delta * night_delta < 0)
    questions["3_domain_tradeoff"] = (f"Changing NormalPair to NightPair gives delta NLL {_format(normal_delta, digits=6)} on Normal "
                                      f"and {_format(night_delta, digits=6)} on Night. "
                                      + ("The domains favor different temperatures by pooled NLL." if tradeoff else
                                         "An opposing pooled-NLL trade-off is not established by these rows."))
    questions["4_calibration_and_rejection"] = (
        _matched_statement(matched, "night", "global", "raw") + " " +
        _matched_statement(matched, "night", "NightPair", "NormalPair") +
        " NLL/ECE changes measure calibration; ranking must be judged from the shared-sample curves and all coverage checkpoints. "
        "Fixed-threshold risk changes alone cannot establish better rejection.")

    high_conf_text = []
    for domain in DOMAINS:
        row = _lookup(calibration, domain, "global")
        if row:
            high_conf_text.append(
                f"{domain.title()} global-T: {row.get('num_high_conf_errors', 'NA')} wrong pixels among "
                f"{row.get('num_high_conf', 'NA')} pixels with confidence >=0.90; error rate within this group "
                f"{_format(_value(row, 'high_conf_error_rate'), percent=True)}%. "
                f"{_format(_value(row, 'error_high_conf_fraction'), percent=True)}% of all errors have confidence >=0.90.")
    questions["5_remaining_confident_errors"] = " ".join(high_conf_text) or "High-confidence error counts are unavailable."

    balanced = {}
    for name in names:
        values = [_value(_lookup(calibration, domain, name), "nll") for domain in DOMAINS]
        if all(value is not None for value in values):
            balanced[name] = sum(values) / 2
    if "Mix" in balanced:
        comparisons = "; ".join(f"versus {name}: delta balanced NLL={balanced['Mix']-balanced[name]:+.6f}"
                                 for name in names if name != "Mix" and name in balanced)
        questions["6_balanced_temperature"] = (
            f"The predefined equal-domain Mix has validation balanced NLL={balanced['Mix']:.6f}; {comparisons}. "
            "Inspect each domain and sequence in addition to this mean. No mixture weight was selected on validation; "
            "this is established scalar temperature scaling, not a new calibration algorithm.")
    else:
        questions["6_balanced_temperature"] = "The optional Mix result is unavailable; no balanced-temperature benefit is claimed."
    questions["7_class_policy_complexity"] = (
        "This experiment does not test a learned class-specific rejection policy. Descriptive per-class errors cannot establish "
        "that extra thresholds would help. Stop at the present scalar/paired study unless a separate training-only analysis "
        "identifies stable predicted-class error differences with adequate sequence support and a predeclared evaluation. "
        "True-sky errors predicted as building or vegetation are not addressed by a predicted-sky threshold. "
        "No additional class policy is recommended solely from these previously explored validation results.")
    summary.update(questions=questions, comparison_deltas=comparison_deltas,
                   balanced_validation_nll=balanced, temperature_names=names,
                   contribution="Empirical calibration-transfer and selective-reliability evidence; no new calibration algorithm.")
    paired_summary = _paired_image_summary(per_image, names)
    summary["paired_image_differences"] = paired_summary

    lines = ["PAIRED NORMAL–NIGHT CALIBRATION AND RELIABILITY", "",
             f"Report status: {summary['status']}. All numerical conclusions below come from the saved extension outputs.",
             "NLL is in nats. ECE/gap/coverage/risk tables use percentages or percentage points as labeled.",
             "Pooled results are valid-pixel weighted within each domain; they are not the mean paired-image difference.", "",
             "CALIBRATION — POOLED VALID PIXELS"]
    table = []
    for name in names:
        for domain in DOMAINS:
            row = _lookup(calibration, domain, name)
            if row:
                table.append((name, domain, _format(row.get("temperature"), digits=6), _format(row.get("nll"), digits=6),
                              _format(row.get("ece"), percent=True), _format(row.get("signed_gap"), percent=True)))
    lines += [_table(("Temperature", "Domain", "T", "NLL", "ECE (%)", "Conf.-acc. gap (pp)"), table), "",
              "FROZEN CUTOFF TRANSFER — COVERAGE MAY CHANGE WITH TEMPERATURE",
              "The original global-T/global-tau configuration is the frozen baseline. Other temperatures at the same tau, "
              "including raw T=1, are cutoff-transfer diagnostics; the raw cutoff was not selected as a separate baseline."]
    table = []
    for name in names:
        for domain in DOMAINS:
            row = _lookup(policy, domain, name)
            if row:
                table.append((name, domain, *(_format(row.get(key), percent=True) for key in
                                              ("coverage_valid", "coverage_all", "accepted_risk"))))
    lines += [_table(("Temperature", "Domain", "Valid coverage (%)", "All coverage (%)", "Accepted error (%)"), table), "",
              "SHARED-SAMPLE RISK — REQUESTED 80% COVERAGE",
              "All requested 50/60/70/80/90% checkpoints and actual-coverage errors are in matched_coverage.csv."]
    table = []
    for name in names:
        for domain in DOMAINS:
            row = _checkpoint(matched, domain, name)
            arow = _lookup(aurc, domain, name)
            if row:
                table.append((name, domain, _format(row.get("actual_coverage"), percent=True),
                              _format(row.get("risk"), percent=True), _format(_value(arow, "aurc"), digits=6)))
    lines += [_table(("Temperature", "Domain", "Actual coverage (%)", "Estimated risk (%)", "Estimated AURC"), table), "",
              "SEQUENCE-SPECIFIC CALIBRATION"]
    table = []
    for sequence in SEQUENCES:
        for domain in DOMAINS:
            for name in names:
                row = _lookup(sequences, domain, name, sequence)
                if row:
                    table.append((sequence, domain, name, _format(row.get("nll"), digits=6), _format(row.get("ece"), percent=True)))
    lines += [_table(("Sequence", "Domain", "Temperature", "NLL", "ECE (%)"), table), "",
              "PAIRED IMAGE DIFFERENCES",
              "paired_differences.csv reports matched image-level differences using each side's own GT; "
              "per_image_metrics.csv retains the underlying image-level values. These are distinct from pooled pixel-weighted results."]
    table = [(r["temperature_name"], r["pairs"], _format(r["mean_night_minus_normal_nll"], digits=6),
              _format(r["mean_night_minus_normal_ece"], percent=True),
              _format(r["mean_night_minus_normal_coverage_valid"], percent=True),
              _format(r["mean_night_minus_normal_accepted_risk"], percent=True))
             for r in paired_summary if r["sequence_id"] == "pooled"]
    lines += ["Equal-image mean Night minus Normal; undefined metric pairs are excluded for that metric "
              "(defined-pair counts are recorded in summary.json).",
              _table(("Temperature", "Pairs", "Delta NLL", "Delta ECE (pp)", "Delta valid coverage (pp)",
                      "Delta accepted error (pp)"), table), "", "RESEARCH QUESTIONS"]
    for index, answer in enumerate(questions.values(), 1):
        lines += [f"{index}. {answer}", ""]
    lines += ["FAILURE ANALYSIS AND RESOURCE RECORD",
              "class_metrics.csv contains all 19 predicted-class and GT-class descriptive statistics; classes were not selected for tuning.",
              "Largest global-T high-confidence-error counts by predicted class (descriptive reporting only):",
              _table(("Domain", "Predicted class", "High-confidence errors", "High-confidence pixels", "Error within group (%)"),
                     _class_failure_table(class_rows)),
              "Night ambiguity strata retain the semantic-valid label mask; this does not redefine the primary benchmark.",
              _table(("Region", "Temperature", "Semantic-valid pixels", "NLL", "Valid coverage (%)", "Accepted error (%)"),
                     [(r.get("region", "NA"), r.get("temperature_name", "NA"), r.get("num_valid", "NA"),
                       _format(r.get("nll"), digits=6), _format(r.get("coverage_valid"), percent=True),
                       _format(r.get("accepted_risk"), percent=True)) for r in ambiguity
                      if r.get("sequence_id", "pooled") in ("pooled", "all", "")]),
              "See ambiguity_metrics.csv for sequence-specific strata; an empty table means no stratum metrics were recorded.",
              "resource_summary.json records observed resource/runtime information. No historical timing is presented as a new measurement.", "",
              "LIMITATIONS AND CONTRIBUTION", *["- " + item for item in summary["limitations"]],
              "The contribution is paired empirical evidence on scalar calibration transfer and rejection behavior. "
              "It does not establish a new calibration algorithm, causal illumination effects, performance on combined Night+Rain/Snow, "
              "or statistically independent evidence from individual pixels.", ""]
    if summary["status"] != "complete":
        lines += ["Some required tables or reference rows are missing. Treat the available findings as partial.", ""]
    if make_plots:
        _plot_calibration(directory, calibration, names)
        summary["risk_coverage_plot_created"] = _plot_curves(directory, names)
    _write_once_or_identical(directory / "findings.txt", "\n".join(lines))
    _write_once_or_identical(directory / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--blocked-reason")
    args = parser.parse_args()
    result = generate_report(args.out_dir, args.blocked_reason)
    print(json.dumps({"status": result["status"], "output": str(args.out_dir)}))


if __name__ == "__main__":
    main()
