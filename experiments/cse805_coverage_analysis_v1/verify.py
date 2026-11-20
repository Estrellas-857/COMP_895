"""Verify the exported CSVs against integer histogram counts and frozen experiment receipts."""
from datetime import datetime, timezone
from fractions import Fraction
from itertools import groupby
import json
from statistics import mean

import matplotlib
import numpy as np
from PIL import Image

from experiments.cse805_class_msp_interaction_v1.verify import verify_snapshot
from experiments.cse805_error_prediction_v1.common import json_safe
from .analysis import (
    BASE, DATA, FOREST, INTERACTION, LR_INTERACTION, METHODS, OUT, RF, SNAPSHOT, TARGETS, TIE_PP, WEATHERS,
    checked_digest, histograms, load_sources, read, read_csv, sha256,
)


def close(a, b, tol=1e-13):
    if abs(float(a) - float(b)) > tol or not np.isfinite([float(a), float(b)]).all():
        raise AssertionError((a, b, tol))


def direction(pp):
    return "win" if pp < -TIE_PP else "loss" if pp > TIE_PP else "tie"


def winning_runs(x, d):
    result = []
    for lower, positions in groupby(range(len(x)), key=lambda i: d[i] < -TIE_PP):
        positions = list(positions)
        if lower:
            result.append([round(float(x[i])*100, 1) for i in (positions[0], positions[-1])])
    return result


def raw_qa(weather):
    """Read the existing Night raw-score, label-independent secondary-order sample checks."""
    qa = {}
    for g, w in weather.items():
        if w != "night":
            continue
        primary = read(BASE / "results/folds" / g / "result.json")["qa"]
        checkpoints = {m: v["exact"]["checkpoints"] for m, v in primary.items()}
        for root, m in ((INTERACTION, LR_INTERACTION), (FOREST, RF)):
            checkpoints[m] = read(root / "folds" / g / "evaluation.json")["qa_raw_secondary"]
        for m, points in checkpoints.items():
            for p in points:
                target = round(p["requested_coverage"] * 100)
                if target in TARGETS:
                    qa[g, m, target] = p["risk"]
    for p in read_csv(FOREST / "results/night_low_coverage.csv"):
        target = round(float(p["requested_coverage"]) * 100)
        if target in TARGETS:
            key = p["group"], p["method"], target
            if key in qa:
                close(qa[key], p["qa_raw_risk"])
            qa[key] = float(p["qa_raw_risk"])
    return qa


def main():
    weather, totals, sources = load_sources()
    report_rows = read_csv(OUT / "coverage_by_group.csv")
    row = {(r["group"], r["method"], int(r["target_coverage_pct"])): r for r in report_rows}
    csvs = ("coverage_overall.csv", "coverage_by_weather.csv", "coverage_by_group.csv", "rf_paired_differences.csv")
    scopes = {"overall": list(weather), **{w: [g for g in weather if weather[g] == w] for w in WEATHERS}}
    all_wide = read_csv(OUT / csvs[0]) + read_csv(OUT / csvs[1])
    wide = {(r.get("weather", "overall"), int(r["target_coverage_pct"])): r for r in all_wide}
    prs = read_csv(OUT / csvs[3])
    paired = {(r["scope"], r["reference"], int(r["target_coverage_pct"])): r for r in prs}
    assert len(report_rows) == len(row) == 918 and len(all_wide) == len(wide) == 45 and len(prs) == len(paired) == 225
    archive = read_csv(FOREST / "results/risk_coverage.csv")
    saved = {(r["group"], r["method"], round(100 * float(r["target"]))): r for r in archive}
    prior = {(r["identity"], r["method"]): r for r in read_csv(FOREST / "results/group_metrics.csv")}
    assert len(saved) == len(archive) == 10200 and len(prior) == 102
    low = {(r["group"], r["method"], round(100 * float(r["requested_coverage"]))): r
           for r in read_csv(FOREST / "results/night_low_coverage.csv")}
    grid = np.arange(100, 901) / 1000
    rates, coarse_rates, curves = {}, {}, {}
    max_archive = max_coverage_error = max_80_shift = max_80_coverage_gap = 0.
    rounding, historical_count_matches, low_count_matches = set(), 0, 0
    for g, (n, e) in totals.items():
        for m, acc, metrics in histograms(g):
            h = acc.correct + acc.incorrect
            occupied = np.flatnonzero(h)
            cum = np.r_[0, np.cumsum(h[occupied])]
            err = np.r_[0, np.cumsum(acc.incorrect[occupied])]
            assert (cum[-1], err[-1]) == (n, e)
            old = prior[g, m]
            assert (int(old["num_valid_pixels"]), int(old["num_error_pixels"])) == totals[g]
            # Independent reconstruction: interpolate cumulative error counts, then divide by integer budget.
            hundredths = np.arange(1, 101) / 100
            k100 = np.floor(hundredths*n + .5).astype(np.int64)
            r100 = np.interp(k100, cum, err) / k100
            archived = np.array([float(saved[g, m, p]["matched_risk"]) for p in range(1, 101)])
            max_archive = max(max_archive, float(np.max(abs(archived - r100))))
            np.testing.assert_allclose(r100, archived, rtol=0, atol=2e-15)
            close(r100[-1], e / n)
            kg = np.floor(grid*n + .5).astype(np.int64)
            curves[g, m] = 100 * np.interp(kg, cum, err) / kg
            coarse_mass = np.r_[0, h.reshape(-1, 10).sum(axis=1).cumsum()]
            coarse_err = np.r_[0, acc.incorrect.reshape(-1, 10).sum(axis=1).cumsum()]
            used = np.r_[True, np.diff(coarse_mass) > 0]
            for p in TARGETS:
                a = row[g, m, p]
                assert a["weather"] == weather[g] and (int(a["valid_pixels"]), int(a["total_errors"])) == totals[g]
                k = int(k100[p-1])
                assert int(a["accepted_pixels"]) == k and a["policy"] == "full_histogram_100000_uniform_boundary_bin"
                close(a["actual_coverage"], k/n)
                max_coverage_error = max(max_coverage_error, abs(k/n - p/100) * 100)
                if k != (2*p*n + 100)//200:
                    rounding.add((g, p, k, (2*p*n + 100)//200))
                j = int(cum.searchsorted(k))
                below, before_e = int(cum[j-1]), int(err[j-1])
                b, be = int(cum[j]-below), int(err[j]-before_e)
                expected = before_e + Fraction((k-below)*be, b)
                for field, v in (("boundary_bin", occupied[j-1]), ("pixels_below_bin", below), ("errors_below_bin", before_e),
                                 ("boundary_pixels", b), ("boundary_errors", be), ("boundary_pixels_accepted", k-below)):
                    assert int(a[field]) == v
                close(a["boundary_included_fraction"], (k-below)/b)
                close(a["expected_accepted_errors"], expected, tol=8e-8)
                rate = float(expected/k)
                close(rate, a["accepted_error_rate"])
                close(rate, archived[p-1])
                coarse = float(np.interp(k, coarse_mass[used], coarse_err[used]) / k)
                close(coarse, a["matched_risk_10000_bins"])
                rates[g, m, p], coarse_rates[g, m, p] = rate, coarse
                wi = 1+int(np.argmin(abs(cum[1:]/n - p/100)))
                assert (int(a["nearest_whole_bin_accepted"]), int(a["nearest_whole_bin_errors"])) == (cum[wi], err[wi])
                close(a["nearest_whole_bin_risk"], err[wi]/cum[wi])
                close(a["nearest_whole_bin_coverage"], cum[wi]/n)
                close(a["nearest_whole_bin_risk"], saved[g, m, p]["whole_bin_risk"])
                close(a["nearest_whole_bin_coverage"], saved[g, m, p]["actual_coverage"])
                if (g, m, p) in low:
                    prev = low[g, m, p]
                    assert int(prev["matched_accepted_pixels"]) == k
                    close(prev["matched_expected_errors"], expected, tol=8e-8)
                    low_count_matches += 1
                if p == 80:
                    primary = next(i for i in metrics["checkpoints"] if i["requested_coverage"] == .8)
                    assert (cum[wi], err[wi]) == (primary["accepted_pixels"], primary["accepted_errors"])
                    close(primary["risk"], old["retrospective80_risk"])
                    close(primary["actual_coverage"], old["retrospective80_actual_coverage"])
                    close(a["nearest_whole_bin_risk"], primary["risk"])
                    close(a["accepted_error_rate"], old["matched80_risk"])
                    max_80_shift = max(max_80_shift, 100*abs(primary["risk"]-rate))
                    max_80_coverage_gap = max(max_80_coverage_gap, 100*abs(primary["actual_coverage"]-k/n))
                    historical_count_matches += 1
    assert (historical_count_matches, low_count_matches) == (102, 36)
    qa = raw_qa(weather)
    archived_night_targets = sorted({k[2] for k in qa})
    assert archived_night_targets == [10, 20, 50, 60, 70, 80, 90] and len(qa) == 3*6*7
    original_macro = {(r["scope"], r["method"]):r for r in read_csv(FOREST / "results/final_comparison.csv")}
    historical, runs, coarse_diagnostics, night_qa = {}, {}, {}, []
    for scope, groups in scopes.items():
        historical[scope], runs[scope] = {}, {}
        for m in METHODS:
            historic = mean(float(row[g, m, 80]["nearest_whole_bin_risk"]) for g in groups)
            exact = mean(rates[g, m, 80] for g in groups)
            archived_row = original_macro["all" if scope == "overall" else scope, m]
            close(historic, archived_row["retrospective80_risk"])
            close(exact, archived_row["matched80_risk"])
            historical[scope][m] = dict(historical_risk=historic, matched_risk=exact, matched_minus_historical_pp=100*(exact-historic))
            for p in TARGETS:
                assert int(wide[scope, p]["n_groups"]) == len(groups)
                vals = [rates[g, m, p] for g in groups]
                close(wide[scope, p][m], mean(vals))
                if m == RF:
                    continue
                d = [100*(rates[g, RF, p]-v) for g, v in zip(groups, vals)]
                cw = mean(100*(coarse_rates[g, RF, p]-coarse_rates[g, m, p]) for g in groups)
                pr = paired[scope, m, p]
                close(pr["rf_mean_error_rate"], wide[scope, p][RF])
                close(pr["reference_mean_error_rate"], wide[scope, p][m])
                close(pr["rf_minus_reference_pp"], mean(d))
                assert int(pr["n_groups"]) == len(groups)
                for field, col in (("win", "wins"), ("tie", "ties"), ("loss", "losses")):
                    members = set(filter(None, pr[field+"_groups"].split("|")))
                    check = {g for g, x in zip(groups, d) if direction(x) == field}
                    assert members == check and len(check) == int(pr[col])
                stat = coarse_diagnostics.setdefault(scope, {}).setdefault(m, dict(max_abs_contrast_change_pp=0., changed_direction_at_pct=[]))
                stat["max_abs_contrast_change_pp"] = max(stat["max_abs_contrast_change_pp"], abs(mean(d)-cw))
                if direction(mean(d)) != direction(cw):
                    stat["changed_direction_at_pct"].append(p)
                if scope == "night" and p in archived_night_targets:
                    nd = mean(100*(qa[g, RF, p]-qa[g, m, p]) for g in groups)
                    night_qa.append(dict(target_coverage_pct=p, reference=m, full_histogram_difference_pp=mean(d),
                                         raw_QA_secondary_order_difference_pp=nd))
            if m != RF:
                dgrid = np.mean([curves[g, RF]-curves[g, m] for g in groups], axis=0)
                runs[scope][m] = winning_runs(grid, dgrid)
    for g in scopes["night"]:
        runs[g] = {m: winning_runs(grid, curves[g, RF]-curves[g, m]) for m in METHODS[:-1]}
    for p in TARGETS:
        for m in METHODS:
            close(wide["overall", p][m], sum(WEATHERS[w]*float(wide[w, p][m]) for w in WEATHERS)/17)
    preserved = {
        "COMP895_original_files": verify_snapshot(BASE.parents[1], read(BASE / "logs/preservation_before.json")),
        "CSE805_primary_files": verify_snapshot(BASE, read(SNAPSHOT)),
        "CSE805_interpretation_files": verify_snapshot(BASE / "interpretation_v1", read(BASE / "interpretation_v1/logs/completion.json")["artifacts"]),
    }
    for root in (INTERACTION, FOREST):
        receipt = read(root / "results/verification.json")
        paths = {**receipt["artifact_sha256"], **receipt["source_sha256"], "run_manifest.json": receipt["manifest_sha256"]}
        for name, digest in paths.items():
            checked_digest(root/name, digest)
        preserved[root.name+"_archive_files"] = len(paths)
    assert sum(t[0] for t in totals.values()) == 3059663748 and sum(t[1] for t in totals.values()) == 523653212
    artifacts = [OUT/name for name in (*csvs, "coverage_analysis_report_zh.txt")]
    sizes = {}
    for name in ("overall_risk_coverage", "weather_risk_coverage", "rf_vs_msp", "night_group_coverage"):
        path = OUT / "figures" / (name+".png")
        with Image.open(path) as im:
            assert im.format == "PNG" and im.width >= 2500 and im.height >= 1400
            sizes[name] = im.size
            im.verify()
        artifacts.append(path)
    provenance = [SNAPSHOT, BASE/"logs/preservation_before.json", BASE/"interpretation_v1/logs/completion.json",
                  BASE/"results/evaluation_complete.json", INTERACTION/"results/verification.json", FOREST/"results/verification.json",
                  INTERACTION/"run_manifest.json", FOREST/"run_manifest.json", FOREST/"results/risk_coverage.csv",
                  FOREST/"results/group_metrics.csv", FOREST/"results/final_comparison.csv", FOREST/"results/night_low_coverage.csv",
                  FOREST/"results/night_qa10_classes.csv", FOREST/"results/qa_aurc_sensitivity.csv",
                  *[BASE / f for f in read(FOREST/"run_manifest.json")["frozen_sources"]]]
    result = dict(status="PASS", verified_utc=datetime.now(timezone.utc).isoformat(), scope="supplemental fixed-budget analysis",
        training_runs=0, model_scoring_runs=0, segmentation_forward_passes=0, histograms_npz=51, method_group_histogram_pairs=102,
        full_valid_pixels=3059663748, full_segmentation_errors=523653212, heldout_images=1600, excluded_original_validation_images=406,
        images_per_weather=400, group_counts=WEATHERS, methods=list(METHODS), targets_pct=list(TARGETS),
        population_checks="Fold image IDs, per-image valid/error counts, model selection split and model digests match frozen archives.",
        data_source_paths_relative_to="experiments/", fold_source_sha256=sources, archive_files_verified=preserved,
        source_manifest_and_reference_sha256={str(p.relative_to(OUT.parent)):sha256(p) for p in provenance},
        policy=dict(score="ascending predicted error; 100000 equally spaced arcsine-square-root score bins",
                    budget="Original float64 floor((p/100)*N+0.5), clamped to [1,N]. Budget identical across all six methods within each group.",
                    rounding_tie_cases_vs_integer_half_up=sorted(rounding),
                    expected_errors="errors_below_bin + boundary_pixels_accepted * boundary_errors / boundary_pixels",
                    within_boundary_bin="Uniform selection without replacement, independent of labels. Report expectation, not observed tie order.",
                    aggregation="Arithmetic mean of within-group accepted error rates. Overall weights each of 17 groups equally.",
                    paired_units="RF minus reference in percentage points. Win below -tolerance, loss above +tolerance.", tie_tolerance_pp=TIE_PP),
        verified_csv_rows=dict(overall=9, weather=36, group=len(row), paired=len(paired)),
        independent_exact_rational_boundary_count_checks=len(row), archived_full_percent_grid_matches=len(saved),
        max_absolute_rate_discrepancy_to_archived_grid=max_archive, max_matched_coverage_rounding_error_pp=max_coverage_error,
        historical_whole_bin80_counts_matches=historical_count_matches, existing_night10_20_matched_counts_matches=low_count_matches,
        max_group_method_historical80_vs_matched_abs_risk_difference_pp=max_80_shift,
        max_group_method_historical80_vs_matched_abs_coverage_difference_pp=max_80_coverage_gap,
        historical80_and_supplemental80=historical,
        fine_grid="Direct histogram queries from 10.0 to 90.0% every 0.1 percentage point. Range endpoints are measured grid values; no between-point sign claim.",
        fine_grid_RF_better_runs_pct=runs,
        sensitivity_full_population_10000_vs_100000_bins=coarse_diagnostics,
        sensitivity_Night_QA="Archived raw scores, weighted QA sample with independent tie keys. Secondary evidence only.",
        sensitivity_Night_QA_available_pct=archived_night_targets, sensitivity_Night_QA_comparisons=night_qa,
        plot_grid="One pixel per group at first target; additional small positive budgets, then every 0.1 percentage point to 100%. No risk at zero.",
        png_dimensions=sizes, versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
        source_sha256={p.name:sha256(p) for p in OUT.glob("*.py")},
        output_sha256={str(p.relative_to(OUT)):sha256(p) for p in artifacts})
    tmp = OUT/"verification.tmp"
    tmp.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(OUT/"verification.json")
    print("PASS: 918 rational boundary counts, 225 paired rows, 10,200 archived grid points and 102 historical 80% endpoints.")
    print("Frozen file checks:", preserved)


if __name__ == "__main__":
    main()
