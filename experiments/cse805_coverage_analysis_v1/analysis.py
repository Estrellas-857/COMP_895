"""Reconstruct paired, equal-budget risk from the archived outer-fold histograms."""
import csv
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.cse805_error_prediction_v1.evaluation import HISTOGRAM_BINS, HistogramAccumulator
from experiments.cse805_error_prediction_v1.protocol import read_csv
from experiments.cse805_error_prediction_v1.report import restore_accumulator
from experiments.cse805_class_msp_interaction_v1.report import matched_budget
from experiments.cse805_random_forest_v1.model import (
    BASE, DATA, INTERACTION, LABELS, LR_INTERACTION, METHODS, OUT as FOREST, RF, read, sha256,
)
from experiments.cse805_random_forest_v1.report import interpolated_curve

OUT = Path(__file__).resolve().parent
WEATHERS = {"fog": 6, "night": 3, "rain": 3, "snow": 5}
TARGETS = tuple(range(10, 100, 10))
TIE_PP = 1e-10  # Same numerical tolerance as the preceding paired RF comparison.
SNAPSHOT = BASE / "interpretation_v1/logs/frozen_extension_before.json"


def write_csv(name, rows):
    path = OUT / name
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def checked_digest(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen source checksum mismatch: {path}")
    return actual


def load_sources():
    """Verify models, split identities and image-level populations before using histograms."""
    frozen = read(SNAPSHOT)
    manifest = read(FOREST / "run_manifest.json")
    rf_receipt = read(FOREST / "results/verification.json")
    lr_receipt = read(INTERACTION / "results/verification.json")
    assert lr_receipt["status"] == rf_receipt["status"] == "PASS"
    checked_digest(FOREST / "run_manifest.json", rf_receipt["manifest_sha256"])
    checked_digest(INTERACTION / "run_manifest.json", lr_receipt["manifest_sha256"])
    checked_digest(INTERACTION / "results/verification.json", manifest["interaction_receipt_sha256"])
    for name, digest in manifest["frozen_sources"].items():
        checked_digest(BASE / name, digest)
    groups = sorted(manifest["outer_groups"])
    splits = {r["eval_group"]: r for r in read(DATA / "assignments.json")}
    metadata = read_csv(DATA / "metadata_train.csv")
    features = {r["image_id"]: r for r in read(DATA / "features/manifest.json")["records"]}
    assert len(groups) == len(splits) == 17 and set(groups) == set(splits)
    assert len(metadata) == len(features) == 1600 and set(features) == {r["image_id"] for r in metadata}
    weather, totals, sources = {}, {}, {}
    for group in groups:
        rows = [r for r in metadata if r["recording_group"] == group]
        ids = {r["image_id"] for r in rows}
        split = splits[group]
        assert len(ids) == len(rows) == split["eval_image_count"]
        assert set(split["train_groups"]) == set(groups) - {group}
        conditions = {r["condition"] for r in rows}
        assert len(conditions) == 1 and all(r["dataset_split"] == "train" for r in rows)
        weather[group] = conditions.pop()
        original = BASE / "results/folds" / group
        original_models = DATA / "models" / group
        sources[group] = {}
        for path in (original / "result.json", original / "image_results.json", original / "histograms.npz",
                     original_models / "selection.json"):
            digest = frozen[str(path.relative_to(BASE))]["sha256"]
            sources[group][str(path.relative_to(OUT.parent))] = checked_digest(path, digest)
        selection = read(original_models / "selection.json")
        assert selection["group"] == group and selection["train_groups"] == split["train_groups"]
        for method, digest in selection["model_hashes"].items():
            path = original_models / (method + ".joblib")
            assert digest == frozen[str(path.relative_to(BASE))]["sha256"]
            sources[group][str(path.relative_to(OUT.parent))] = checked_digest(path, digest)
        assert set(selection["model_hashes"]) == {"lr3", "lr3_class"}
        scored = read(original / "image_results.json")
        assert len(scored) == len(ids) and {r["image_id"] for r in scored} == ids
        for r in scored:
            cache = features[r["image_id"]]
            assert cache["recording_group"] == r["recording_group"] == group
            assert cache["condition"] == r["condition"] == weather[group] and set(r["methods"]) == set(METHODS[:4])
            assert all((m["num_pixels"], m["num_errors"]) == (cache["num_valid"], cache["num_errors"])
                       for m in r["methods"].values())
        totals[group] = tuple(sum(features[i][k] for i in ids) for k in ("num_valid", "num_errors"))
        for root, receipt, model, image_key in (
                (INTERACTION, lr_receipt, "interaction.joblib", "evaluated_image_ids"),
                (FOREST, rf_receipt, "forest.joblib", "images")):
            fold = root / "folds" / group
            for name in ("evaluation.json", "selection.json", "histograms.npz", model):
                path = fold / name
                digest = receipt["artifact_sha256"][str(path.relative_to(root))]
                sources[group][str(path.relative_to(OUT.parent))] = checked_digest(path, digest)
            ev, sel = read(fold / "evaluation.json"), read(fold / "selection.json")
            assert len(ev[image_key]) == len(ids) and set(ev[image_key]) == ids
            assert ev["weather"] == weather[group] and sel["group"] == ev["group"] == group
            assert sel["train_groups"] == split["train_groups"] and ev["manifest_sha256"] == receipt["manifest_sha256"]
            assert ev["model_sha256"] == sel["model_sha256"] == sources[group][str((fold / model).relative_to(OUT.parent))]
            assert ev["histogram_sha256"] == sources[group][str((fold / "histograms.npz").relative_to(OUT.parent))]
    assert {w: list(weather.values()).count(w) for w in WEATHERS} == WEATHERS
    assert {w: sum(r["condition"] == w for r in metadata) for w in WEATHERS} == dict.fromkeys(WEATHERS, 400)
    assert tuple(map(sum, zip(*totals.values()))) == (rf_receipt["full_valid_pixels"], rf_receipt["segmentation_errors"])
    return weather, totals, sources


def histograms(group):
    original = BASE / "results/folds" / group
    lr = INTERACTION / "folds" / group
    rf = FOREST / "folds" / group
    metrics = read(original / "result.json")["methods"]
    metrics.update({LR_INTERACTION: read(lr / "evaluation.json")["metrics"], RF: read(rf / "evaluation.json")["metrics"]})
    with (np.load(original / "histograms.npz", allow_pickle=False) as h1,
          np.load(lr / "histograms.npz", allow_pickle=False) as h2,
          np.load(rf / "histograms.npz", allow_pickle=False) as h3):
        for m in METHODS:
            h, prefix = (h2, "") if m == LR_INTERACTION else (h3, "") if m == RF else (h1, m + "_")
            correct, errors = h[prefix + "correct"], h[prefix + "incorrect"]
            assert correct.shape == errors.shape == (HISTOGRAM_BINS,)
            assert all(np.issubdtype(a.dtype, np.integer) and a.min() >= 0 for a in (correct, errors))
            yield m, restore_accumulator(metrics[m], correct, errors), metrics[m]


def measurements(weather, totals):
    # At the first target each group accepts exactly one pixel. Risk at zero is undefined.
    grid = np.r_[1 / max(n for n, _ in totals.values()), 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4,
                 np.arange(1, 1001) / 1000]
    assert np.all(np.diff(grid) > 0)
    target_indices = np.searchsorted(grid, np.asarray(TARGETS) / 100)
    np.testing.assert_array_equal(grid[target_indices], np.asarray(TARGETS) / 100)
    rows, curves = [], {}
    for g, w in weather.items():
        n, e = totals[g]
        for m, acc, metrics in histograms(g):
            assert (metrics["num_pixels"], metrics["num_errors"]) == (n, e)
            curves[g, m] = interpolated_curve(acc, grid)
            assert curves[g, m][-1] == e / n
            coarse = HistogramAccumulator(bins=HISTOGRAM_BINS // 10)
            coarse.correct[:] = acc.correct.reshape(-1, 10).sum(axis=1)
            coarse.incorrect[:] = acc.incorrect.reshape(-1, 10).sum(axis=1)
            coarse_risk = interpolated_curve(coarse, grid[target_indices])
            whole = acc.finalize(grid[target_indices])["checkpoints"]
            counts = np.cumsum(acc.correct + acc.incorrect)
            wrong = np.cumsum(acc.incorrect)
            for j, pct in enumerate(TARGETS):
                point = matched_budget(acc, pct / 100)
                k, b = point["matched_accepted_pixels"], point["boundary_bin"]
                before = int(counts[b] - point["boundary_pixels"])
                before_err = int(wrong[b] - acc.incorrect[b])
                assert abs(k - pct / 100 * n) <= .5000001
                assert np.isclose(curves[g, m][target_indices[j]], point["matched_risk"], rtol=1e-14, atol=1e-15)
                near = whole[j]
                rows.append(dict(weather=w, group=g, method=m, target_coverage_pct=pct,
                    valid_pixels=n, total_errors=e, accepted_pixels=k, actual_coverage=k / n,
                    expected_accepted_errors=point["matched_expected_errors"], accepted_error_rate=point["matched_risk"],
                    boundary_bin=b, pixels_below_bin=before, errors_below_bin=before_err,
                    boundary_pixels=point["boundary_pixels"], boundary_errors=int(acc.incorrect[b]),
                    boundary_pixels_accepted=k-before, boundary_included_fraction=point["boundary_included_fraction"],
                    nearest_whole_bin_accepted=near["accepted_pixels"], nearest_whole_bin_errors=near["accepted_errors"],
                    nearest_whole_bin_coverage=near["actual_coverage"], nearest_whole_bin_risk=near["risk"],
                    matched_risk_10000_bins=coarse_risk[j], policy="full_histogram_100000_uniform_boundary_bin"))
    return rows, grid, curves


def aggregate(rows, weather):
    values = {(r["group"], r["method"], r["target_coverage_pct"]): r["accepted_error_rate"] for r in rows}
    assert len(values) == len(rows) == 17 * 6 * 9
    scopes = {"overall": list(weather), **{w: [g for g in weather if weather[g] == w] for w in WEATHERS}}
    overall, by_weather, paired = [], [], []
    for scope, groups in scopes.items():
        for p in TARGETS:
            rates = {m: mean(values[g, m, p] for g in groups) for m in METHODS}
            if scope == "overall":
                overall.append(dict(target_coverage_pct=p, n_groups=len(groups), **rates))
            else:
                by_weather.append(dict(weather=scope, target_coverage_pct=p, n_groups=len(groups), **rates))
            for m in METHODS[:-1]:
                d = {g: 100 * (values[g, RF, p] - values[g, m, p]) for g in groups}
                wins = [g for g in groups if d[g] < -TIE_PP]
                losses = [g for g in groups if d[g] > TIE_PP]
                ties = [g for g in groups if abs(d[g]) <= TIE_PP]
                paired.append(dict(scope=scope, target_coverage_pct=p, reference=m, n_groups=len(groups),
                    rf_mean_error_rate=rates[RF], reference_mean_error_rate=rates[m], rf_minus_reference_pp=mean(d.values()),
                    wins=len(wins), ties=len(ties), losses=len(losses), win_groups="|".join(wins),
                    tie_groups="|".join(ties), loss_groups="|".join(losses)))
    return overall, by_weather, paired


def main():
    from .figures import make_figures
    from .report_zh import write_report
    weather, totals, _ = load_sources()
    rows, grid, curves = measurements(weather, totals)
    overall, by_weather, paired = aggregate(rows, weather)
    write_csv("coverage_overall.csv", overall)
    write_csv("coverage_by_weather.csv", by_weather)
    write_csv("coverage_by_group.csv", rows)
    write_csv("rf_paired_differences.csv", paired)
    make_figures(grid, curves, weather)
    write_report()
    print(f"Recovered {len(rows)} primary-budget rows and {len(paired)} paired contrasts from 51 histogram files.")


if __name__ == "__main__":
    main()
