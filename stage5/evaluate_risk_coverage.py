"""Stage 5B descriptive 1000-bin risk–coverage analysis; no deployment retuning."""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from stage4.run_final_reliability import load_final_config
from stage4.reliability_pipeline import ReliabilityPipeline, METHODS
from stage2.evaluate_uncertainty import UncertaintyAccumulator, HIST_EDGES, NUM_BINS
from stage1.evaluate_condition import RGB_ROOT, find_gt_path, update_confusion_matrix
from stage5.evaluate_frozen_operating_point import InferenceBeforeGTAudit, sha256

from stage1._layout import matches_frozen_hash

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "results/stage5_final_evaluation/stage5b"
STAGE5A = PROJECT / "results/stage5_final_evaluation/stage5a"
EXPECTED = {"fog": 100, "rain": 100, "snow": 100, "night": 106}
TARGETS = (1.0, .9, .8, .7, .6, .5)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def risk_coverage(correct, incorrect, edges=HIST_EDGES):
    """Accept entire bins from low to high uncertainty; no fabricated risk at 0.

    Bin m is [edges[m],edges[m+1]), with the final bin including 1. Therefore
    cumulative point m accepts U < edges[m+1] (<=1 at the final bin).
    Empty-bin duplicate coverages are retained, with zero width in a plot.
    """
    correct, incorrect = np.asarray(correct), np.asarray(incorrect)
    assert correct.dtype == incorrect.dtype == np.int64
    assert correct.shape == incorrect.shape == (len(edges)-1,)
    assert np.all(correct >= 0) and np.all(incorrect >= 0)
    accepted_correct = np.cumsum(correct, dtype=np.int64)
    accepted_incorrect = np.cumsum(incorrect, dtype=np.int64)
    accepted = accepted_correct + accepted_incorrect
    total = int(accepted[-1])
    assert total > 0
    nonzero = accepted > 0
    coverage = accepted[nonzero] / total
    risk = accepted_incorrect[nonzero] / accepted[nonzero]
    assert coverage[-1] == 1.0
    assert risk[-1] == int(incorrect.sum())/total
    assert np.all(np.diff(coverage) >= 0) and np.all((risk >= 0) & (risk <= 1))
    return dict(bin_index=np.arange(len(correct), dtype=np.int64)[nonzero],
        threshold_upper=edges[1:][nonzero], coverage=coverage, risk=risk,
        accepted_correct=accepted_correct[nonzero], accepted_incorrect=accepted_incorrect[nonzero],
        accepted=accepted[nonzero])


def checkpoint(curve, target):
    # Nearest available coverage. Ties use the first ascending bin: lower
    # coverage, or the lowest threshold among duplicate-coverage points.
    i = int(np.argmin(np.abs(curve["coverage"] - target)))
    return dict(requested_coverage=target, actual_histogram_coverage=float(curve["coverage"][i]),
        coverage_error=float(curve["coverage"][i]-target), risk=float(curve["risk"][i]),
        uncertainty_threshold_bin=int(curve["bin_index"][i]),
        uncertainty_threshold_upper=float(curve["threshold_upper"][i]),
        threshold_upper_inclusive=int(curve["bin_index"][i]) == NUM_BINS-1,
        accepted_pixels=int(curve["accepted"][i]), incorrect_accepted_pixels=int(curve["accepted_incorrect"][i]))


def infer():
    destination = ROOT / "condition_histograms"
    if destination.exists():
        raise RuntimeError("Refusing to overwrite existing Stage 5B histogram inference")
    config, config_hash = load_final_config()
    pipeline = ReliabilityPipeline(threshold=config["threshold"], uncertainty_method=config["uncertainty_method"],
        device="cuda", temperature_path=PROJECT/config["temperature_artifact"], local_files_only=True)
    assert pipeline.temperature == config["temperature"]
    assert NUM_BINS == 1000 and np.array_equal(HIST_EDGES, np.linspace(0, 1, 1001))
    baseline = {r["condition"]: r for r in read_csv(STAGE5A/"stage5a_combined_summary.csv")}
    inventory = {c: sorted((RGB_ROOT/c/"val").rglob("*_rgb_anon.png")) for c in EXPECTED}
    assert {c: len(paths) for c, paths in inventory.items()} == EXPECTED
    guard = InferenceBeforeGTAudit()
    destination.mkdir()
    (ROOT/"confusion_matrices").mkdir()
    checked = valid_count = changes = done = 0
    condition_audits = []
    print(f"Stage 5B clean pass: four uncertainties, canonical logit argmax; T={config['temperature']!r}", flush=True)
    for condition, paths in inventory.items():
        accumulators = {f"{method}_{label}_hist": UncertaintyAccumulator()
                        for method in METHODS for label in ("correct", "incorrect")}
        confusion = np.zeros((19,19), dtype=np.int64)
        n = errors = condition_changes = 0
        for path in paths:
            gt_path = find_gt_path(path, condition)
            guard.begin(PROJECT/path, PROJECT/gt_path)
            with torch.inference_mode():
                logits = pipeline.infer_logits(path)
                pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                scaled_pred = (logits.to(torch.float64)/pipeline.temperature).argmax(dim=1)[0].cpu().numpy()
                condition_changes += int(np.count_nonzero(pred != scaled_pred))
                scores = {}
                for method in METHODS:
                    values, confidence = pipeline.compute_uncertainty(logits, method)
                    assert values.shape == pred.shape and np.isfinite(values).all()
                    assert values.min() >= -1e-7 and values.max() <= 1+1e-7
                    scores[method] = values
                    del confidence
            checked += pred.size
            guard.inference_completed()
            with Image.open(gt_path) as source:
                gt = np.array(source)
            guard.finish()
            assert gt.shape == pred.shape
            valid = (gt >= 0) & (gt < 19)
            assert np.all(valid | (gt == 255))
            correct, incorrect = valid & (pred == gt), valid & (pred != gt)
            for method, values in scores.items():
                # Reuse exact Stage 2/3 histogram accumulation and clipping.
                accumulators[f"{method}_correct_hist"].update(values[correct])
                accumulators[f"{method}_incorrect_hist"].update(values[incorrect])
            n += int(valid.sum())
            errors += int(incorrect.sum())
            update_confusion_matrix(confusion, pred, gt)
            done += 1
            if done == 1 or done % 25 == 0 or done == 406:
                print(f"[{done}/406] {condition}; four histograms accumulated", flush=True)
            del logits, pred, scaled_pred, scores, values, gt, valid, correct, incorrect
        assert condition_changes == 0
        for old_path in (PROJECT/f"results/{condition}_val/confusion_matrix.npy",
                         PROJECT/f"results/stage3_calibration/stage3c/{condition}/confusion_matrix.npy",
                         STAGE5A/f"confusion_matrices/{condition}_confusion_matrix.npy"):
            np.testing.assert_array_equal(confusion, np.load(old_path))
        assert n == int(baseline[condition]["valid_pixels"]) and errors == int(baseline[condition]["incorrect_valid_pixels"])
        hist = {key: acc.hist for key, acc in accumulators.items()}
        for method in METHODS:
            assert hist[f"{method}_correct_hist"].dtype == np.int64
            assert int(hist[f"{method}_incorrect_hist"].sum()) == errors
            assert int(hist[f"{method}_correct_hist"].sum()) == n-errors
        # Calibrated Stage 3D histograms already use the same canonical labels.
        old = np.load(PROJECT/f"results/stage3_calibration/stage3d/{condition}/calibrated_uncertainty_histograms.npz")
        for method in ("msp", "entropy"):
            for label in ("correct", "incorrect"):
                np.testing.assert_array_equal(hist[f"calibrated_{method}_{label}_hist"],old[f"{method}_{label}_hist"])
        old.close()
        np.savez(destination/f"{condition}.npz", bin_edges=HIST_EDGES, **hist)
        np.save(ROOT/f"confusion_matrices/{condition}.npy",confusion)
        condition_audits.append(dict(condition=condition, num_images=len(paths), valid_pixels=n,
            incorrect_pixels=errors, baseline_risk=errors/n, prediction_changes=0,
            exact_stage1_stage3c_stage5a_confusion=True, exact_calibrated_stage3d_histograms=True))
        valid_count += n
        changes += condition_changes
        print(f"{condition}: exact canonical baseline and calibrated Stage 3D histograms PASS", flush=True)
    assert done == 406 and valid_count == 790712012 and changes == 0
    assert len(guard.events) == 406 and not guard.blocked
    (ROOT/"inference_audit.json").write_text(json.dumps(dict(
        num_images=done, valid_pixels=valid_count, all_prediction_pixels_checked=checked,
        prediction_changes=changes, conditions=condition_audits, final_config_hash=config_hash,
        temperature=config["temperature"], frozen_threshold=config["threshold"],
        frozen_deployed_method=config["uncertainty_method"], methods=list(METHODS),
        model_revision=config["model_revision"], evaluation_source_sha256=sha256(Path(__file__)),
        full_resolution_logits_shared=True, canonical_prediction="full_resolution_logits.argmax(dim=1)",
        data_access_records=guard.events, blocked_dataset_opens=guard.blocked,
        no_runtime_or_memory_benchmark=True), indent=2)+"\n")


def plot_curve(condition, curves, exact, risk_limit):
    names = {"raw_msp":"Raw MSP", "calibrated_msp":"Calibrated MSP (primary)",
             "raw_entropy":"Raw entropy", "calibrated_entropy":"Calibrated entropy"}
    styles = {"raw_msp":("#707070","--",1.7), "calibrated_msp":("#1268a3","-",2.8),
              "raw_entropy":("#db861b","--",1.7), "calibrated_entropy":("#9a4f9f","-.",1.7)}
    fig, ax = plt.subplots(figsize=(9,6.7))
    for method in METHODS:
        curve = curves[(condition,method)]
        color, style, width = styles[method]
        ax.plot(100*curve["coverage"],100*curve["risk"],color=color,ls=style,lw=width,label=names[method])
    ax.axhline(100*float(exact["baseline_risk"]),color="#657c57",ls=":",lw=1.8,
               label="Random rejection: expected risk")
    ax.scatter([100*float(exact["valid_coverage"])],[100*float(exact["accepted_risk"])],
               marker="D",s=75,c="black",edgecolors="white",linewidths=1,zorder=10,
               label="Frozen τ: exact Stage 5A point")
    ax.set(xlim=(0,100),ylim=(0,risk_limit),xlabel="Valid-pixel coverage (%)",ylabel="Accepted-pixel risk (%)",
           title=f"{condition.capitalize()} ACDC validation · risk–coverage")
    ax.grid(alpha=.2)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.17),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.1,right=.98,top=.9,bottom=.30)
    fig.text(.5,.055,"1000-bin descriptive curves; frozen deployment threshold remains unchanged.",ha="center",fontsize=9)
    fig.text(.5,.025,"Curves start at first nonzero histogram coverage; no risk is assigned at 0%.",ha="center",fontsize=9)
    filename="fig1_pooled_risk_coverage.png" if condition=="pooled" else f"fig_condition_{condition}.png"
    fig.savefig(ROOT/filename,dpi=300)
    plt.close(fig)


def analyze():
    if (ROOT/"stage5b_combined_summary.csv").exists():
        raise RuntimeError("Refusing to overwrite completed Stage 5B numerical summaries")
    config, config_hash = load_final_config()
    audit=json.loads((ROOT/"inference_audit.json").read_text())
    assert audit["num_images"]==406 and audit["valid_pixels"]==790712012 and audit["prediction_changes"]==0
    assert config_hash==audit["final_config_hash"]
    exact={r["condition"]:r for r in read_csv(STAGE5A/"stage5a_combined_summary.csv")}
    histograms={}
    for condition in EXPECTED:
        with np.load(ROOT/f"condition_histograms/{condition}.npz") as data:
            np.testing.assert_array_equal(data["bin_edges"],HIST_EDGES)
            histograms[condition]={k:data[k].copy() for k in data.files if k!="bin_edges"}
    pooled={k:sum((h[k] for h in histograms.values()),start=np.zeros(1000,dtype=np.int64))
            for k in histograms["fog"]}
    histograms["pooled"]=pooled
    np.savez(ROOT/"pooled_histograms.npz",bin_edges=HIST_EDGES,**pooled)
    curves, summaries, checkpoints, exact_points = {}, [], [], []
    packed={}
    for condition,hist in histograms.items():
        reference=exact[condition]
        assert float(reference["threshold"])==config["threshold"] and float(reference["temperature"])==config["temperature"]
        exact_points.append(dict(condition=condition,threshold=float(reference["threshold"]),
            valid_coverage=float(reference["valid_coverage"]),accepted_risk=float(reference["accepted_risk"]),
            baseline_risk=float(reference["baseline_risk"]),source="frozen Stage 5A combined CSV; exact, not histogram-derived"))
        for method in METHODS:
            correct,incorrect=hist[f"{method}_correct_hist"],hist[f"{method}_incorrect_hist"]
            assert int(correct.sum()+incorrect.sum())==int(reference["valid_pixels"])
            assert int(incorrect.sum())==int(reference["incorrect_valid_pixels"])
            curve=risk_coverage(correct,incorrect)
            assert curve["risk"][-1]==float(reference["baseline_risk"])
            curves[(condition,method)]=curve
            for key,value in curve.items():packed[f"{condition}__{method}__{key}"]=value
            summary=dict(condition=condition,method=method,num_images=int(reference["num_images"]),
                valid_pixels=int(reference["valid_pixels"]),incorrect_pixels=int(incorrect.sum()),
                baseline_risk=curve["risk"][-1],histogram_bins=1000,
                first_observed_coverage=float(curve["coverage"][0]),
                metric_method="1000-bin histogram approximation; canonical logit-argmax correctness",
                aurc_computed=False)
            for target in TARGETS:
                point=checkpoint(curve,target)
                checkpoints.append(dict(condition=condition,method=method,**point,
                    descriptive_only=True,final_threshold_unchanged=True))
                suffix=f"{int(round(target*100))}"
                summary[f"risk_at_requested_{suffix}_percent"]=point["risk"]
                summary[f"actual_coverage_at_requested_{suffix}_percent"]=point["actual_histogram_coverage"]
            summaries.append(summary)
    np.savez(ROOT/"risk_coverage_curves.npz",**packed)
    write_csv(ROOT/"stage5b_combined_summary.csv",summaries)
    write_csv(ROOT/"risk_at_target_coverages.csv",checkpoints)
    write_csv(ROOT/"frozen_operating_points.csv",exact_points)
    # The exact frozen point must lie inside the coverage bracket of its
    # containing bin. Its risk is checked against possible within-bin labels,
    # not forced onto a linearly interpolated curve sample.
    brackets=[]
    for point in exact_points:
        condition=point["condition"]
        h=histograms[condition]
        c,i=h["calibrated_msp_correct_hist"],h["calibrated_msp_incorrect_hist"]
        b=int(np.searchsorted(HIST_EDGES,config["threshold"],side="right")-1)
        accepted_before=int((c[:b]+i[:b]).sum()); errors_before=int(i[:b].sum())
        n=int(c.sum()+i.sum())
        exact_accepted=int(exact[condition]["accepted_valid_pixels"])
        exact_errors=int(exact[condition]["FN"])
        within=exact_accepted-accepted_before
        assert 0<=within<=int(c[b]+i[b])
        lower=errors_before+max(0,within-int(c[b]))
        upper=errors_before+min(within,int(i[b]))
        assert lower<=exact_errors<=upper
        brackets.append(dict(condition=condition,threshold_bin=b,
            bin_lower=float(HIST_EDGES[b]),bin_upper=float(HIST_EDGES[b+1]),
            coverage_before_bin=accepted_before/n,coverage_after_bin=(accepted_before+int(c[b]+i[b]))/n,
            exact_coverage=point["valid_coverage"],exact_risk=point["accepted_risk"],
            risk_before_bin=errors_before/accepted_before if accepted_before else None,
            risk_after_bin=(errors_before+int(i[b]))/(accepted_before+int(c[b]+i[b])),
            compatible_with_histogram=True,within_bin_error_lower=lower,within_bin_error_upper=upper))
    write_csv(ROOT/"frozen_point_histogram_consistency.csv",brackets)
    maxrisk=max(float(curve["risk"].max()) for curve in curves.values())
    risk_limit=min(100,math.ceil(maxrisk*100/5)*5+5)
    for condition in histograms:plot_curve(condition,curves,exact[condition],risk_limit)
    manifest=json.loads((ROOT/"protected_files_before.json").read_text())
    changed=[p for p,h in manifest.items() if not matches_frozen_hash(PROJECT / p, h)]
    assert not changed,changed
    assert load_final_config()==(config,config_hash)
    preservation=(f"PASS: all {len(manifest)} pre-existing protected files pass SHA-256 checks with the documented source relocation.\n"
        "Stages 1–5A results, temperature, threshold, final configuration and splits unchanged; sources match the documented relocation.\n"
        "Prior caches hashed as opaque bytes only; no train images/GT or train diagnostics were used.\n")
    (ROOT/"preservation_check.txt").write_text(preservation)
    explanation=("Stage 5B: descriptive risk–coverage evaluation only.\n"
        "Clean 406-image pass; canonical full-resolution logit argmax shared by all four methods.\n"
        "790,712,012 valid pixels; zero prediction changes from positive temperature scaling.\n"
        "Exact Stage 1/3C/5A confusion matrices; calibrated histograms exactly reproduce Stage 3D.\n"
        "Raw Stage 2 correctness labels were not reused because of its two historical softmax ties.\n"
        "1000 equal-width bins [0,1]; int64 correct/incorrect counts; pooled=sum(condition histograms).\n"
        "Accept bins LOW-to-HIGH uncertainty. Skip zero-accepted points; retain duplicate coverages from empty bins.\n"
        "Bin convention [lower,upper), last includes 1. Cumulative upper-bound thresholds are exclusive except final=1.\n"
        "100%-coverage risk equals exact frozen baseline for every condition/method.\n"
        "Checkpoint selection minimizes absolute coverage difference; ties use first ascending bin. No interpolation/subsampling.\n"
        "Exact deployed operating points loaded from Stage 5A separately; compatible with containing-bin count bounds.\n"
        "Random-rejection line is expected risk under correctness-independent rejection, not a confidence interval.\n"
        "AURC omitted: the first observed nonzero coverage depends on method and condition; integrating unequal domains\n"
        "would confound ordering comparisons, and risk at coverage=0 is not fabricated. Fixed-coverage risks are primary.\n"
        "The fixed 1000-bin approximation can coarsen coverage especially where low-uncertainty mass is concentrated.\n"
        "No deployment threshold selected/revised. No targeted failure audit or runtime/memory benchmark.\n")
    (ROOT/"evaluation_audit.txt").write_text(explanation+preservation)
    lines=["Stage 5B — descriptive risk–coverage analysis", "",explanation,
        f"Frozen deployed method=calibrated_msp; T={config['temperature']!r}; tau={config['threshold']!r}.",
        "Validation was previously analyzed in Stages 1–3; not a completely untouched test set.","",
        "Primary calibrated-MSP risk checkpoints: requested / actual coverage / risk (percent)"]
    for condition in histograms:
        lines.append(condition)
        for row in checkpoints:
            if row["condition"]==condition and row["method"]=="calibrated_msp":
                lines.append(f"  {100*row['requested_coverage']:.0f}% / {100*row['actual_histogram_coverage']:.6f}% / {100*row['risk']:.6f}%")
    lines += ["", "Exact frozen Stage 5A operating points:",json.dumps(exact_points,indent=2),"",
        "All-method risk comparisons and actual coverage values: stage5b_combined_summary.csv and risk_at_target_coverages.csv.",
        "The primary method and operating threshold remain frozen regardless of comparison-curve results.",
        "Lower risk at reduced coverage describes a retained subset with fewer errors. Rejected classes are not corrected.",
        "No statistical significance or real-world safety claims are made.", "",preservation,
        "STOP: Stage 5C and Stage 5D not begun."]
    (ROOT/"stage5b_summary.txt").write_text("\n".join(lines)+"\n")
    print(explanation,flush=True)
    for row in checkpoints:
        if row["method"]=="calibrated_msp":
            print(f"{row['condition']}: target={100*row['requested_coverage']:.0f}%, actual={100*row['actual_histogram_coverage']:.6f}%, risk={100*row['risk']:.6f}%",flush=True)
    print(preservation,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("infer","analyze"),required=True)
    args=parser.parse_args()
    os.chdir(PROJECT)
    infer() if args.phase=="infer" else analyze()
