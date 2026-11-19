"""Paired group summaries and equal-coverage diagnostics from completed histograms."""
import csv
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.cse805_error_prediction_v1.protocol import read_csv
from experiments.cse805_error_prediction_v1.report import metric_row, restore_accumulator
from .model import BASE, OUT, METHOD, OLD, MSP, read, sha256

RESULTS = OUT/"results"
LOW = (.01, .05, .10, .20)
METHODS = (MSP, OLD, METHOD)
NAMES = {MSP:"msp", OLD:"old", METHOD:"new"}
METRICS = ("auroc", "average_precision", "aurc", "retrospective80_risk",
           "retrospective80_actual_coverage", "training_threshold_actual_coverage",
           "training_threshold_accepted_error")


def csv_table(name, rows):
    path = RESULTS/name
    with path.with_suffix(".tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def matched_budget(acc, target):
    # Random inclusion within the boundary bin; expectations are not observed counts.
    mass = acc.correct + acc.incorrect
    counts, errors = np.cumsum(mass), np.cumsum(acc.incorrect)
    k = min(int(counts[-1]), max(1, int(np.floor(target*counts[-1] + .5))))
    b = int(np.searchsorted(counts, k, side="left"))
    before = int(counts[b] - mass[b])
    fraction = (k - before) / mass[b]
    expected = float(errors[b] - acc.incorrect[b] + fraction*acc.incorrect[b])
    return dict(matched_accepted_pixels=k, matched_expected_errors=expected, matched_risk=expected/k,
                boundary_bin=b, boundary_pixels=int(mass[b]), boundary_included_fraction=fraction)


def restored(metrics, saved, method):
    return restore_accumulator(metrics, saved[method+"_correct"], saved[method+"_incorrect"])


def collect():
    groups, low_rows = [], []
    lookup = BASE/"interpretation_v1/tables"
    old_veg = {(r["group"],r["method"],float(r["requested_coverage"])):r for r in
               read_csv(lookup/"pixel_class_acceptance.csv") if r["condition"]=="night" and r["predicted_class"]=="8"}
    old_qa = {(r["group"],r["method"],float(r["requested_coverage"])):r for r in
              read_csv(lookup/"metrics_policy_coverage.csv") if r["scope"]=="heldout_family"
              and r["weather"]=="night" and r["policy"]=="raw_secondary"}
    for group in read(OUT/"run_manifest.json")["outer_groups"]:
        original = BASE/"results/folds"/group
        record = read(original/"result.json")
        folder = OUT/"folds"/group
        new = read(folder/"evaluation.json")
        assert sha256(folder/"histograms.npz") == new["histogram_sha256"]
        assert (new["group"],new["weather"],new["num_images"]) == (group,record["weather"],record["num_images"])
        metrics = {**{m:record["methods"][m] for m in (MSP,OLD)}, METHOD:new["metrics"]}
        assert len({(v["num_pixels"],v["num_errors"]) for v in metrics.values()}) == 1
        with np.load(original/"histograms.npz", allow_pickle=False) as old_hist, \
             np.load(folder/"histograms.npz", allow_pickle=False) as new_hist:
            accumulators = {m:restored(metrics[m],old_hist,m) for m in (MSP,OLD)}
            accumulators[METHOD] = restore_accumulator(metrics[METHOD],new_hist["correct"],new_hist["incorrect"])
            for method,acc in accumulators.items():
                groups.append(dict(group=group, num_images=new["num_images"],
                              **metric_row("group",group,method,metrics[method],new["weather"])))
                if new["weather"]!="night":
                    continue
                detail = acc.finalize(LOW)
                curve = detail["risk_coverage"]
                for pt in detail["checkpoints"]:
                    target=pt["requested_coverage"]
                    if method == METHOD:
                        point=next(p for p in new["qa_raw_secondary"] if p["requested_coverage"]==target)
                        qa_cov,qa_risk=point["actual_coverage"],point["risk"]
                        idx=int(np.searchsorted(curve["accepted"],pt["accepted_pixels"]))
                        cut=int(curve["bin_index"][idx])
                        veg=int(new_hist["vegetation"][:cut+1].sum())
                        verr=int(new_hist["vegetation_errors"][:cut+1].sum())
                    else:
                        key=(group,method,target)
                        qa_cov,qa_risk=(float(old_qa[key][s]) for s in ("actual_coverage","accepted_error"))
                        ref=old_veg[key]
                        assert abs(float(ref["actual_coverage"])-pt["actual_coverage"]) < 1e-14
                        veg,verr=(int(ref[s]) for s in ("accepted_class_pixels","accepted_class_errors"))
                    assert 0 <= verr <= min(veg,pt["accepted_errors"]) and veg <= pt["accepted_pixels"]
                    low_rows.append(dict(group=group,method=method,**pt,**matched_budget(acc,target),
                        vegetation_accepted=veg,vegetation_accepted_errors=verr,
                        vegetation_share_of_accepted_errors=verr/pt["accepted_errors"] if pt["accepted_errors"] else None,
                        qa_actual_coverage=qa_cov,qa_raw_secondary_risk=qa_risk))
    return groups,low_rows


def compare(rows):
    maps = {m:{r["group"]:r for r in rows if r["method"]==m} for m in METHODS}
    assert maps[METHOD].keys()==maps[OLD].keys()==maps[MSP].keys()
    result=[]
    for scope in ("all","fog","night","rain","snow",*sorted(maps[OLD])):
        members=[g for g,r in maps[OLD].items() if scope in ("all",g,r["weather"])]
        entry=dict(scope=scope,num_groups=len(members),
                   num_images=sum(maps[OLD][g]["num_images"] for g in members),
                   num_valid_pixels=sum(maps[OLD][g]["num_valid_pixels"] for g in members),
                   num_error_pixels=sum(maps[OLD][g]["num_error_pixels"] for g in members))
        entry["group_macro_prevalence"]=mean(maps[OLD][g]["error_prevalence"] for g in members)
        for m in METHODS:
            for field in METRICS:
                entry[NAMES[m]+"_"+field]=mean(maps[m][g][field] for g in members)
        for method,base in ((METHOD,OLD),(METHOD,MSP),(OLD,MSP)):
            tag=NAMES[method]+"_minus_"+NAMES[base]
            delta=[100*(maps[method][g]["retrospective80_risk"]-maps[base][g]["retrospective80_risk"]) for g in members]
            entry[tag+"_risk80_pp"]=mean(delta)
            entry[tag+"_groups_improved"]=sum(d < -1e-10 for d in delta)
            entry[tag+"_groups_worsened"]=sum(d > 1e-10 for d in delta)
        result.append(entry)
    return result


def night_budget_summary(rows):
    result=[]
    for target in LOW:
        selected=[r for r in rows if r["requested_coverage"]==target]
        budgets={m:{r["group"]:r["matched_accepted_pixels"] for r in selected if r["method"]==m} for m in METHODS}
        if len(budgets[OLD])!=3 or budgets[MSP]!=budgets[OLD] or budgets[METHOD]!=budgets[OLD]:
            raise ValueError("Night methods must share each group's budget")
        total=sum(budgets[OLD].values())
        entry=dict(target=target,groups=3,matched_accepted_pixels_total=total)
        for m in METHODS:
            values=[r for r in selected if r["method"]==m]
            expected=sum(r["matched_expected_errors"] for r in values)
            tag=NAMES[m]
            entry[tag+"_equal_group_matched_risk"]=mean(r["matched_risk"] for r in values)
            entry[tag+"_matched_expected_errors_total"]=expected
            entry[tag+"_matched_risk_by_pixel_count"]=expected/total
            entry[tag+"_whole_bin_accepted_pixels_total"]=sum(r["accepted_pixels"] for r in values)
            entry[tag+"_whole_bin_accepted_errors_total"]=sum(r["accepted_errors"] for r in values)
        entry["new_minus_old_matched_expected_errors"]=entry["new_matched_expected_errors_total"]-entry["old_matched_expected_errors_total"]
        result.append(entry)
    return result


def plot(low, comparisons, group_rows):
    selected=[r for r in low if r["requested_coverage"]==.1]
    indexed={(r["group"],r["method"]):r for r in selected}
    nights=sorted({r["group"] for r in selected})
    fog_groups={r["group"] for r in group_rows if r["weather"]=="fog"}
    fogs=[r for r in comparisons if r["scope"] in fog_groups]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout="constrained")
    colors={MSP:"#6a6d75",OLD:"#b25020",METHOD:"#00796c"}
    markers={MSP:"x",OLD:"o",METHOD:"D"}
    left,right=axes
    for i,g in enumerate(nights):
        for m in METHODS:
            val=100*indexed[g,m]["matched_risk"]
            left.scatter(val,i+{MSP:-.15,OLD:0,METHOD:.15}[m],color=colors[m],marker=markers[m],s=55,zorder=3)
    left.set(yticks=range(len(nights)),yticklabels=[g[-4:] for g in nights],title="Night at 10% coverage",
             xlabel="Expected error among accepted pixels (%)",ylabel="Held-out filename group")
    left.invert_yaxis()
    for i,r in enumerate(fogs):
        a,b=r["old_minus_msp_risk80_pp"],r["new_minus_msp_risk80_pp"]
        right.plot([a,b],[i,i],color="#a0a2a4",zorder=1)
        for m,d in ((OLD,a),(METHOD,b)):
            right.scatter(d,i,color=colors[m],marker=markers[m],s=53,zorder=3)
    right.axvline(0,color=colors[MSP],linestyle="--",linewidth=1.1)
    right.set(yticks=range(len(fogs)),yticklabels=[r["scope"][-4:] for r in fogs],title="Fog near 80% coverage",
              xlabel="Accepted error minus MSP (percentage points)",ylabel="Held-out filename group")
    right.invert_yaxis()
    for axis in axes:
        axis.grid(axis="x",alpha=.2)
        axis.spines[["top","right"]].set_visible(False)
        axis.margins(y=.13)
    labels={MSP:"Raw MSP",OLD:"LR + Class",METHOD:"LR + Class × MSP"}
    handles=[Line2D([],[],linestyle="",marker=markers[m],color=colors[m],label=labels[m]) for m in METHODS]
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False)
    fig.savefig(RESULTS/"night_fog.png",dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    groups,low=collect()
    assert len(groups)==51 and len(low)==36
    comparisons=compare(groups)
    csv_table("group_metrics.csv",groups)
    csv_table("paired_comparisons.csv",comparisons)
    csv_table("night_low_coverage.csv",low)
    csv_table("night_budget_summary.csv",night_budget_summary(low))
    plot(low,comparisons,groups)
    for r in comparisons[:5]:
        print(r["scope"],"risk80 pp vs LR+Class / vs MSP:",r["new_minus_old_risk80_pp"],r["new_minus_msp_risk80_pp"])
    for r in low:
        if r["requested_coverage"]==.1:
            print(r["group"],r["method"],"matched 10% risk",r["matched_risk"],"whole-bin errors",r["accepted_errors"])


if __name__=="__main__":
    main()
