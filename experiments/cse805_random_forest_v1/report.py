"""Six-way paired comparison using original full-population histograms and RF folds."""
import csv
from statistics import mean

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from experiments.cse805_error_prediction_v1.protocol import read_csv
from experiments.cse805_error_prediction_v1.report import metric_row, restore_accumulator
from experiments.cse805_error_prediction_v1.evaluation import predict_scores
from experiments.cse805_error_prediction_v1.interpretation_v1.shared import models, samples
from experiments.cse805_error_prediction_v1.interpretation_v1.metric_sensitivity import policies
from experiments.cse805_class_msp_interaction_v1.report import LOW, METRICS, matched_budget
from .model import BASE, OUT, INTERACTION, RF, LR_INTERACTION, METHODS, LABELS, read, sha256

RESULTS=OUT/"results"
SCOPES=("all","fog","night","rain","snow")


def table(name,rows):
    path=RESULTS/name
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def interpolated_curve(acc,targets):
    mass=acc.correct+acc.incorrect
    occupied=mass>0
    accepted=np.r_[0,np.cumsum(mass[occupied])]
    errors=np.r_[0,np.cumsum(acc.incorrect[occupied])]
    budgets=np.maximum(1,np.floor(targets*accepted[-1]+.5)).astype(np.int64)
    return np.interp(budgets,accepted,errors)/budgets


def collect():
    rows,low,curve_rows,timings=[],[],[],[]
    src=BASE/"interpretation_v1/tables"
    old_veg={(r["group"],r["method"],float(r["requested_coverage"])):r
             for r in read_csv(src/"pixel_class_acceptance.csv") if r["condition"]=="night" and r["predicted_class"]=="8"}
    old_qa={(r["group"],r["method"],float(r["requested_coverage"])):r
            for r in read_csv(src/"metrics_policy_coverage.csv") if r["scope"]=="heldout_family"
            and r["weather"]=="night" and r["policy"]=="raw_secondary"}
    for group in read(OUT/"run_manifest.json")["outer_groups"]:
        orig=BASE/"results/folds"/group
        lr_folder=INTERACTION/"folds"/group
        rf_folder=OUT/"folds"/group
        frozen=read(orig/"result.json")
        exts={LR_INTERACTION:read(lr_folder/"evaluation.json"),RF:read(rf_folder/"evaluation.json")}
        selection=read(rf_folder/"selection.json")
        rf=exts[RF]
        weather=frozen["weather"]
        assert all(e["weather"]==weather and e["group"]==group for e in exts.values())
        assert frozen["num_images"]==len(rf["images"])==exts[LR_INTERACTION]["num_images"]
        timings.append(dict(group=group,weather=weather,images=frozen["num_images"],depth=selection["selected_depth"],
                            inner_AP_d8=selection["mean_inner_weighted_AP"]["8"],inner_AP_d14=selection["mean_inner_weighted_AP"]["14"],
                            training_sample_rows=selection["n_train"],per_tree_bootstrap_draws=selection["bootstrap_draws_per_tree"],
                            fitting_and_selection_seconds=selection["seconds"],evaluation_seconds=rf["eval_seconds"],
                            rf_scoring_seconds=rf["score_seconds"],model_bytes=(rf_folder/"forest.joblib").stat().st_size))
        metrics={**frozen["methods"],**{m:e["metrics"] for m,e in exts.items()}}
        assert set(metrics)==set(METHODS) and len({(v["num_pixels"],v["num_errors"]) for v in metrics.values()})==1
        for directory,rec in ((lr_folder,exts[LR_INTERACTION]),(rf_folder,rf)):
            assert sha256(directory/"histograms.npz")==rec["histogram_sha256"]
        with (np.load(orig/"histograms.npz",allow_pickle=False) as h1,
              np.load(lr_folder/"histograms.npz",allow_pickle=False) as h2,
              np.load(rf_folder/"histograms.npz",allow_pickle=False) as h3):
            for m in METHODS:
                hist,prefix=(h2,"") if m==LR_INTERACTION else (h3,"") if m==RF else (h1,m+"_")
                acc=restore_accumulator(metrics[m],hist[prefix+"correct"],hist[prefix+"incorrect"])
                rows.append(dict(num_images=frozen["num_images"],**metric_row("group",group,m,metrics[m],weather)))
                full=acc.finalize(LOW)
                curve=full["risk_coverage"]
                # A common grid makes the full RC curve portable without copying raw pixels.
                targets=np.arange(1,101,dtype=np.float64)/100
                matched=interpolated_curve(acc,targets)
                rows[-1]["matched80_risk"]=float(matched[79])
                for q,risk in zip(targets,matched):
                    index=int(np.argmin(abs(curve["coverage"]-q)))
                    curve_rows.append(dict(group=group,weather=weather,method=m,target=float(q),
                        actual_coverage=float(curve["coverage"][index]),whole_bin_risk=float(curve["risk"][index]),
                        matched_risk=float(risk)))
                if weather!="night":
                    continue
                for pt in full["checkpoints"]:
                    q=pt["requested_coverage"]
                    if m in exts:
                        point=next(p for p in exts[m]["qa_raw_secondary"] if p["requested_coverage"]==q)
                        qcov,qrisk=point["actual_coverage"],point["risk"]
                        i=int(np.searchsorted(curve["accepted"],pt["accepted_pixels"]))
                        b=int(curve["bin_index"][i])
                        veg=int(hist["vegetation"][:b+1].sum())
                        verr=int(hist["vegetation_errors"][:b+1].sum())
                    else:
                        idx=(group,m,q)
                        ref=old_veg[idx]
                        assert float(ref["actual_coverage"])==pt["actual_coverage"]
                        veg,verr=int(ref["accepted_class_pixels"]),int(ref["accepted_class_errors"])
                        qcov,qrisk=float(old_qa[idx]["actual_coverage"]),float(old_qa[idx]["accepted_error"])
                    budget=matched_budget(acc,q)
                    np.testing.assert_allclose(budget["matched_risk"],matched[round(q*100)-1],rtol=1e-12,atol=1e-14)
                    b=budget["boundary_bin"]
                    below=int(acc.correct[:b].sum()+acc.incorrect[:b].sum())
                    total=metrics[m]["num_pixels"]
                    low.append(dict(group=group,method=m,**pt,**budget,qa_actual_coverage=qcov,qa_raw_risk=qrisk,
                        boundary_coverage_lower=below/total,boundary_coverage_upper=(below+budget["boundary_pixels"])/total,
                        vegetation_accepted_pixels=veg,vegetation_accepted_errors=verr,
                        vegetation_share_of_accepted_errors=verr/pt["accepted_errors"] if pt["accepted_errors"] else None))
    return rows,low,curve_rows,timings


def summarize(rows):
    summary,paired=[],[]
    by_method={m:{r["identity"]:r for r in rows if r["method"]==m} for m in METHODS}
    groups=sorted(by_method[RF])
    for scope in (*SCOPES,*groups):
        members=[g for g in groups if scope in ("all",g,by_method[RF][g]["weather"])]
        if scope in SCOPES:
            for m in METHODS:
                selected=[by_method[m][g] for g in members]
                vals={k:mean(r[k] for r in selected) for k in ("error_prevalence",*METRICS,"matched80_risk")}
                summary.append(dict(scope=scope,method=m,num_groups=len(members),num_images=sum(r["num_images"] for r in selected),
                    num_valid_pixels=sum(r["num_valid_pixels"] for r in selected),num_error_pixels=sum(r["num_error_pixels"] for r in selected),
                    **vals,max_abs_retrospective80_coverage_error_pp=100*max(abs(r["retrospective80_coverage_error"]) for r in selected),
                    training_threshold_coverage_min=min(r["training_threshold_actual_coverage"] for r in selected),
                    training_threshold_coverage_max=max(r["training_threshold_actual_coverage"] for r in selected)))
        for other in METHODS[:-1]:
            ds={k:np.array([by_method[RF][g][k]-by_method[other][g][k] for g in members]) for k in (*METRICS,"matched80_risk")}
            delta=100*ds["retrospective80_risk"]
            omitted=(delta.sum()-delta)/(len(delta)-1) if len(delta)>1 else None
            paired.append(dict(scope=scope,reference=other,groups=len(members),rf_minus_ref_risk80_pp=float(delta.mean()),
                rf_minus_ref_matched80_pp=100*float(ds["matched80_risk"].mean()),
                improved=int(sum(delta < -1e-10)),tied=int(sum(abs(delta)<=1e-10)),worsened=int(sum(delta>1e-10)),
                smallest_group_delta_pp=float(delta.min()),largest_group_delta_pp=float(delta.max()),
                leave_one_group_out_mean_min_pp=float(omitted.min()) if omitted is not None else None,
                leave_one_group_out_mean_max_pp=float(omitted.max()) if omitted is not None else None,
                rf_minus_ref_auroc=float(ds["auroc"].mean()),rf_minus_ref_ap=float(ds["average_precision"].mean()),
                rf_minus_ref_aurc=float(ds["aurc"].mean()),
                max_abs_paired_coverage_difference_pp=float(abs(ds["retrospective80_actual_coverage"]).max()*100)))
    return summary,paired


def night_summary(rows):
    summaries=[]
    for q in LOW:
        budgets={}
        for m in METHODS:
            batch=[r for r in rows if r["method"]==m and r["requested_coverage"]==q]
            budgets[m]={r["group"]:r["matched_accepted_pixels"] for r in batch}
            K=sum(budgets[m].values())
            E=sum(r["matched_expected_errors"] for r in batch)
            summaries.append(dict(target=q,method=m,num_groups=len(batch),matched_pixels=K,
                expected_errors_at_matched_budget=E,matched_risk_pixel_weighted=E/K,
                matched_risk_group_macro=mean(r["matched_risk"] for r in batch),
                raw_QA_risk_group_macro=mean(r["qa_raw_risk"] for r in batch),
                whole_bin_accepted_pixels=sum(r["accepted_pixels"] for r in batch),
                whole_bin_accepted_errors=sum(r["accepted_errors"] for r in batch)))
        assert all(b==budgets[RF] and len(b)==3 for b in budgets.values())
    return summaries


def qa_diagnostics(rows,groups):
    """Apply the original QA tie policies and inspect the Night 10% accepted set."""
    classes, sensitivity = [], []
    original = read_csv(BASE/"interpretation_v1/tables/metrics_policy_summary.csv")
    for r in original:
        if r["scope"]=="heldout_family":
            sensitivity.append({k:r[k] for k in ("group","weather","method","policy","sampled_pixels","aurc")})
    for group,weather in sorted({(r["identity"],r["weather"]) for r in groups}):
        qa=samples(group)
        X,y,w=qa["X"],qa["errors"],qa["weights"]
        old=models(group)[0] if weather=="night" else {}
        extra={LR_INTERACTION:joblib.load(INTERACTION/"folds"/group/"interaction.joblib"),
               RF:joblib.load(OUT/"folds"/group/"forest.joblib")}
        for m in (METHODS if weather=="night" else extra):
            scores=extra[m].predict_proba(X)[:,1] if m in extra else predict_scores(m,X,old.get(m))
            if m in extra:
                for policy,(stats,_) in policies(scores,y,w,qa["tie_keys"]).items():
                    sensitivity.append(dict(group=group,weather=weather,method=m,policy=policy,
                                            sampled_pixels=len(X),aurc=stats["aurc"]))
            if weather!="night":
                continue
            order=np.lexsort((qa["tie_keys"],scores))
            mass=np.cumsum(w[order],dtype=np.float64)
            idx=int(np.argmin(abs(mass/mass[-1]-.1)))
            keep=order[:idx+1]
            risk=float(np.average(y[keep],weights=w[keep]))
            ref=next(r for r in rows if r["group"]==group and r["method"]==m and r["requested_coverage"]==.1)
            np.testing.assert_allclose([risk,mass[idx]/mass[-1]],[ref["qa_raw_risk"],ref["qa_actual_coverage"]],rtol=1e-11,atol=1e-12)
            edge=scores[order[np.searchsorted(mass,.1*mass[-1])]]
            below=w[scores<edge].sum()/w.sum()
            equal=w[scores==edge].sum()/w.sum()
            err=w[keep]*y[keep]
            classes.append(dict(group=group,method=m,sampled_pixels=len(X),actual_coverage=mass[idx]/mass[-1],
                accepted_error=risk,raw_score_at_10=edge,raw_coverage_strictly_below=below,
                raw_coverage_including_tie=below+equal,raw_true_tie_mass=equal,
                predicted_vegetation_share_of_errors=err[X[keep,3]==8].sum()/err.sum(),
                predicted_road_share_of_accepted=w[keep][X[keep,3]==0].sum()/w[keep].sum()))
    assert len(classes)==18 and len(sensitivity)==17*6*5
    return classes,sensitivity


def figures(summary,curves):
    fig=plt.figure(figsize=(12,7.8),layout="constrained")
    gs=fig.add_gridspec(2,3,height_ratios=(1,1.25))
    heat=fig.add_subplot(gs[0,:])
    values={(r["scope"],r["method"]):r["retrospective80_risk"] for r in summary}
    grid=np.array([[100*(values[s,m]-values[s,"raw_msp"]) for s in SCOPES] for m in METHODS[1:]])
    # Symmetric colors distinguish increased error from reduced error.
    scale=max(abs(grid.ravel()).max(),.01)
    heat.imshow(grid,cmap="RdBu_r",norm=TwoSlopeNorm(0,-scale,scale),aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            heat.text(j,i,f"{grid[i,j]:+.3f}",ha="center",va="center",color="white" if abs(grid[i,j])>.58*scale else "#12202e")
    heat.set(xticks=range(5),xticklabels=[s.title() for s in SCOPES],yticks=range(5),
             yticklabels=[LABELS[m] for m in METHODS[1:]],title="Accepted error near 80% coverage minus MSP  (percentage points, lower is better)")
    heat.tick_params(length=0)
    styles=[("#4c5968",":"),("#8299ae","--"),("#885ea8","-."),("#c17915","-"),("#008681","--"),("#b22938","-")]
    handles=[]
    nights=sorted({r["group"] for r in curves if r["weather"]=="night"})
    for i,g in enumerate(nights):
        ax=fig.add_subplot(gs[1,i])
        for m,(color,style) in zip(METHODS,styles):
            series=[r for r in curves if r["group"]==g and r["method"]==m and r["target"]<=.20]
            line,=ax.plot([r["target"]*100 for r in series],[r["matched_risk"]*100 for r in series],
                         color=color,linestyle=style,linewidth=2.4 if m==RF else 1.65,label=LABELS[m])
            if i==0:
                handles.append(line)
        ax.axvline(10,color="#a5a8ac",linestyle=":",lw=1)
        ax.set(title="Night group "+g[-4:],xlabel="Accepted pixel budget (%)",xticks=[1,5,10,15,20])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
        if i==0:
            ax.set_ylabel("Expected accepted error (%)")
    axes=fig.axes[1:]
    upper=max(a.get_ylim()[1] for a in axes)
    for ax in axes:
        ax.set_ylim(0,upper)
    fig.legend(handles=handles,loc="outside lower center",ncol=6,frameon=False,fontsize=9)
    fig.savefig(RESULTS/"final_comparison.png",dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    groups,low,curves,timings=collect()
    summary,paired=summarize(groups)
    night=night_summary(low)
    qa,sensitivity=qa_diagnostics(low,groups)
    assert (len(groups),len(low),len(curves),len(summary),len(paired),len(night))==(102,72,10200,30,110,24)
    for name,rows in (("final_comparison.csv",summary),("group_metrics.csv",groups),("rf_paired_differences.csv",paired),
                      ("night_low_coverage.csv",low),("night_summary.csv",night),("risk_coverage.csv",curves),
                      ("night_qa10_classes.csv",qa),("qa_aurc_sensitivity.csv",sensitivity),("training_and_runtime.csv",timings)):
        table(name,rows)
    figures(summary,curves)
    for r in summary:
        if r["scope"]=="all":
            print(r["method"],*[round(r[k],7) for k in ("auroc","average_precision","aurc","retrospective80_risk")])
    for r in night:
        if r["target"]==.10:
            print("Night 10%",r["method"],r["matched_risk_group_macro"],r["expected_errors_at_matched_budget"])


if __name__=="__main__":
    main()
