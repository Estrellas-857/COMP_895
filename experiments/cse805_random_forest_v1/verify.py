"""Independent split, input identity, count and comparative-summary checks."""
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata

import joblib
import numpy as np

from experiments.cse805_error_prediction_v1.interpretation_v1.shared import ROWS, verify_execution
from experiments.cse805_error_prediction_v1.protocol import read_csv
from experiments.cse805_class_msp_interaction_v1.verify import verify_snapshot
from experiments.cse805_class_msp_interaction_v1.report import METRICS
from .model import BASE, DATA, OUT, INTERACTION, LR_INTERACTION, RF, METHODS, OPTIONS, read, sha256, write


def main():
    verify_execution()
    seal=read(OUT/"run_manifest.json")
    manifest_hash=sha256(OUT/"run_manifest.json")
    pre=read(INTERACTION/"results/verification.json")
    assert sha256(INTERACTION/"results/verification.json")==seal["interaction_receipt_sha256"]
    for folder,index in ((BASE,seal["frozen_sources"]),(OUT,seal["rf_source_sha256"]),
                         (INTERACTION,pre["artifact_sha256"]),(INTERACTION,pre["source_sha256"])):
        assert all(sha256(folder/k)==v for k,v in index.items()),folder
    result=OUT/"results"
    groups=read_csv(result/"group_metrics.csv")
    finals=read_csv(result/"final_comparison.csv")
    lows=read_csv(result/"night_low_coverage.csv")
    summaries=read_csv(result/"night_summary.csv")
    qa=read_csv(result/"night_qa10_classes.csv")
    sensitivity=read_csv(result/"qa_aurc_sensitivity.csv")
    assert len(groups)==len({(r["identity"],r["method"]) for r in groups})==102
    assert len(finals)==len({(r["scope"],r["method"]) for r in finals})==30
    assert len(lows)==72 and len(qa)==18 and len(summaries)==24
    by_policy={(r["group"],r["method"],r["policy"]):r for r in sensitivity}
    assert len(by_policy)==len(sensitivity)==510
    for r in read_csv(BASE/"interpretation_v1/tables/metrics_policy_summary.csv"):
        if r["scope"]=="heldout_family":
            assert r["aurc"]==by_policy[r["group"],r["method"],r["policy"]]["aurc"]
    for r in groups:
        for policy in ("raw_grouped","raw_secondary","arcsine_10000_grouped","arcsine_100000_grouped","arcsine_1000000_grouped"):
            qa_auc=by_policy[r["identity"],r["method"],policy]
            assert int(qa_auc["sampled_pixels"])==8192*int(r["num_images"]) and 0<=float(qa_auc["aurc"])<=1
    original={(r["identity"],r["method"]):r for r in read_csv(BASE/"results/report/group_metrics.csv") if r["scope"]=="group"}
    original.update({(r["identity"],r["method"]):r for r in read_csv(INTERACTION/"results/group_metrics.csv") if r["method"]==LR_INTERACTION})
    assert len(original)==85
    for r in groups:
        if r["method"]!=RF:
            baseline=original[r["identity"],r["method"]]
            for key in (*METRICS,"error_prevalence","num_valid_pixels","num_error_pixels","threshold_accepted_error_pixels"):
                assert float(r[key])==float(baseline[key]),(r["identity"],r["method"],key)
    for r in finals:
        selected=[g for g in groups if g["method"]==r["method"] and r["scope"] in ("all",g["weather"])]
        assert len(selected)==int(r["num_groups"])
        for key in (*METRICS,"matched80_risk"):
            np.testing.assert_allclose(np.mean([float(g[key]) for g in selected]),float(r[key]),rtol=1e-13,atol=1e-14)
    checked=set()
    chosen=Counter()
    inner_count=qa_count=valid_count=error_count=hist_checks=0
    for fold in read(DATA/"assignments.json"):
        g=fold["eval_group"]
        folder=OUT/"folds"/g
        sel=read(folder/"selection.json")
        ev=read(folder/"evaluation.json")
        assert sel["train_groups"]==fold["train_groups"] and g not in sel["train_groups"] and sel["inner_folds"]==fold["inner_folds"]
        assert sel["n_train"]==512*(1600-fold["eval_image_count"])
        assert sel["manifest_sha256"]==ev["manifest_sha256"]==manifest_hash
        assert sha256(folder/"forest.joblib")==sel["model_sha256"]==ev["model_sha256"]
        assert sha256(folder/"histograms.npz")==ev["histogram_sha256"]
        for name,digest in ev["scoring_source_sha256"].items():
            assert sha256(OUT/name)==digest
        assert ev["metrics"]["fixed_threshold"]["complete_population"] and ev["metrics"]["fixed_threshold"]["threshold"]==sel["threshold"]
        correct_images={r["image_id"] for r in ROWS if r["recording_group"]==g}
        assert set(ev["images"])==correct_images and len(ev["images"])==len(correct_images) and checked.isdisjoint(correct_images)
        checked.update(correct_images)
        assert ev["qa_fullstream_vs_sklearn_exact"]==len(correct_images)*8192
        qa_count+=ev["qa_fullstream_vs_sklearn_exact"]
        valid_count+=ev["metrics"]["num_pixels"]
        error_count+=ev["metrics"]["num_errors"]
        scores={}
        for d in OPTIONS["candidate_depth"]:
            entries=[r for r in sel["candidates"] if r["depth"]==d]
            assert len(entries)==3 and {r["inner"] for r in entries}=={0,1,2}
            scores[d]=np.mean([r["weighted_AP"] for r in entries])
            assert np.isfinite(scores[d]) and abs(scores[d]-sel["mean_inner_weighted_AP"][str(d)])<1e-12
            inner_count+=len(entries)
        assert sel["selected_depth"]==min(d for d,v in scores.items() if max(scores.values())-v<=1e-12)
        chosen[sel["selected_depth"]]+=1
        pipeline=joblib.load(folder/"forest.joblib")
        model=pipeline["forest"]
        assert model.max_depth==sel["selected_depth"] and len(model.estimators_)==OPTIONS["n_estimators"]
        assert all(getattr(model,k)==v for k,v in OPTIONS.items() if k not in ("candidate_depth","fit_threads"))
        assert model.n_features_in_==22 and np.array_equal(model.classes_,[0,1]) and model.n_jobs==1
        assert model._n_samples_bootstrap==sel["bootstrap_draws_per_tree"] and abs(model._n_samples_bootstrap-sel["n_train"]*.5)<=1
        with np.load(DATA/"features"/(ev["images"][0]+".npz")) as sample:
            x=sample["qa_X"]
            z=pipeline["features"].transform(x)
            np.testing.assert_array_equal(z[:,:3],x[:,:3])
            np.testing.assert_array_equal(z[:,3:],np.eye(19,dtype=np.float32)[x[:,3].astype(int)])
        if ev["weather"]!="night":
            continue
        with (np.load(BASE/"results/folds"/g/"histograms.npz") as h1,
              np.load(INTERACTION/"folds"/g/"histograms.npz") as h2,np.load(folder/"histograms.npz") as h3):
            for p in (r for r in lows if r["group"]==g):
                m=p["method"]
                hist,prefix=(h2,"") if m==LR_INTERACTION else (h3,"") if m==RF else (h1,m+"_")
                n=hist[prefix+"correct"]+hist[prefix+"incorrect"]
                pos=n>0
                nc=np.r_[0,np.cumsum(n[pos])]
                ec=np.r_[0,np.cumsum(hist[prefix+"incorrect"][pos])]
                q=float(p["requested_coverage"])
                k=int(np.floor(q*nc[-1]+.5))
                i=1+np.argmin(abs(nc[1:]/nc[-1]-q))
                assert k==int(p["matched_accepted_pixels"]) and (nc[i],ec[i])==(int(p["accepted_pixels"]),int(p["accepted_errors"]))
                np.testing.assert_allclose(np.interp(k,nc,ec),float(p["matched_expected_errors"]),rtol=1e-13,atol=1e-7)
                hist_checks+=1
    for r in summaries:
        rows=[p for p in lows if p["method"]==r["method"] and p["requested_coverage"]==r["target"]]
        assert len(rows)==3
        np.testing.assert_allclose(sum(float(p["matched_expected_errors"]) for p in rows),float(r["expected_errors_at_matched_budget"]),rtol=1e-13)
        assert sum(int(p["matched_accepted_pixels"]) for p in rows)==int(r["matched_pixels"])
    assert (len(checked),inner_count,qa_count,valid_count,error_count,hist_checks)==(1600,102,13107200,3059663748,523653212,72)
    preserved=dict(original_extension=verify_snapshot(BASE,read(BASE/"interpretation_v1/logs/frozen_extension_before.json")),
                   interpretation=verify_snapshot(BASE/"interpretation_v1",read(BASE/"interpretation_v1/logs/completion.json")["artifacts"]),
                   comp895=verify_snapshot(BASE.parents[1],read(BASE/"logs/preservation_before.json")),
                   bound_interaction_artifacts=len(pre["artifact_sha256"]))
    assets={str(p.relative_to(OUT)):sha256(p) for directory in (OUT/"folds",result) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name!="verification.json"}
    write(result/"verification.json",dict(status="PASS",verified_utc=datetime.now(timezone.utc).isoformat(),
          methods=list(METHODS),images=len(checked),full_valid_pixels=valid_count,segmentation_errors=error_count,
          outer_models=17,inner_fits=inner_count,selected_depths=chosen,matched_original_group_method_rows=85,
          fullstream_scores_bitwise_equal_to_sklearn_on_saved_QA_pixels=qa_count,
          night_coverage_budget_and_count_checks=hist_checks,raw_QA_class_and_tie_checks=len(qa),
          secondary_QA_group_method_AURC_policies=len(sensitivity),
          original_files_hash_verified=preserved,manifest_sha256=manifest_hash,triton_version=importlib.metadata.version("triton"),
          source_sha256={p.name:sha256(p) for p in sorted(OUT.glob("*.py"))},artifact_sha256=assets))
    print("PASS: all 17 RF models, 102 inner fits, 102 group/method rows, 3,059,663,748 pixels and 72 Night checkpoints.")
    print("Frozen inputs verified:",preserved)


if __name__=="__main__":
    main()
