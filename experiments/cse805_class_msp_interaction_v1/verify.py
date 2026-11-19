"""Check fitted design, baseline identity, coverage accounting and frozen file hashes."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from experiments.cse805_error_prediction_v1.interpretation_v1.shared import ROWS, verify_execution
from experiments.cse805_error_prediction_v1.protocol import read_csv
from .model import BASE, DATA, OUT, OLD, MSP, METHOD, read, score_rows, sha256, write
from .report import RESULTS, METRICS


def verify_snapshot(folder, records):
    for relative, expected in records.items():
        path=folder/relative
        if path.stat().st_size!=expected["bytes"] or sha256(path)!=expected["sha256"]:
            raise RuntimeError("Frozen artifact no longer matches its recorded fingerprint: "+str(path))
    return len(records)


def main():
    verify_execution()
    seal=read(OUT/"run_manifest.json")
    for path,digest in seal["inputs"].items():
        assert sha256(BASE/path)==digest,path
    assert seal["analysis_plan_sha256"]==sha256(OUT/"protocol.json")
    assert seal["model_sha256"]==sha256(OUT/"model.py") and seal["train_sha256"]==sha256(OUT/"train.py")
    group_rows=read_csv(RESULTS/"group_metrics.csv")
    comp={r["scope"]:r for r in read_csv(RESULTS/"paired_comparisons.csv")}
    low=read_csv(RESULTS/"night_low_coverage.csv")
    original={(r["identity"],r["method"]):r for r in read_csv(BASE/"results/report/group_metrics.csv")
              if r["scope"]=="group" and r["method"] in (MSP,OLD)}
    for r in group_rows:
        if r["method"] in (MSP,OLD):
            src=original[r["group"],r["method"]]
            for k in (*METRICS,"num_valid_pixels","num_error_pixels","error_prevalence"):
                assert float(r[k])==float(src[k]),(r["group"],k)
    assert len(group_rows)==51 and len(comp)==22 and len(low)==36
    image_ids=[]
    all_qa=all_valid=all_errors=candidate_folds=candidate_pass=0
    selected=Counter()
    hist_checks=0
    for fold in read(DATA/"assignments.json"):
        group=fold["eval_group"]
        folder=OUT/"folds"/group
        fit=read(folder/"selection.json")
        ev=read(folder/"evaluation.json")
        assert fit["train_groups"]==fold["train_groups"] and fit["inner_folds"]==fold["inner_folds"]
        assert fit["n_train"]==512*(1600-fold["eval_image_count"])
        assert ev["manifest_sha256"]==fit["manifest_sha256"]==sha256(OUT/"run_manifest.json")
        assert ev["evaluate_sha256"]==sha256(OUT/"evaluate.py")
        assert ev["model_sha256"]==fit["model_sha256"]==sha256(folder/"interaction.joblib")
        assert ev["histogram_sha256"]==sha256(folder/"histograms.npz")
        means={}
        for C,records in fit["candidates"].items():
            assert {r["inner_fold"] for r in records}=={0,1,2}
            candidate_folds+=len(records)
            candidate_pass+=sum(r["status"]=="PASS" for r in records)
            if all(r["status"]=="PASS" for r in records):
                means[float(C)]=np.mean([r["ap"] for r in records])
        assert fit["selected_C"]==min(c for c,s in means.items() if max(means.values())-s<=1e-12)
        selected[fit["selected_C"]]+=1
        eva_ids=ev["evaluated_image_ids"]
        assert set(eva_ids)=={r["image_id"] for r in ROWS if r["recording_group"]==group}
        image_ids.extend(eva_ids)
        all_qa+=ev["qa_features_and_scores_verified"]
        all_valid+=ev["metrics"]["num_pixels"]
        all_errors+=ev["metrics"]["num_errors"]
        m=joblib.load(folder/"interaction.joblib")
        old_dir=DATA/"models"/group
        old_path=old_dir/"lr3_class.joblib"
        assert sha256(old_path)==read(old_dir/"selection.json")["model_hashes"][OLD]
        baseline=joblib.load(old_path)
        assert m["classifier"].coef_.shape==(1,41)
        np.testing.assert_array_equal(m["classifier"].classes_,[0,1])
        with np.load(DATA/"features"/(eva_ids[0]+".npz"),allow_pickle=False) as sample:
            X=sample["qa_X"]
            original_design=baseline["features"].transform(X)
            design=m["features"].transform(X).toarray()
            np.testing.assert_array_equal(design[:,:22],original_design)
            np.testing.assert_array_equal(design[:,22:],original_design[:,3:]*original_design[:,0,None])
            np.testing.assert_array_equal(score_rows(m,X),m.predict_proba(X)[:,1])
        with np.load(BASE/"results/folds"/group/"histograms.npz") as h_old, np.load(folder/"histograms.npz") as h_new:
            for point in (p for p in low if p["group"]==group):
                method=point["method"]
                hist,prefix=(h_new,"") if method==METHOD else (h_old,method+"_")
                n=hist[prefix+"correct"]+hist[prefix+"incorrect"]
                e=hist[prefix+"incorrect"]
                nc=np.r_[0,np.cumsum(n[n>0])]
                ec=np.r_[0,np.cumsum(e[n>0])]
                k=int(point["matched_accepted_pixels"])
                expected=np.interp(k,nc,ec)
                np.testing.assert_allclose(expected,float(point["matched_expected_errors"]),atol=1e-7,rtol=1e-13)
                idx=1+np.argmin(abs(nc[1:]/nc[-1]-float(point["requested_coverage"])))
                assert (int(nc[idx]),int(ec[idx]))==(int(point["accepted_pixels"]),int(point["accepted_errors"]))
                hist_checks+=1
    assert len(set(image_ids))==len(image_ids)==1600
    assert all_qa==1600*8192 and all_valid==3059663748 and all_errors==523653212
    assert candidate_folds==153 and sum(selected.values())==17 and hist_checks==36
    assert abs(float(comp["fog"]["old_minus_msp_risk80_pp"])+1.424295)<1e-5
    assert abs(float(comp["all"]["old_minus_msp_risk80_pp"])+.728093)<1e-5
    preserved=dict(cse805=verify_snapshot(BASE,read(BASE/"interpretation_v1/logs/frozen_extension_before.json")),
                   interpretation=verify_snapshot(BASE/"interpretation_v1",read(BASE/"interpretation_v1/logs/completion.json")["artifacts"]),
                   comp895=verify_snapshot(BASE.parents[1],read(BASE/"logs/preservation_before.json")))
    artifacts={str(p.relative_to(OUT)):sha256(p) for folder in (OUT/"folds",RESULTS) for p in sorted(folder.rglob("*"))
               if p.is_file() and p.name!="verification.json"}
    write(RESULTS/"verification.json", dict(status="PASS", verified_utc=datetime.now(timezone.utc).isoformat(),
          images=1600,outer_groups=17,selected_C_counts=selected,inner_fits=candidate_folds,converged_inner_fits=candidate_pass,
          full_valid_pixels=all_valid,segmentation_errors=all_errors,original_QA_vectors_and_fast_scores_bitwise_verified=all_qa,
          original_preprocessing_and_interaction_formula_checked_on_all_folds=True,
          night_whole_bin_and_matched_budget_cross_checks=hist_checks,original_files_hash_verified=preserved,
          manifest_sha256=sha256(OUT/"run_manifest.json"),artifact_sha256=artifacts,
          source_sha256={p.name:sha256(p) for p in sorted(OUT.glob("*.py"))}))
    print("PASS: 17 folds, 1600 images, 3,059,663,748 pixels; 36 Night count/budget comparisons.")
    print("Original artifacts matched their frozen hashes:",preserved)


if __name__=="__main__":
    main()
