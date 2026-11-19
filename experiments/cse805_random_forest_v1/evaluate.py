"""Stream frozen full-pixel features, verifying actual RF scores against sklearn."""
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch

from experiments.cse805_error_prediction_v1 import features
from experiments.cse805_error_prediction_v1.evaluation import HistogramAccumulator, HISTOGRAM_BINS, histogram_indices
from experiments.cse805_error_prediction_v1.interpretation_v1.shared import group_rows
from experiments.paired_calibration_v1.metrics import weighted_risk_coverage
from .model import BASE, DATA, OUT, TARGETS, read, sha256, write
from .gpu_forest import ForestScorer


def evaluate(group, records, manifest, scorer_hash):
    folder=OUT/"folds"/group
    sel=read(folder/"selection.json")
    model_path=folder/"forest.joblib"
    assert sel["manifest_sha256"]==manifest and sel["model_sha256"]==sha256(model_path)
    output=folder/"evaluation.json"
    histogram=folder/"histograms.npz"
    if output.exists():
        old=read(output)
        if (old["model_sha256"]!=sel["model_sha256"] or old["scoring_source_sha256"]!=scorer_hash
                or old["histogram_sha256"]!=sha256(histogram)):
            raise RuntimeError("Stale evaluation for "+group)
        return
    start=perf_counter()
    model=joblib.load(model_path)
    gpu=ForestScorer(model)
    rows=group_rows(group)
    weather=rows[0]["condition"]
    acc=HistogramAccumulator(threshold=sel["threshold"])
    veg,veg_err=np.zeros((2,HISTOGRAM_BINS),dtype=np.int64)
    qa={k:[] for k in ("scores","errors","weights","tie_keys")}
    checked=0
    score_seconds=0.
    for i,row in enumerate(rows,1):
        rec=records[row["image_id"]]
        native=rec["native_cache"]
        assert rec["recording_group"]==group and rec["gt_sha256"]==row["gt_sha256"] and row["condition"]==weather
        if sha256(rec["path"])!=rec["sha256"] or sha256(native["path"])!=native["sha256"]:
            raise RuntimeError("Frozen input checksum mismatch: "+row["image_id"])
        logits=np.load(native["path"],allow_pickle=False)
        gt=features.load_target(row).ravel()
        with np.load(rec["path"],allow_pickle=False) as sample:
            ids,Xqa,yqa=(sample["qa_"+key] for key in ("indices","X","errors"))
            assert len(ids)==8192 and np.all(np.diff(ids)>0) and np.all(gt[ids]<19)
            np.testing.assert_array_equal(yqa,Xqa[:,3]!=gt[ids])
            ref=model.predict_proba(Xqa)[:,1]
            if weather=="night":
                qa["scores"].append(ref)
                for key in ("errors","weights","tie_keys"):
                    qa[key].append(sample["qa_"+key])
        valid_count=err_count=hits=0
        for first,last,X,_ in features.iter_feature_blocks(logits,"cuda"):
            lo,hi=np.searchsorted(ids,(first,last))
            ix=ids[lo:hi]-first
            np.testing.assert_array_equal(X[ix],Xqa[lo:hi])
            target=gt[first:last]
            valid_ix=np.flatnonzero(target<19)
            if not valid_ix.size:
                assert hi==lo
                continue
            values=X[valid_ix]
            errors=values[:,3]!=target[valid_ix]
            tic=perf_counter()
            scores=gpu(values)
            score_seconds+=perf_counter()-tic
            qa_pos=np.searchsorted(valid_ix,ix)
            np.testing.assert_array_equal(valid_ix[qa_pos],ix)
            np.testing.assert_array_equal(scores[qa_pos],ref[lo:hi])
            if i==1 and first==0:
                np.testing.assert_array_equal(scores,model.predict_proba(values)[:,1])
            hits+=hi-lo
            acc.update(scores,errors)
            if weather=="night":
                mask=values[:,3]==8
                bins=histogram_indices(scores[mask])
                veg+=np.bincount(bins,minlength=HISTOGRAM_BINS)
                veg_err+=np.bincount(bins[errors[mask]],minlength=HISTOGRAM_BINS)
            valid_count+=len(scores)
            err_count+=int(errors.sum())
        assert (valid_count,err_count,hits)==(rec["num_valid"],rec["num_errors"],8192),row["image_id"]
        checked+=hits
        if i%50==0:
            print(f"Evaluate {group}: {i}/{len(rows)} images ({perf_counter()-start:.1f}s)",flush=True)
    metrics=acc.finalize(TARGETS)
    del metrics["risk_coverage"]
    frozen=read(BASE/"results/folds"/group/"result.json")["methods"]["raw_msp"]
    assert (metrics["num_pixels"],metrics["num_errors"])==(frozen["num_pixels"],frozen["num_errors"])
    assert metrics["fixed_threshold"]["complete_population"]
    qpts=[]
    if weather=="night":
        values={k:np.concatenate(v) for k,v in qa.items()}
        qpts=weighted_risk_coverage(values["scores"],values["errors"],values["weights"],values["tie_keys"],TARGETS)["checkpoints"]
    tmp=histogram.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,correct=acc.correct,incorrect=acc.incorrect,
                            **(dict(vegetation=veg,vegetation_errors=veg_err) if weather=="night" else {}))
    tmp.replace(histogram)
    write(output,dict(group=group,weather=weather,images=[r["image_id"] for r in rows],metrics=metrics,
          qa_raw_secondary=qpts,qa_fullstream_vs_sklearn_exact=checked,score_seconds=score_seconds,
          eval_seconds=perf_counter()-start,gpu=torch.cuda.get_device_name(0),model_sha256=sel["model_sha256"],
          manifest_sha256=manifest,histogram_sha256=sha256(histogram),scoring_source_sha256=scorer_hash))
    print(f"Evaluated {group} ({weather}): {len(rows)} images, {metrics['num_pixels']} pixels in {perf_counter()-start:.1f}s",flush=True)


def main():
    features.configure_torch("cuda")
    path=OUT/"run_manifest.json"
    seal=read(path)
    assert sha256(DATA/"features/manifest.json")==seal["frozen_sources"]["data/features/manifest.json"]
    records={r["image_id"]:r for r in read(DATA/"features/manifest.json")["records"]}
    groups=sys.argv[1:] or seal["outer_groups"]
    assert len(set(groups))==len(groups) and set(groups)<=set(seal["outer_groups"])
    scorer_hash={p:sha256(OUT/p) for p in ("gpu_forest.py","evaluate.py")}
    for group in groups:
        evaluate(group,records,sha256(path),scorer_hash)


if __name__=="__main__":
    main()
