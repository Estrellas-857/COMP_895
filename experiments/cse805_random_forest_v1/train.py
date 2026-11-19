"""Grouped inner AP selection and outer-training refits of the RF."""
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from experiments.cse805_error_prediction_v1.evaluation import training_threshold
from experiments.cse805_error_prediction_v1.interpretation_v1.shared import ROWS
from experiments.cse805_error_prediction_v1.run import load_training
from .model import DATA, OUT, OPTIONS, fit_forest, read, sha256, write


def fit_fold(fold, arrays, manifest_hash):
    group = fold["eval_group"]
    folder = OUT/"folds"/group
    model_path = folder/"forest.joblib"
    record_path = folder/"selection.json"
    if record_path.exists():
        rec=read(record_path)
        if rec["manifest_sha256"]!=manifest_hash or rec["model_sha256"]!=sha256(model_path):
            raise RuntimeError("Stale forest for "+group)
        return
    X,y,w,groups=arrays
    start=time.perf_counter()
    records=[]
    for inner in fold["inner_folds"]:
        fit=np.isin(groups,inner["fit_groups"])
        dev=np.isin(groups,inner["dev_groups"])
        assert not np.any(fit & dev) and not np.any((fit | dev) & (groups==group))
        assert np.array_equal(np.unique(y[dev]),[0,1])
        for depth in OPTIONS["candidate_depth"]:
            before=time.perf_counter()
            model=fit_forest(X[fit],y[fit],w[fit],depth)
            p=model.predict_proba(X[dev])[:,1]
            ap=float(average_precision_score(y[dev],p,sample_weight=w[dev]))
            records.append(dict(inner=inner["inner_fold"],depth=depth,weighted_AP=ap,seconds=time.perf_counter()-before))
    means={d:float(np.mean([r["weighted_AP"] for r in records if r["depth"]==d])) for d in OPTIONS["candidate_depth"]}
    selected=min(d for d,s in means.items() if max(means.values())-s<=1e-12)
    fit=np.isin(groups,fold["train_groups"])
    assert np.array_equal(fit,groups!=group)
    model=fit_forest(X[fit],y[fit],w[fit],selected)
    scores=model.predict_proba(X[fit])[:,1]
    theta=training_threshold(scores,w[fit])
    forest=model["forest"]
    folder.mkdir(parents=True,exist_ok=True)
    tmp=model_path.with_suffix(".tmp")
    joblib.dump(model,tmp,compress=3)
    tmp.replace(model_path)
    write(record_path,dict(group=group,train_groups=fold["train_groups"],inner_folds=fold["inner_folds"],
          n_train=int(fit.sum()),candidates=records,mean_inner_weighted_AP=means,selected_depth=selected,
          threshold=theta,training_coverage=float(w[fit][scores<=theta].sum()/w[fit].sum()),
          bootstrap_draws_per_tree=int(forest._n_samples_bootstrap),trees=len(forest.estimators_),
          leaf_counts=[t.get_n_leaves() for t in forest.estimators_],model_sha256=sha256(model_path),
          manifest_sha256=manifest_hash,seconds=time.perf_counter()-start))
    print(f"Fit {group}: depth {selected}, {fit.sum()} outer training pixels ({time.perf_counter()-start:.1f}s)",flush=True)


def main():
    manifest=OUT/"run_manifest.json"
    seal=read(manifest)
    for name,digest in seal["rf_source_sha256"].items():
        assert sha256(OUT/name)==digest,name
    arrays=load_training(ROWS)
    assert len(arrays[0])==1600*512
    folds=read(DATA/"assignments.json")
    requested=set(sys.argv[1:] or seal["outer_groups"])
    if not requested<=set(seal["outer_groups"]):
        raise ValueError("Unknown outer group")
    for fold in folds:
        if fold["eval_group"] in requested:
            fit_fold(fold,arrays,sha256(manifest))


if __name__=="__main__":
    main()
