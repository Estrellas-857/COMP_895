"""Fit only the interaction model, using the frozen group splits and samples."""
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from experiments.cse805_error_prediction_v1.evaluation import C_VALUES, training_threshold
from experiments.cse805_error_prediction_v1.interpretation_v1.shared import ROWS
from experiments.cse805_error_prediction_v1.run import load_training
from .model import ClassMSPDesign, DATA, OUT, read, score_rows, sha256, write


def fit_classifier(design, outcomes, weights, C):
    if len(np.unique(outcomes)) != 2 or not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("Expected two outcomes and finite positive training weights")
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, tol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        lr.fit(design, outcomes, sample_weight=weights / weights.mean())
    return lr


def fit_fold(fold, arrays, manifest):
    X, y, weights, groups = arrays
    group = fold["eval_group"]
    folder = OUT / "folds" / group
    path = folder / "selection.json"
    model_path = folder / "interaction.joblib"
    if path.exists():
        rec = read(path)
        if rec["manifest_sha256"] != manifest or rec["model_sha256"] != sha256(model_path):
            raise RuntimeError(f"Stale model {group}")
        return
    started = time.perf_counter()
    candidates = {C:[] for C in C_VALUES}
    for inner in fold["inner_folds"]:
        fit = np.isin(groups, inner["fit_groups"])
        dev = np.isin(groups, inner["dev_groups"])
        assert not np.any(fit & dev) and not np.any((fit | dev) & (groups == group))
        pre = ClassMSPDesign().fit(X[fit])
        A, B = pre.transform(X[fit]), pre.transform(X[dev])
        assert len(np.unique(y[dev])) == 2
        for C in C_VALUES:
            try:
                lr = fit_classifier(A, y[fit], weights[fit], C)
                ap = average_precision_score(y[dev], lr.predict_proba(B)[:, 1], sample_weight=weights[dev])
                result = dict(status="PASS", ap=float(ap), n_iter=int(lr.n_iter_[0]))
            except (ConvergenceWarning, ValueError) as exc:
                result = dict(status="FAILED", reason=str(exc))
            candidates[C].append(dict(inner_fold=inner["inner_fold"], **result))
    means = {C:float(np.mean([r["ap"] for r in records])) for C, records in candidates.items()
             if len(records) == 3 and all(r["status"] == "PASS" for r in records)}
    if not means:
        write(folder/"failed.json", candidates)
        raise RuntimeError(f"No converged candidate for {group}")
    chosen = min(C for C, ap in means.items() if max(means.values())-ap <= 1e-12)
    fit = np.isin(groups, fold["train_groups"])
    assert np.array_equal(fit, groups != group)
    train_X, w = X[fit], weights[fit]
    feature = ClassMSPDesign().fit(train_X)
    lr = fit_classifier(feature.transform(train_X), y[fit], w, chosen)
    model = Pipeline([("features", feature), ("classifier", lr)])
    scores = model.predict_proba(train_X)[:, 1]
    np.testing.assert_array_equal(score_rows(model, train_X), scores)
    theta = training_threshold(scores, w)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp")
    joblib.dump(model, tmp)
    tmp.replace(model_path)
    write(path, dict(group=group, train_groups=fold["train_groups"], inner_folds=fold["inner_folds"],
          candidates=candidates, mean_weighted_inner_AP=means, selected_C=chosen, threshold=theta,
          achieved_training_coverage=w[scores<=theta].sum()/w.sum(), n_train=len(train_X),
          n_iter=int(lr.n_iter_[0]), model_sha256=sha256(model_path), manifest_sha256=manifest,
          seconds=time.perf_counter()-started))
    print(f"Fit {group}: C={chosen}, {len(train_X)} training pixels, {time.perf_counter()-started:.1f}s", flush=True)


def main():
    seal = read(OUT/"run_manifest.json")
    assert seal["model_sha256"] == sha256(OUT/"model.py") and seal["train_sha256"] == sha256(__file__)
    assert seal["analysis_plan_sha256"] == sha256(OUT/"protocol.json")
    digest = sha256(OUT/"run_manifest.json")
    arrays = load_training(ROWS)
    assert len(arrays[0]) == 1600 * 512
    folds = read(DATA/"assignments.json")
    requested = set(sys.argv[1:] or [f["eval_group"] for f in folds])
    if requested - {f["eval_group"] for f in folds}:
        raise ValueError("Unknown group")
    for fold in folds:
        if fold["eval_group"] in requested:
            fit_fold(fold, arrays, digest)


if __name__ == "__main__":
    main()
