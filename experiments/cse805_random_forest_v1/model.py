"""The categorical design and the bounded forest used in all RF folds."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from experiments.cse805_error_prediction_v1.common import ROOT as BASE, json_safe, sha256
from experiments.cse805_error_prediction_v1.evaluation import _features
from experiments.cse805_class_msp_interaction_v1.model import OUT as INTERACTION, METHOD as LR_INTERACTION, read, TARGETS

OUT = Path(__file__).resolve().parent
DATA = BASE/"data"
RF = "rf_class"
METHODS = ("raw_msp", "raw_entropy", "lr3", "lr3_class", LR_INTERACTION, RF)
LABELS = dict(zip(METHODS, ("MSP", "Entropy", "LR3", "LR + Class", "LR + Class × MSP", "RF + Class")))
OPTIONS = read(OUT/"protocol.json")["rf"]


def write(path, record):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT):
        raise ValueError("Output outside RF extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(json_safe(record), indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def encode(X):
    X = _features(X)
    design = np.zeros((len(X),22), dtype=np.float32)
    design[:,:3] = X[:,:3]
    design[np.arange(len(X)),3+X[:,3].astype(np.intp)] = 1
    return design


def fit_forest(X, y, w, depth):
    if depth not in OPTIONS["candidate_depth"] or not np.array_equal(np.unique(y),[0,1]):
        raise ValueError("Unsupported depth or fit missing an outcome")
    if len(X)!=len(y) or w.shape!=y.shape or not np.all(np.isfinite(w) & (w>0)):
        raise ValueError("Training features, outcomes or weights disagree")
    params = {k:v for k,v in OPTIONS.items() if k not in ("candidate_depth","fit_threads")}
    forest = RandomForestClassifier(**params,max_depth=depth,n_jobs=OPTIONS["fit_threads"])
    model = Pipeline([("features",FunctionTransformer(encode)),("forest",forest)])
    model.fit(X,y,forest__sample_weight=w/w.mean())
    # Preserve floating-point tie order when scores are averaged across trees.
    forest.n_jobs=1
    if not np.array_equal(forest.classes_,[0,1]):
        raise ValueError("Expected the second score column to correspond to an error")
    return model
