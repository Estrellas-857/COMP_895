"""Fixed taxonomy one-hot encoding and a single class-by-MSP interaction."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from experiments.cse805_error_prediction_v1.common import ROOT as BASE, json_safe, sha256
from experiments.cse805_error_prediction_v1.evaluation import _features

OUT = Path(__file__).resolve().parent
DATA = BASE / "data"
METHOD = "lr_class_msp_interaction"
OLD = "lr3_class"
MSP = "raw_msp"
TARGETS = (.01, .05, .10, .20, .50, .60, .70, .80, .90)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT):
        raise ValueError("Output outside the interaction experiment")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class ClassMSPDesign(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(_features(X)[:, :3])
        self.n_features_in_ = 4
        return self

    def transform(self, X):
        X = _features(X)
        z = self.scaler_.transform(X[:, :3])
        cls = X[:, 3].astype(np.int32)
        # Only five entries per row: global features, active class, active interaction.
        col = np.empty((len(X), 5), dtype=np.int32)
        col[:, :3] = (0, 1, 2)
        col[:, 3], col[:, 4] = 3 + cls, 22 + cls
        values = np.empty((len(X), 5), dtype=np.float64)
        values[:, :3], values[:, 3], values[:, 4] = z, 1., z[:, 0]
        return csr_matrix((values.ravel(), col.ravel(), np.arange(0, len(X)*5+1, 5, dtype=np.int32)),
                          shape=(len(X), 41))


def score_rows(model, X):
    """Same CSR dot-product order and sigmoid, without materializing 5 entries/pixel."""
    z = model.named_steps["features"].scaler_.transform(X[:, :3])
    cls = X[:, 3].astype(np.intp)
    lr = model.named_steps["classifier"]
    w = lr.coef_[0]
    logit = np.multiply(z[:, 0], w[0], dtype=np.float64)
    logit += np.multiply(z[:, 1], w[1], dtype=np.float64)
    logit += np.multiply(z[:, 2], w[2], dtype=np.float64)
    logit += w[3 + cls]
    logit += z[:, 0] * w[22 + cls]
    logit += lr.intercept_[0]
    return expit(logit)
