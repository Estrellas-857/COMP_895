"""Train-only error classifiers and shared exact/sample or full-population metrics.

Features are MSP, normalized entropy, probability margin, and predicted class.
The positive outcome is a segmentation error; all scores increase with error.
This module performs no file I/O, feature extraction, split selection or tuning.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from experiments.paired_calibration_v1.metrics import weighted_risk_coverage


CONFIGS = ("raw_msp", "raw_entropy", "lr3", "lr3_class")
C_VALUES = (0.01, 0.1, 1.0)
TARGETS = (0.5, 0.6, 0.7, 0.8, 0.9)
CPU_THREADS = 4
HISTOGRAM_BINS = 100000


def _features(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 4 or not np.isfinite(X).all():
        raise ValueError("Features must be finite rows of MSP, entropy, margin, predicted class")
    if np.any(X[:, :3] < -1e-6) or np.any(X[:, :3] > 1 + 1e-6):
        raise ValueError("MSP, normalized entropy and probability margin must lie in [0,1]")
    if np.any(X[:, 3] != np.floor(X[:, 3])) or np.any((X[:, 3] < 0) | (X[:, 3] > 18)):
        raise ValueError("Predicted classes must be integer trainIds 0..18")
    return X


def _scores(scores):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector")
    if np.any(scores < -1e-6) or np.any(scores > 1 + 1e-6):
        raise ValueError("Error scores must lie in [0,1]")
    return np.clip(scores, 0, 1)


def _outcomes(errors, shape):
    errors = np.asarray(errors)
    if errors.shape != shape or not np.all((errors == 0) | (errors == 1)):
        raise ValueError("Errors must be same-length binary outcomes, with error=1")
    return errors


def _weights(weights, shape):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Weights must be same-length, finite and positive")
    return weights


def histogram_indices(scores, bins=HISTOGRAM_BINS):
    """Fixed arcsine bins preserve fine score resolution near both endpoints."""
    if int(bins) != bins or bins < 1:
        raise ValueError("bins must be a positive integer")
    scores = _scores(scores).astype(np.float64)
    transformed = (2 / np.pi) * np.arcsin(np.sqrt(scores))
    return np.minimum(np.floor(transformed * bins).astype(np.int64), int(bins) - 1)


def histogram_upper_edges(bin_indices, bins=HISTOGRAM_BINS):
    """Inverse-transform upper bin edges into the original error-score scale."""
    if int(bins) != bins or bins < 1:
        raise ValueError("bins must be a positive integer")
    indices = np.asarray(bin_indices)
    if (indices.ndim != 1 or not np.isfinite(indices).all()
            or np.any(indices != np.floor(indices)) or np.any((indices < 0) | (indices >= bins))):
        raise ValueError("Bin indices must be integers in 0..bins-1")
    return np.sin((np.pi / 2) * (indices.astype(np.float64) + 1) / bins) ** 2


def fit_pipeline(config, C, X, y, w):
    """Fit one training partition; weights have mean one, scaling is unweighted.

    Fixed one-hot categories describe the known 19-class taxonomy, not observed
    held-out data. Nonconvergence is a failed fit, never a successful candidate.
    """
    if config not in ("lr3", "lr3_class") or C not in C_VALUES:
        raise ValueError("Expected lr3/lr3_class and a preregistered C value")
    X = _features(X)
    y = _outcomes(y, (len(X),))
    w = _weights(w, y.shape)
    if not len(X) or len(np.unique(y)) != 2:
        raise ValueError("Training requires nonempty data with both correct and error outcomes")
    transforms = [("continuous", StandardScaler(), [0, 1, 2])]
    if config == "lr3_class":
        transforms.append(("class", OneHotEncoder(categories=[np.arange(19)],
            handle_unknown="ignore", sparse_output=False), [3]))
    model = Pipeline([
        ("features", ColumnTransformer(transforms, remainder="drop")),
        ("classifier", LogisticRegression(C=C, penalty="l2", solver="lbfgs",
            max_iter=1000, tol=1e-5)),
    ])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=CPU_THREADS):
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y, classifier__sample_weight=w / w.mean())
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError(f"{config}, C={C}: lbfgs did not converge within 1000 iterations")
    model.diagnostics_ = dict(config=config, C=float(C), num_training_pixels=len(X),
        n_iter=int(model.named_steps["classifier"].n_iter_.max()),
        converged=True, sample_weight_normalization="mean one within this fit",
        scaler="unweighted mean and variance fitted only on this training partition",
        cpu_threads=CPU_THREADS)
    return model


def predict_scores(config, X, model=None):
    """Return error-oriented scores without labels or evaluation-set fitting."""
    X = _features(X)
    if config == "raw_msp":
        return _scores(1.0 - X[:, 0])
    if config == "raw_entropy":
        return _scores(X[:, 1])
    if config not in ("lr3", "lr3_class") or model is None:
        raise ValueError("A fitted classifier is required for learned error scores")
    if model.diagnostics_["config"] != config:
        raise ValueError("Classifier configuration does not match requested scores")
    if len(X) == 0:
        return np.empty(0, dtype=np.float64)
    with threadpool_limits(limits=CPU_THREADS):
        return _scores(model.predict_proba(X)[:, 1])


def training_threshold(scores, weights, coverage=0.8):
    """Training-only weighted quantile; deployment accepts score <= threshold.

    Ties can increase coverage beyond the target. No outcome labels are used.
    """
    scores = _scores(scores)
    weights = _weights(weights, scores.shape)
    if not scores.size or not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("A nonempty population and coverage in (0,1] are required")
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, coverage * cumulative[-1])), len(order) - 1)
    return float(scores[order[index]])


def exact_metrics(scores, errors, weights, tie_keys, targets=TARGETS):
    """Exact weighted score ranks on the shared sample; AUROC/AP group ties.

    RC uses the existing label-independent secondary tie keys. These are sample
    estimates of population metrics, not full-pixel measurements.
    """
    scores = _scores(scores)
    errors = _outcomes(errors, scores.shape)
    weights = _weights(weights, scores.shape)
    curve = weighted_risk_coverage(scores, errors, weights, tie_keys, targets)
    positive = float(np.sum(weights * errors))
    total = float(weights.sum())
    return dict(num_pixels=len(scores), total_weight=total,
        error_prevalence=positive / total if total else None,
        auroc=float(roc_auc_score(errors, scores, sample_weight=weights))
        if 0 < positive < total else None,
        average_precision=float(average_precision_score(errors, scores, sample_weight=weights))
        if positive else None,
        aurc=curve["aurc"], checkpoints=curve["checkpoints"], risk_coverage=curve,
        method="shared sample; weighted exact AUROC/AP with score ties grouped; weighted sampled RC")


class HistogramAccumulator:
    """Full semantic-valid population, 100,000 fixed arcsine bins by default.

    Transform u=2/pi*arcsin(sqrt(score)) in float64 and use equal-width u bins.
    Bins are [lower, upper) in u, with score=1 included in the final bin.
    AUROC/AP treat a bin as tied; RC accepts whole bins and uses right-step AURC.
    They approximate ranking, while prevalence and supplied-threshold counts are
    exact. Merging preserves exact counts for fold-specific training thresholds.
    """

    def __init__(self, bins=HISTOGRAM_BINS, threshold=None):
        if int(bins) != bins or bins < 1:
            raise ValueError("bins must be a positive integer")
        if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("threshold must lie in [0,1]")
        self.bins, self.threshold = int(bins), threshold
        self.correct = np.zeros(self.bins, dtype=np.int64)
        self.incorrect = np.zeros(self.bins, dtype=np.int64)
        self.threshold_evaluated = self.accepted = self.accepted_errors = 0
        self.thresholds = set()

    def update(self, scores, errors):
        scores = _scores(scores)
        errors = _outcomes(errors, scores.shape).astype(bool, copy=False)
        indices = histogram_indices(scores, self.bins)
        self.correct += np.bincount(indices[~errors], minlength=self.bins)
        self.incorrect += np.bincount(indices[errors], minlength=self.bins)
        if self.threshold is not None:
            accepted = scores <= self.threshold
            self.threshold_evaluated += len(scores)
            self.accepted += int(accepted.sum())
            self.accepted_errors += int(np.count_nonzero(accepted & errors))
            self.thresholds.add(float(self.threshold))
        return self

    def merge(self, other):
        if not isinstance(other, HistogramAccumulator) or self.bins != other.bins:
            raise ValueError("Only accumulators with identical score bins can merge")
        self.correct += other.correct
        self.incorrect += other.incorrect
        self.threshold_evaluated += other.threshold_evaluated
        self.accepted += other.accepted
        self.accepted_errors += other.accepted_errors
        self.thresholds.update(other.thresholds)
        return self

    def finalize(self, targets=TARGETS):
        targets = tuple(float(t) for t in targets)
        if any(not np.isfinite(t) or not 0 < t <= 1 for t in targets):
            raise ValueError("Coverage targets must lie in (0,1]")
        p, n = self.incorrect.astype(np.float64), self.correct.astype(np.float64)
        positives, negatives = int(self.incorrect.sum()), int(self.correct.sum())
        total = positives + negatives
        auroc = float(np.sum(p * (np.cumsum(n) - n / 2)) / (positives * negatives)) \
            if positives and negatives else None
        tp, accepted_descending = np.cumsum(p[::-1]), np.cumsum((p + n)[::-1])
        precision = np.divide(tp, accepted_descending, out=np.zeros_like(tp), where=accepted_descending > 0)
        ap = float(np.sum(p[::-1] * precision) / positives) if positives else None
        occupied = p + n > 0
        accepted = np.cumsum(p + n)[occupied]
        wrong = np.cumsum(p)[occupied]
        coverage = accepted / total if total else np.empty(0)
        risk = wrong / accepted
        aurc = float(np.sum(np.diff(np.r_[0.0, coverage]) * risk)) if total else None
        checkpoints = []
        for target in targets:
            i = int(np.argmin(np.abs(coverage - target))) if total else None
            checkpoints.append(dict(requested_coverage=target,
                actual_coverage=float(coverage[i]) if total else None,
                coverage_error=float(coverage[i] - target) if total else None,
                risk=float(risk[i]) if total else None,
                accepted_pixels=int(accepted[i]) if total else 0,
                accepted_errors=int(wrong[i]) if total else 0))
        threshold_result = dict(num_evaluated=self.threshold_evaluated,
            threshold=next(iter(self.thresholds)) if len(self.thresholds) == 1 else None,
            distinct_thresholds=len(self.thresholds), accepted_pixels=self.accepted,
            accepted_errors=self.accepted_errors,
            coverage=self.accepted / self.threshold_evaluated if self.threshold_evaluated else None,
            risk=self.accepted_errors / self.accepted if self.accepted else None,
            complete_population=self.threshold_evaluated == total)
        return dict(num_pixels=total, num_errors=positives,
            error_prevalence=positives / total if total else None,
            auroc=auroc, average_precision=ap, aurc=aurc, checkpoints=checkpoints,
            fixed_threshold=threshold_result,
            risk_coverage=dict(coverage=coverage, risk=risk, accepted=accepted,
                errors=wrong, bin_index=np.flatnonzero(occupied),
                threshold_upper=histogram_upper_edges(np.flatnonzero(occupied), self.bins)),
            method=f"full valid-pixel population; {self.bins} fixed bins in float64 u=2/pi*arcsin(sqrt(score)); tied-bin AUROC/AP; whole-bin RC; right-step AURC")
