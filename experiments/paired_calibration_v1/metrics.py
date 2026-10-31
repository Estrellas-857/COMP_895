"""Streaming metrics and a shared-sample risk--coverage estimator.

The semantic population is GT trainIds 0..18, optionally intersected with a
region mask. An ambiguity-region mask never replaces the semantic validity
mask. Every image is evaluated against its own GT; no paired pixel alignment
is assumed. NLL values must be supplied from stable log-softmax, not clipped
probabilities. This module performs no file reads, writes, or model inference.
"""
from __future__ import annotations

import hashlib

import numpy as np


GLOBAL_THRESHOLD = 0.386413580715649


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


class MetricAccumulator:
    """Pixel-weighted metrics with explicit counts and undefined ratios.

    ``coverage_all`` uses all region pixels, including ignored semantic GT;
    all other error/calibration statistics use valid semantic pixels only.
    Acceptance is ``1 - confidence <= tau``. High confidence is ``>= highconf``.
    Canonical unscaled-logit predictions are supplied by the caller.
    """

    _SCALARS = (
        "num_images", "num_all", "num_accepted_all", "num_valid", "num_correct",
        "num_accepted", "num_accepted_errors", "num_high_conf",
        "num_high_conf_errors", "confidence_sum", "nll_sum",
    )
    _ARRAYS = (
        "bin_count", "bin_confidence_sum", "bin_correct", "confusion",
        "pred_count", "pred_accepted", "pred_accepted_errors", "pred_high_conf",
        "pred_high_conf_errors", "gt_count", "gt_accepted", "gt_accepted_errors",
        "gt_high_conf", "gt_high_conf_errors",
    )

    def __init__(self, num_classes=19, bins=15, tau=GLOBAL_THRESHOLD, highconf=.9):
        if int(num_classes) != num_classes or not 1 <= num_classes <= 255:
            raise ValueError("num_classes must be an integer in 1..255")
        if int(bins) != bins or bins < 1:
            raise ValueError("bins must be a positive integer")
        if not np.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError("tau must lie in [0,1]")
        if not np.isfinite(highconf) or not 0 <= highconf <= 1:
            raise ValueError("highconf must lie in [0,1]")
        self.num_classes, self.bins = int(num_classes), int(bins)
        self.tau, self.highconf = float(tau), float(highconf)
        self.bin_edges = np.linspace(0, 1, self.bins + 1)
        for key in self._SCALARS:
            setattr(self, key, 0.0 if key.endswith("_sum") else 0)
        self.bin_count = np.zeros(self.bins, dtype=np.int64)
        self.bin_confidence_sum = np.zeros(self.bins, dtype=np.float64)
        self.bin_correct = np.zeros(self.bins, dtype=np.int64)
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for key in self._ARRAYS[4:]:
            setattr(self, key, np.zeros(self.num_classes, dtype=np.int64))

    def update(self, confidence, nll_map, prediction, gt, region_mask=None):
        confidence, nll_map = np.asarray(confidence), np.asarray(nll_map)
        prediction, gt = np.asarray(prediction), np.asarray(gt)
        if gt.ndim != 2 or any(a.shape != gt.shape for a in (confidence, nll_map, prediction)):
            raise ValueError("Expected same-shape 2D confidence, NLL, prediction, and GT")
        if not np.issubdtype(gt.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
            raise ValueError("Prediction and GT must contain integer class IDs")
        region = np.ones(gt.shape, dtype=bool) if region_mask is None else np.asarray(region_mask)
        if region.shape != gt.shape or region.dtype != np.bool_:
            raise ValueError("region_mask must be a same-shape boolean array")
        if np.any((prediction[region] < 0) | (prediction[region] >= self.num_classes)):
            raise ValueError("Prediction contains a class outside the canonical range")
        if not np.isfinite(confidence[region]).all() or np.any((confidence[region] < 0) | (confidence[region] > 1)):
            raise ValueError("Region confidence must be finite and in [0,1]")
        semantic_valid = (gt >= 0) & (gt < self.num_classes)
        if np.any(region & ~(semantic_valid | (gt == 255))):
            raise ValueError("Unexpected GT ID; only class IDs and ignore=255 are supported")
        valid = semantic_valid & region
        if not np.isfinite(nll_map[valid]).all() or np.any(nll_map[valid] < 0):
            raise ValueError("Valid-pixel NLL must be finite and nonnegative")
        accepted = (1.0 - confidence) <= self.tau
        self.num_images += 1
        self.num_all += int(region.sum())
        self.num_accepted_all += int((region & accepted).sum())
        c = confidence[valid]
        p, y = prediction[valid].astype(np.int64), gt[valid].astype(np.int64)
        correct = p == y
        a, h = accepted[valid], c >= self.highconf
        self.num_valid += int(c.size)
        self.num_correct += int(correct.sum())
        self.num_accepted += int(a.sum())
        self.num_accepted_errors += int((a & ~correct).sum())
        self.num_high_conf += int(h.sum())
        self.num_high_conf_errors += int((h & ~correct).sum())
        self.confidence_sum += float(c.sum(dtype=np.float64))
        self.nll_sum += float(nll_map[valid].sum(dtype=np.float64))
        # Exactly the frozen 15-bin convention: interior edges go right, 1 last.
        b = np.searchsorted(self.bin_edges[1:-1], c.astype(np.float64), side="right")
        self.bin_count += np.bincount(b, minlength=self.bins)
        self.bin_confidence_sum += np.bincount(b, weights=c, minlength=self.bins)
        self.bin_correct += np.bincount(b[correct], minlength=self.bins)
        self.confusion += np.bincount(y * self.num_classes + p,
                                     minlength=self.num_classes ** 2).reshape(self.confusion.shape)
        for prefix, labels in (("pred", p), ("gt", y)):
            for suffix, mask in (("count", np.ones(c.size, dtype=bool)), ("accepted", a),
                                 ("accepted_errors", a & ~correct), ("high_conf", h),
                                 ("high_conf_errors", h & ~correct)):
                getattr(self, prefix + "_" + suffix)[:] += np.bincount(labels[mask], minlength=self.num_classes)
        return self

    def merge(self, other):
        """Pool sufficient statistics, never average image-level metrics."""
        if not isinstance(other, MetricAccumulator) or (
            self.num_classes, self.bins, self.tau, self.highconf
        ) != (other.num_classes, other.bins, other.tau, other.highconf):
            raise ValueError("Cannot merge accumulators with different definitions")
        if other is self:
            raise ValueError("Cannot merge an accumulator with itself")
        for key in self._SCALARS:
            setattr(self, key, getattr(self, key) + getattr(other, key))
        for key in self._ARRAYS:
            getattr(self, key)[:] += getattr(other, key)
        return self

    def finalize(self):
        errors = self.num_valid - self.num_correct
        mean_confidence = _ratio(self.confidence_sum, self.num_valid)
        accuracy = _ratio(self.num_correct, self.num_valid)
        ece = (_ratio(np.abs(self.bin_confidence_sum - self.bin_correct).sum(), self.num_valid))
        return dict(
            num_images=self.num_images, num_all=self.num_all, num_valid=self.num_valid,
            num_ignored=self.num_all-self.num_valid, num_correct=self.num_correct,
            num_errors=errors, num_accepted_all=self.num_accepted_all,
            num_rejected_all=self.num_all-self.num_accepted_all,
            num_accepted=self.num_accepted, num_rejected=self.num_valid-self.num_accepted,
            num_accepted_errors=self.num_accepted_errors,
            num_accepted_correct=self.num_accepted-self.num_accepted_errors,
            num_rejected_errors=errors-self.num_accepted_errors,
            num_high_conf=self.num_high_conf, num_high_conf_errors=self.num_high_conf_errors,
            accuracy=accuracy, baseline_risk=_ratio(errors, self.num_valid),
            mean_confidence=mean_confidence, nll=_ratio(self.nll_sum, self.num_valid), ece=ece,
            signed_gap=mean_confidence-accuracy if self.num_valid else None,
            coverage_valid=_ratio(self.num_accepted, self.num_valid),
            coverage_all=_ratio(self.num_accepted_all, self.num_all),
            accepted_risk=_ratio(self.num_accepted_errors, self.num_accepted),
            high_conf_coverage=_ratio(self.num_high_conf, self.num_valid),
            high_conf_error_rate=_ratio(self.num_high_conf_errors, self.num_high_conf),
            error_high_conf_fraction=_ratio(self.num_high_conf_errors, errors),
            high_conf_error_fraction_all_valid=_ratio(self.num_high_conf_errors, self.num_valid),
            error_recall=_ratio(errors-self.num_accepted_errors, errors),
            rejection_precision=_ratio(errors-self.num_accepted_errors, self.num_valid-self.num_accepted),
            tau=self.tau, high_conf_threshold=self.highconf, num_bins=self.bins,
        )

    def confidence_bin_rows(self):
        return [dict(bin_index=i, lower=float(self.bin_edges[i]), upper=float(self.bin_edges[i+1]),
                     count=int(self.bin_count[i]), correct=int(self.bin_correct[i]),
                     mean_confidence=_ratio(self.bin_confidence_sum[i], self.bin_count[i]),
                     accuracy=_ratio(self.bin_correct[i], self.bin_count[i]),
                     fraction=_ratio(self.bin_count[i], self.num_valid)) for i in range(self.bins)]

    def class_rows(self):
        """All classes, with predicted- and GT-class populations kept separate.

        ``gt_retained_correct_fraction`` divides accepted correct class pixels
        by every valid GT pixel of that class; it includes abstentions as lost
        coverage. It must not be confused with accuracy on the accepted subset.
        """
        rows = []
        for j in range(self.num_classes):
            row = dict(class_id=j, num_correct=int(self.confusion[j, j]))
            for prefix in ("pred", "gt"):
                n, a, e, h, he = (int(getattr(self, prefix+"_"+s)[j]) for s in
                                  ("count", "accepted", "accepted_errors", "high_conf", "high_conf_errors"))
                row.update({prefix+"_num_valid": n, prefix+"_num_errors": n-int(self.confusion[j,j]),
                            prefix+"_num_accepted": a, prefix+"_num_accepted_errors": e,
                            prefix+"_coverage": _ratio(a,n), prefix+"_accepted_risk": _ratio(e,a),
                            prefix+"_num_high_conf": h, prefix+"_num_high_conf_errors": he,
                            prefix+"_high_conf_error_rate": _ratio(he,h),
                            prefix+"_error_high_conf_fraction": _ratio(he,n-int(self.confusion[j,j]))})
            row["gt_recall"] = _ratio(self.confusion[j,j], self.gt_count[j])
            row["pred_precision"] = _ratio(self.confusion[j,j], self.pred_count[j])
            row["gt_retained_correct_fraction"] = _ratio(self.gt_accepted[j]-self.gt_accepted_errors[j], self.gt_count[j])
            rows.append(row)
        return rows


def sample_indices(gt, image_id, seed=42, n=8192):
    """Deterministically sample valid flat pixel indices without replacement.

    Sampling depends on image identity, the semantic validity mask, and seed,
    never on correctness, predicted class, semantic class, or temperature.
    Each sampled pixel has weight N_valid / n_sample. Independent deterministic
    uint64 tie keys are reused across methods; exact key collisions fall back
    to stable input order (manifest image order then returned sample order).
    """
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.issubdtype(gt.dtype, np.integer):
        raise ValueError("GT must be a 2D integer array")
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a nonempty stable identifier")
    valid = np.flatnonzero((gt >= 0) & (gt < 19))
    count = min(int(n), valid.size)
    digest = hashlib.sha256((str(seed)+"\0"+image_id).encode("utf-8")).digest()
    selection_seed, tie_seed = np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")).spawn(2)
    rng = np.random.default_rng(selection_seed)
    indices = np.sort(rng.choice(valid, size=count, replace=False)) if count else np.empty(0,dtype=np.int64)
    tie_rng = np.random.default_rng(tie_seed)
    tie_keys = tie_rng.bit_generator.random_raw(count)
    return dict(indices=indices, weights=np.full(count, valid.size/count if count else 0,dtype=np.float64),
                tie_keys=tie_keys, num_valid=int(valid.size))


def weighted_risk_coverage(scores, errors, weights, tie_keys, targets=(.5,.6,.7,.8,.9)):
    """Weighted sampled RC and right-step AURC, low score accepted first.

    Sort by score then the shared label-independent tie key. Stable input order
    resolves the exceptional duplicate score/key pair. Whole sampled pixels
    are accepted; checkpoints are the nearest observed coverage, ties choosing
    the earlier (lower-coverage) point. Actual coverage and its error are always
    returned. There is no fractional boundary label or fabricated zero-risk
    point. AURC sums each coverage increment times its observed endpoint risk.
    This is a sample estimator, not an exact full-population ranking metric.
    """
    scores, errors, weights, tie_keys = map(np.asarray, (scores, errors, weights, tie_keys))
    if scores.ndim != 1 or any(a.shape != scores.shape for a in (errors,weights,tie_keys)):
        raise ValueError("Scores/errors/weights/tie_keys must be same-length vectors")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Scores must be finite; weights must be finite and positive")
    if not np.all((errors == 0) | (errors == 1)):
        raise ValueError("Errors must be boolean or binary")
    if not np.issubdtype(tie_keys.dtype, np.integer):
        raise ValueError("Tie keys must be integer labels independent of correctness")
    targets = tuple(float(t) for t in targets)
    if any(not np.isfinite(t) or not 0 < t <= 1 for t in targets):
        raise ValueError("Target coverages must lie in (0,1]")
    order = np.lexsort((tie_keys,scores))
    accepted = np.cumsum(weights[order], dtype=np.float64)
    wrong = np.cumsum(weights[order] * errors[order], dtype=np.float64)
    coverage = accepted/accepted[-1] if accepted.size else np.empty(0,dtype=np.float64)
    risk = wrong/accepted if accepted.size else np.empty(0,dtype=np.float64)
    aurc = float(np.sum(np.diff(np.r_[0.0,coverage])*risk)) if accepted.size else None
    checkpoints = []
    for target in targets:
        i = int(np.argmin(np.abs(coverage-target))) if accepted.size else None
        checkpoints.append(dict(requested_coverage=target,
            actual_coverage=float(coverage[i]) if i is not None else None,
            coverage_error=float(coverage[i]-target) if i is not None else None,
            risk=float(risk[i]) if i is not None else None,
            accepted_weight=float(accepted[i]) if i is not None else None,
            error_weight=float(wrong[i]) if i is not None else None))
    return dict(coverage=coverage,risk=risk,accepted_weight=accepted,error_weight=wrong,
                checkpoints=checkpoints,aurc=aurc,num_samples=int(scores.size),
                total_weight=float(accepted[-1]) if accepted.size else 0.0,
                method="shared weighted sample; score/tie-key order; nearest observed coverage; right-step AURC")
