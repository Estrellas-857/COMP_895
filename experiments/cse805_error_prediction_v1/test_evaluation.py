"""Synthetic checks only: no ACDC data or segmentation model is loaded."""
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

from .evaluation import (HistogramAccumulator, exact_metrics, fit_pipeline,
    predict_scores, training_threshold, histogram_indices, histogram_upper_edges,
    HISTOGRAM_BINS)


class EvaluationTests(unittest.TestCase):
    def test_arcsine_bins_endpoints_midpoints_and_inverse_bounds(self):
        self.assertEqual(HISTOGRAM_BINS, 100000)
        np.testing.assert_array_equal(histogram_indices(np.array([0, 1], dtype=np.float32)),
                                      [0, HISTOGRAM_BINS - 1])
        bins = 64
        indices = np.arange(bins)
        midpoints = np.sin(np.pi * (indices + 0.5) / (2 * bins)) ** 2
        np.testing.assert_array_equal(histogram_indices(midpoints, bins), indices)
        upper = histogram_upper_edges(indices, bins)
        lower = np.r_[0.0, upper[:-1]]
        self.assertTrue(np.all((midpoints > lower) & (midpoints < upper)))
        self.assertEqual(upper[-1], 1)
        self.assertAlmostEqual(upper[bins // 2 - 1], 0.5)
        self.assertLess(upper[0], 1 / bins)
        self.assertAlmostEqual(upper[0], 1 - lower[-1])

    def test_random_float32_histogram_matches_grouped_score_metrics(self):
        rng = np.random.default_rng(805)
        scores = np.r_[0., 1., rng.beta(0.15, 0.15, 4096)].astype(np.float32)
        y = rng.integers(0, 2, size=len(scores))
        bins = 64
        ids = histogram_indices(scores, bins)
        result = HistogramAccumulator(bins=bins).update(scores, y).finalize()
        self.assertAlmostEqual(result["auroc"], roc_auc_score(y, ids))
        self.assertAlmostEqual(result["average_precision"], average_precision_score(y, ids))
        order = np.argsort(scores, kind="stable")
        self.assertTrue(np.all(np.diff(ids[order]) >= 0))
        upper = histogram_upper_edges(ids, bins)
        lower = np.sin((np.pi / 2) * ids.astype(np.float64) / bins) ** 2
        self.assertTrue(np.all(scores.astype(np.float64) >= lower - 1e-15))
        self.assertTrue(np.all(scores.astype(np.float64) <= upper + 1e-15))
        np.testing.assert_allclose(result["risk_coverage"]["threshold_upper"],
                                   histogram_upper_edges(result["risk_coverage"]["bin_index"], bins))

    def test_exact_weighted_ties_match_sklearn(self):
        scores = np.array([0.2, 0.2, 0.7, 0.7, 0.9])
        y = np.array([0, 1, 0, 1, 1])
        w = np.array([4.0, 1.0, 2.0, 3.0, 5.0])
        result = exact_metrics(scores, y, w, np.arange(len(y), dtype=np.uint64))
        self.assertAlmostEqual(result["auroc"], roc_auc_score(y, scores, sample_weight=w))
        self.assertAlmostEqual(result["average_precision"], average_precision_score(y, scores, sample_weight=w))
        self.assertAlmostEqual(result["risk_coverage"]["risk"][-1], np.average(y, weights=w))
        self.assertEqual(result["risk_coverage"]["coverage"][-1], 1)

    def test_histogram_matches_expanded_tied_bins_and_thresholds(self):
        scores = np.array([0.2, 0.2, 0.7, 0.7, 0.9])
        y = np.array([0, 1, 0, 1, 1])
        w = np.array([4, 1, 2, 3, 5])
        expanded_scores, expanded_y = np.repeat(scores, w), np.repeat(y, w)
        accumulator = HistogramAccumulator(threshold=0.7).update(expanded_scores, expanded_y)
        result = accumulator.finalize(targets=(0.8,))
        self.assertAlmostEqual(result["auroc"], roc_auc_score(y, scores, sample_weight=w))
        self.assertAlmostEqual(result["average_precision"], average_precision_score(y, scores, sample_weight=w))
        self.assertEqual(result["num_pixels"], 15)
        self.assertEqual(result["fixed_threshold"]["accepted_pixels"], 10)
        self.assertEqual(result["fixed_threshold"]["accepted_errors"], 4)
        self.assertAlmostEqual(result["fixed_threshold"]["risk"], 0.4)
        self.assertEqual(result["risk_coverage"]["coverage"][-1], 1)

    def test_histogram_merge_preserves_fold_specific_threshold_counts(self):
        a = HistogramAccumulator(threshold=0.2).update([0.1, 0.3], [0, 1])
        b = HistogramAccumulator(threshold=0.8).update([0.7, 0.9], [1, 0])
        pooled = HistogramAccumulator().merge(a).merge(b).finalize()
        self.assertEqual(pooled["num_pixels"], 4)
        self.assertEqual(pooled["fixed_threshold"]["distinct_thresholds"], 2)
        self.assertIsNone(pooled["fixed_threshold"]["threshold"])
        self.assertEqual(pooled["fixed_threshold"]["coverage"], 0.5)
        self.assertEqual(pooled["fixed_threshold"]["risk"], 0.5)
        self.assertTrue(pooled["fixed_threshold"]["complete_population"])

    def test_quantile_is_weighted_and_accepts_all_ties(self):
        scores = np.array([0.1, 0.4, 0.4, 0.9])
        self.assertEqual(training_threshold(scores, [1, 1, 7, 1]), 0.4)
        self.assertEqual(np.mean(scores <= training_threshold(scores, [1, 1, 1, 1], 0.5)), 0.75)
        self.assertEqual(training_threshold(scores, [1, 1, 1, 1], 1), 0.9)

    def test_train_only_pipeline_and_predict_class_ablation(self):
        X = np.array([[0.95, 0.1, 0.8, 0], [0.8, 0.3, 0.6, 0],
                      [0.6, 0.5, 0.2, 1], [0.4, 0.8, 0.1, 1]], dtype=np.float32)
        y, w = np.array([0, 0, 1, 1]), np.array([1., 2., 3., 4.])
        held_out = np.array([[0.05, 1., 0., 18]], dtype=np.float32)
        for config, width in [("lr3", 3), ("lr3_class", 22)]:
            model = fit_pipeline(config, 1.0, X, y, w)
            transformer = model.named_steps["features"]
            scaler = transformer.named_transformers_["continuous"]
            np.testing.assert_allclose(scaler.mean_, X[:, :3].mean(axis=0), rtol=1e-6)
            before = scaler.mean_.copy()
            self.assertEqual(transformer.transform(held_out).shape, (1, width))
            scores = predict_scores(config, held_out, model)
            self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
            np.testing.assert_array_equal(scaler.mean_, before)
            self.assertTrue(model.diagnostics_["converged"])
        np.testing.assert_allclose(predict_scores("raw_msp", X), 1 - X[:, 0])
        np.testing.assert_allclose(predict_scores("raw_entropy", X), X[:, 1])

    def test_undefined_outcomes_and_input_validation(self):
        empty = HistogramAccumulator().finalize()
        self.assertIsNone(empty["auroc"])
        self.assertIsNone(empty["average_precision"])
        self.assertIsNone(empty["aurc"])
        for y, expected_ap in [([0, 0], None), ([1, 1], 1.0)]:
            result = HistogramAccumulator().update([0, 1], y).finalize()
            self.assertIsNone(result["auroc"])
            self.assertEqual(result["average_precision"], expected_ap)
        with self.assertRaises(ValueError):
            HistogramAccumulator().update([0.2, np.nan], [0, 1])
        with self.assertRaises(ValueError):
            HistogramAccumulator().update([0.2, 0.5], [0, 255])
        with self.assertRaises(ValueError):
            training_threshold([0.2], [0])
        with self.assertRaises(ValueError):
            fit_pipeline("lr3", 1, np.zeros((2, 4)), [0, 0], [1, 1])
        with self.assertRaises(ValueError):
            histogram_indices([0.2], bins=0)
        with self.assertRaises(ValueError):
            histogram_upper_edges([HISTOGRAM_BINS])

    def test_all_ignored_chunk_adds_no_pixels_or_errors(self):
        X = np.array([[.95, .1, .8, 0], [.4, .8, .1, 1]], dtype=np.float32)
        for config in ('raw_msp', 'raw_entropy', 'lr3', 'lr3_class'):
            model = fit_pipeline(config, .1, X, [0, 1], [1, 1]) if config.startswith('lr') else None
            scores = predict_scores(config, np.empty((0, 4), dtype=np.float32), model)
            self.assertEqual(scores.shape, (0,))
            result = HistogramAccumulator(threshold=.5).update([.1, .9], [0, 1])
            result.update(scores, np.empty(0, dtype=np.uint8))
            final = result.finalize()
            self.assertEqual((final['num_pixels'], final['num_errors']), (2, 1))
            self.assertEqual(final['fixed_threshold']['accepted_pixels'], 1)
            self.assertTrue(final['fixed_threshold']['complete_population'])

    def test_nonconvergence_is_a_failed_candidate(self):
        def fail_to_converge(*args, **kwargs):
            warnings.warn("Iteration limit", ConvergenceWarning)
        with patch("experiments.cse805_error_prediction_v1.evaluation.Pipeline.fit", fail_to_converge):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                fit_pipeline("lr3", 1, np.zeros((2, 4)), [0, 1], [1, 1])


if __name__ == "__main__":
    unittest.main()
