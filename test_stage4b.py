"""Stage 4B algorithm checks using synthetic scores/counts only; no dataset GT."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from select_operating_threshold import (
    exact_external_quantile, float_prefix_hist, metrics_from_counts, safe_ratio,
)


class Stage4BTests(unittest.TestCase):
    def check_quantile(self, values, cap=128 * 1024**2):
        values = np.asarray(values, dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, chunk in enumerate(np.array_split(values, 3)):
                p = Path(directory) / f"{i}.npy"
                np.save(p, chunk)
                paths.append(p)
            actual, rank, trace = exact_external_quantile(paths, len(values), float_prefix_hist(values),
                                                        max_candidate_bytes=cap)
        expected = float(np.quantile(values, .8, method="inverted_cdf"))
        self.assertEqual(actual, expected)
        self.assertLess(np.count_nonzero(values < actual), rank)
        self.assertGreaterEqual(np.count_nonzero(values <= actual), rank)
        return trace

    def test_exact_random_and_endpoint_quantiles(self):
        rng = np.random.default_rng(42)
        self.check_quantile(np.concatenate(([0., 1.], rng.random(10003))))
        self.check_quantile([0, 0, 0, 0, 1])
        self.check_quantile([0, .1, .2, .3, 1, 1])

    def test_refinement_and_ties_with_tiny_memory_budget(self):
        values = .5 + np.arange(1000, dtype=np.float64) * 1e-12
        trace = self.check_quantile(values, cap=64)
        self.assertGreater(len(trace), 1)
        trace = self.check_quantile(np.full(101, .123456789), cap=0)
        self.assertEqual(trace[-1]["prefix_bits"], 64)

    def test_metrics_and_conservation(self):
        counts = dict(num_images=1, all_pixels=110, all_accepted_pixels=88, all_rejected_pixels=22,
                      valid_pixels=100, ignored_pixels=10, TP=15, FP=5, FN=5, TN=75)
        m = metrics_from_counts("synthetic", counts, .5)
        self.assertEqual(m["coverage"], .8)
        self.assertEqual(m["risk"], 5 / 80)
        self.assertEqual(m["baseline_risk"], .2)
        self.assertEqual(m["risk_reduction_absolute"], .2 - 5 / 80)
        self.assertEqual(m["error_recall"], .75)
        self.assertEqual(m["rejection_precision"], .75)
        self.assertEqual(m["incorrect_accepted_fraction"], .25)
        self.assertEqual(m["correct_rejection_rate"], 5 / 80)
        self.assertEqual(m["zero_denominators"], "")

    def test_zero_denominators_are_explicit(self):
        self.assertIsNone(safe_ratio(0, 0))
        counts = dict(num_images=1, all_pixels=10, all_accepted_pixels=0, all_rejected_pixels=10,
                      valid_pixels=10, ignored_pixels=0, TP=0, FP=10, FN=0, TN=0)
        m = metrics_from_counts("synthetic", counts, .1)
        self.assertIsNone(m["risk"])
        self.assertIsNone(m["error_recall"])
        self.assertIn("accepted_valid_pixels", m["zero_denominators"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
