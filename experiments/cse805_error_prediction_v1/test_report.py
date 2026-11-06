"""Reporting checks focus on estimand and leakage-safe completeness."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from .evaluation import HistogramAccumulator, exact_metrics
from .report import summarize_deltas, restore_accumulator, collect, extraction_summary


class ReportTests(unittest.TestCase):
    def test_family_mean_is_not_weighted_by_correlated_pixels(self):
        rows = [dict(group=str(i), weather="rain", method="lr3", baseline="raw_msp", risk_delta=value,
                     method_actual_coverage=.8, baseline_actual_coverage=.8, actual_coverage_difference=0., num_pixels=pixels)
                for i, (value, pixels) in enumerate([(-.1, 1), (.05, 1_000_000), (.02, 10)])]
        result = next(r for r in summarize_deltas(rows) if r["scope"] == "all")
        self.assertAlmostEqual(result["mean_risk_delta"], -.01)
        self.assertEqual((result["improved_groups"], result["worsened_groups"]), (1, 2))
        self.assertAlmostEqual(result["leave_one_group_out_mean_min"], -.04)
        self.assertAlmostEqual(result["leave_one_group_out_mean_max"], .035)

    def test_secondary_macro_deltas_retain_opposite_metric_directions(self):
        rows = [dict(group=str(i), weather="night", method="lr3_class", baseline="raw_msp",
                     risk_delta=-.01, auroc_delta=auroc, ap_delta=ap, aurc_delta=aurc,
                     method_actual_coverage=.8, baseline_actual_coverage=.8, actual_coverage_difference=0.)
                for i, (auroc, ap, aurc) in enumerate(((-.08, -.04, .01), (-.02, None, -.02)))]
        result = next(r for r in summarize_deltas(rows) if r["scope"] == "all")
        self.assertAlmostEqual(result["mean_auroc_delta"], -.05)
        self.assertAlmostEqual(result["mean_ap_delta"], -.04)
        self.assertAlmostEqual(result["mean_aurc_delta"], -.005)
        self.assertEqual(result["auroc_num_groups"], 2)
        self.assertEqual(result["ap_num_groups"], 1)
        self.assertTrue(result["opposite_metric_directions"])

    def test_histogram_restore_preserves_exact_fold_threshold_counts(self):
        original = HistogramAccumulator(threshold=.5).update(np.array([.1, .4, .7, .9]), np.array([0, 1, 0, 1]))
        rebuilt = restore_accumulator(original.finalize(), original.correct, original.incorrect)
        self.assertEqual(rebuilt.finalize()["fixed_threshold"], original.finalize()["fixed_threshold"])
        damaged = original.correct.copy()
        damaged[0] += 1
        with self.assertRaisesRegex(ValueError, "population mismatch"):
            restore_accumulator(original.finalize(), damaged, original.incorrect)

    def test_histogram_resolution_is_read_from_counts(self):
        original = HistogramAccumulator(bins=37, threshold=.4).update([.02, .25, .6, .99], [0, 1, 0, 1])
        rebuilt = restore_accumulator(original.finalize(), original.correct, original.incorrect)
        self.assertEqual(rebuilt.bins, 37)
        self.assertAlmostEqual(rebuilt.finalize()["aurc"], original.finalize()["aurc"])

    def test_true_score_ties_can_change_risk_without_quantization_loss(self):
        scores, errors = np.full(4, .5), np.array([0, 0, 1, 1])
        exact = exact_metrics(scores, errors, np.ones(4), np.arange(4, dtype=np.uint64))
        binned = HistogramAccumulator().update(scores, errors).finalize()
        self.assertAlmostEqual(exact["auroc"], binned["auroc"])
        self.assertAlmostEqual(exact["average_precision"], binned["average_precision"])
        self.assertGreater(binned["aurc"], exact["aurc"])
        exact80 = next(p for p in exact["checkpoints"] if p["requested_coverage"] == .8)
        binned80 = next(p for p in binned["checkpoints"] if p["requested_coverage"] == .8)
        self.assertNotEqual(exact80["actual_coverage"], binned80["actual_coverage"])

    def test_extraction_summary_distinguishes_zero_difference_from_not_compared(self):
        records = [dict(image_id=str(i), native_cache={"origin": origin}, num_valid=2_000_000,
                        num_ignored=73_600, num_errors=100 + i,
                        probability_normalization_max_error=1e-7 * (i+1),
                        historical_prediction_changes=changed)
                   for i, (origin, changed) in enumerate((("reused", None), ("new_frozen_forward", 0),
                                                          ("new_frozen_forward", 2)))]
        result = extraction_summary(records)
        self.assertEqual(result["historical_prediction_images_compared"], 2)
        self.assertEqual(result["historical_prediction_images_differing"], 1)
        self.assertEqual(result["historical_prediction_pixels_differing"], 2)
        self.assertEqual(result["num_valid_pixels"], 6_000_000)
        self.assertEqual(result["num_error_pixels"], 303)
        self.assertEqual(result["num_ignored_pixels"], 220_800)
        self.assertEqual((result["reused_native_images"], result["new_forward_images"]), (1, 2))
        with self.assertRaisesRegex(ValueError, "unique"):
            extraction_summary(records + records[:1])

    def test_partial_results_cannot_be_reported_as_final(self):
        with TemporaryDirectory() as directory, patch("experiments.cse805_error_prediction_v1.report.load_protocol"), \
                patch("experiments.cse805_error_prediction_v1.report._read_json", return_value=[{"eval_group": f"g{i}"} for i in range(17)]):
            with self.assertRaisesRegex(ValueError, "exactly all 17"):
                collect(Path(directory))


if __name__ == "__main__":
    unittest.main()
