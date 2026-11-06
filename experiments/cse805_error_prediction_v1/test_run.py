"""Check metric bin parity and nested selection isolation without real fitting."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import run
from .evaluation import HistogramAccumulator


class RunTests(unittest.TestCase):
    def test_weighted_qa_bins_match_full_histogram_at_float32_edges(self):
        scores = np.array([0, np.nextafter(np.float32(0), np.float32(1)),
            np.finfo(np.float32).eps, 1e-6, .01, .02, .2, .7, .9,
            np.nextafter(np.float32(1), np.float32(0)), 1], dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
        weights = np.array([3, 5, 7, 2, 4, 8, 1, 3, 2, 1, 4])
        qa = run.quantized_qa(scores, labels, weights)
        expanded = HistogramAccumulator().update(
            np.repeat(scores, weights), np.repeat(labels, weights)).finalize()
        for key in ('auroc', 'average_precision', 'aurc', 'error_prevalence'):
            self.assertAlmostEqual(qa[key], expanded[key], places=12)
        for sampled, full in zip(qa['checkpoints'], expanded['checkpoints']):
            for key in ('actual_coverage', 'risk'):
                self.assertAlmostEqual(sampled[key], full[key], places=12)

    def test_real_score_ties_explain_rc_differences_without_quantization_loss(self):
        scores = np.array([.1, .1, .9, .9])
        labels = np.array([0, 1, 0, 1], dtype=np.uint8)
        weights = np.ones(4)
        grouped = run.quantized_qa(scores, labels, weights)
        exact = run.exact_metrics(scores, labels, weights, np.arange(4, dtype=np.uint64))
        for key in ('auroc', 'average_precision'):
            self.assertAlmostEqual(grouped[key], exact[key])
        self.assertAlmostEqual(grouped['aurc'], .5)
        self.assertAlmostEqual(exact['aurc'], 1 / 3)
        self.assertNotEqual(grouped['checkpoints'][3]['actual_coverage'],
                            exact['checkpoints'][3]['actual_coverage'])

    def test_nested_fits_and_thresholds_never_receive_outer_group(self):
        groups = np.repeat(['fit_a', 'fit_b', 'fit_c', 'outer'], 4)
        identifiers = np.arange(len(groups))
        X = np.column_stack((identifiers, np.zeros((len(groups), 3))))
        y = identifiers % 2
        w = identifiers + 1.
        training_groups = ['fit_a', 'fit_b', 'fit_c']
        outer = dict(eval_group='outer', train_groups=training_groups,
            inner_folds=[dict(inner_fold=i, dev_groups=[group],
                fit_groups=[other for other in training_groups if other != group])
                for i, group in enumerate(training_groups)])
        fits, thresholds = [], []

        def fake_fit(method, C, values, labels, weights):
            ids = values[:, 0].astype(int)
            self.assertTrue(np.all(groups[ids] != 'outer'))
            np.testing.assert_array_equal(labels, y[ids])
            np.testing.assert_array_equal(weights, w[ids])
            fits.append((method, C, ids.copy()))
            return SimpleNamespace(ids=ids, diagnostics_={'config': method})

        def fake_predict(method, values, model):
            ids = values[:, 0].astype(int)
            self.assertTrue(np.all(groups[ids] != 'outer'))
            self.assertFalse(np.intersect1d(ids, model.ids).size)
            return ids / 100.

        def fake_thresholds(values, weights, models):
            ids = values[:, 0].astype(int)
            np.testing.assert_array_equal(ids, np.flatnonzero(groups != 'outer'))
            np.testing.assert_array_equal(weights, w[ids])
            for model in models.values():
                np.testing.assert_array_equal(model.ids, ids)
            thresholds.append(ids)
            return {}

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(run, 'DATA', Path(temporary)), \
                patch.object(run, 'fit_pipeline', side_effect=fake_fit), \
                patch.object(run, 'predict_scores', side_effect=fake_predict), \
                patch.object(run, 'training_thresholds', side_effect=fake_thresholds), \
                patch.object(run, 'source_hashes', return_value={'source': 'fixed'}), \
                patch.object(run, 'sha256', return_value='fixed'), \
                patch.object(run, 'save_json'), patch.object(run, 'event'), \
                patch.object(run.joblib, 'dump'), patch('builtins.print'):
            models, selection = run.select_and_fit(outer, (X, y, w, groups), 'protocol')
        self.assertEqual(len(fits), 20)
        self.assertEqual(len(thresholds), 1)
        self.assertEqual(set(models), {'lr3', 'lr3_class'})
        self.assertEqual(selection['selected'], {'lr3': .01, 'lr3_class': .01})
        self.assertTrue(all(candidate['status'] == 'PASS' for candidate in selection['candidates']))


if __name__ == '__main__':
    unittest.main()
