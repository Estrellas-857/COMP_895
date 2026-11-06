"""No fitting: verify frozen-model algebra, mapping and weighted class counts."""
import unittest
from copy import deepcopy

import numpy as np

from .shared import models
from .coefficient_analysis import inspect_pipeline, decompose, profile
from ..evaluation import CONFIGS


class CoefficientTests(unittest.TestCase):
    def test_frozen_pipeline_feature_mapping_and_reconstruction(self):
        fitted, _ = models("gopro_0376")
        X = np.column_stack((np.linspace(.2, .99, 19), np.linspace(.9, .01, 19),
                             np.linspace(.01, .98, 19), np.arange(19))).astype(np.float32)
        for method in ("lr3", "lr3_class"):
            a = inspect_pipeline(fitted[method], method)
            result = decompose(fitted[method], method, a, X)
            self.assertLess(result["transformed_max_abs_error"], 1e-10)
            self.assertLess(result["raw_units_max_abs_error"], 1e-4)
            centered = a["centered_raw_intercept"] + X[:, :3] @ a["raw_beta"] + a["gamma"][X[:, 3].astype(int)] - a["gamma"].mean()
            np.testing.assert_allclose(centered, result["logit"], atol=1e-4, rtol=0)
        damaged = deepcopy(fitted["lr3_class"])
        damaged.named_steps["features"].named_transformers_["class"].categories_[0] = np.arange(19)[::-1]
        with self.assertRaisesRegex(ValueError, "mapping"):
            inspect_pipeline(damaged, "lr3_class")

    def test_profile_preserves_weighted_prevalence_and_empty_classes(self):
        X = np.array([[.9, .2, .8, 0], [.99, .1, .9, 0], [.4, .8, .1, 1]])
        y, w = np.array([0, 1, 1]), np.array([9., 1., 10.])
        scores = {m: np.array([.1, .2, .3]) for m in CONFIGS}
        result = profile(X, y, w, scores, X[:, 3] == 0)
        self.assertEqual(result["sample_errors"], 1)
        self.assertAlmostEqual(result["error_rate"], .1)
        self.assertAlmostEqual(result["class_population_share"], .5)
        self.assertAlmostEqual(result["mean_lr3_score"], .11)
        empty = profile(X, y, w, scores, X[:, 3] == 18)
        self.assertEqual(empty["weighted_pixels"], 0)
        self.assertIsNone(empty["error_rate"])
        self.assertTrue(empty["small_sample"])


if __name__ == "__main__":
    unittest.main()
