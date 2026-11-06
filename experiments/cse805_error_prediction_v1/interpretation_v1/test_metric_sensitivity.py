"""Exact synthetic examples separating true ties, artificial ties, and weights."""
import unittest

import numpy as np

from .metric_sensitivity import METHODS, grouped_sorted, night_pair_decomposition, policies


class MetricSensitivityTests(unittest.TestCase):
    def test_true_tie_policy_changes_rc_not_grouped_roc_or_ap(self):
        result = policies(np.array([.1, .1, .9, .9]), np.array([0, 1, 0, 1]),
                          np.ones(4), np.arange(4, dtype=np.uint64))
        grouped, secondary = result['raw_grouped'][0], result['raw_secondary'][0]
        self.assertAlmostEqual(grouped['aurc'], .5)
        self.assertAlmostEqual(secondary['aurc'], 1 / 3)
        self.assertEqual(grouped['auroc'], secondary['auroc'])
        self.assertEqual(grouped['average_precision'], secondary['average_precision'])
        for bins in (10000, 100000, 1000000):
            self.assertAlmostEqual(result[f'arcsine_{bins}_grouped'][0]['aurc'], .5)

    def test_artificial_ties_are_removed_by_resolution(self):
        result = policies(np.array([1e-10, 2e-10]), np.array([0, 1]),
                          np.ones(2), np.arange(2, dtype=np.uint64))
        raw, coarse, fine = [result[key][0] for key in
                            ('raw_grouped', 'arcsine_10000_grouped', 'arcsine_1000000_grouped')]
        self.assertEqual((raw['auroc'], raw['average_precision'], raw['aurc']), (1., 1., .25))
        self.assertEqual((coarse['auroc'], coarse['average_precision'], coarse['aurc']), (.5, .5, .5))
        self.assertEqual((fine['auroc'], fine['average_precision'], fine['aurc']), (1., 1., .25))
        self.assertEqual(coarse['largest_accepted_block_mass'], 1.)

    def test_integer_weights_match_repeated_pixels_under_same_policy(self):
        scores = np.array([.1, .1, .2, .9])
        errors = np.array([0, 1, 0, 1])
        weights = np.array([3, 2, 4, 1])
        weighted, wp = grouped_sorted(scores, errors, weights)
        repeated, rp = grouped_sorted(np.repeat(scores, weights), np.repeat(errors, weights), np.ones(weights.sum()))
        for key in ('auroc', 'average_precision', 'aurc'):
            self.assertAlmostEqual(weighted[key], repeated[key])
        self.assertEqual(wp, rp)

    def test_auc_pair_decomposition_separates_cross_group_ordering(self):
        rows = []
        for method in METHODS:
            for group in ('a', 'b'):
                rows.append(dict(scope='heldout_family', group=group, weather='night', method=method,
                    policy='raw_grouped', population_weight=2, error_prevalence=.5, auroc=1.))
            rows.append(dict(scope='pooled_crossfitted_night', method=method, policy='raw_grouped', auroc=.75))
        for row in night_pair_decomposition(rows):
            self.assertEqual(row['within_family_pair_weight_fraction'], .5)
            self.assertEqual(row['cross_family_pair_weight_fraction'], .5)
            self.assertEqual(row['within_family_pair_weighted_auroc'], 1.)
            self.assertEqual(row['cross_family_pair_weighted_auroc'], .5)


if __name__ == '__main__':
    unittest.main()
