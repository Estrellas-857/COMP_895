"""Synthetic aggregate counts only; no dataset, model fitting or GPU work."""
import unittest

import numpy as np

from .pixel_diagnostics import confusion_at_cutoffs


class PixelDiagnosticTests(unittest.TestCase):
    def test_cumulative_confusions_match_direct_acceptance_including_ties(self):
        gt = np.array([0, 1, 2, 2, 10, 0, 18])
        predicted = np.array([0, 0, 2, 1, 2, 1, 18])
        bins = np.array([0, 5, 5, 9, 10, 99, 99])
        cutoffs = np.array([0, 5, 5, 9, 99])
        counts = confusion_at_cutoffs(gt * 19 + predicted, bins, cutoffs)
        for i, cutoff in enumerate(cutoffs):
            accepted = bins <= cutoff
            expected = np.bincount((gt * 19 + predicted)[accepted], minlength=361).reshape(19, 19)
            np.testing.assert_array_equal(counts[i], expected)
            self.assertEqual(int(counts[i].sum() - np.trace(counts[i])), int(np.count_nonzero(accepted & (gt != predicted))))
        self.assertEqual(counts[-1].sum(), len(gt))

    def test_empty_and_invalid_inputs(self):
        empty = np.array([], dtype=np.int64)
        self.assertEqual(confusion_at_cutoffs(empty, empty, np.array([0, 99])).shape, (2, 19, 19))
        with self.assertRaises(ValueError):
            confusion_at_cutoffs(np.array([361]), np.array([0]), np.array([0]))
        with self.assertRaises(ValueError):
            confusion_at_cutoffs(np.array([0]), np.array([0]), np.array([2, 1]))


if __name__ == "__main__":
    unittest.main()
