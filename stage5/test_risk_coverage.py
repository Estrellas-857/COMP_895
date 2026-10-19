"""Synthetic Stage 5B curve/endpoint checks; no dataset access."""
import unittest
import numpy as np
from stage5.evaluate_risk_coverage import risk_coverage, checkpoint


class RiskCoverageTests(unittest.TestCase):
    def test_low_to_high_direction_and_baseline_endpoint(self):
        c=np.zeros(1000,dtype=np.int64);i=c.copy()
        c[0]=3;i[1]=1;c[2]=4;i[-1]=2
        curve=risk_coverage(c,i)
        np.testing.assert_allclose(curve['coverage'][:3],[.3,.4,.8])
        np.testing.assert_allclose(curve['risk'][:3],[0,.25,.125])
        self.assertEqual(curve['coverage'][-1],1)
        self.assertEqual(curve['risk'][-1],.3)
        self.assertEqual(checkpoint(curve,.8)['uncertainty_threshold_bin'],2)

    def test_skip_zero_endpoint_keep_empty_bin_duplicates(self):
        c=np.zeros(1000,dtype=np.int64);i=c.copy()
        c[7]=2;i[12]=2
        curve=risk_coverage(c,i)
        self.assertEqual(curve['bin_index'][0],7)
        self.assertEqual(curve['coverage'][0],.5)
        self.assertEqual(curve['coverage'][1],.5)
        point=checkpoint(curve,.5)
        self.assertEqual(point['uncertainty_threshold_bin'],7)
        self.assertEqual(point['uncertainty_threshold_upper'],.008)
        self.assertFalse(point['threshold_upper_inclusive'])

    def test_nearest_not_interpolated_and_tie_uses_first(self):
        c=np.zeros(1000,dtype=np.int64);i=c.copy()
        c[0]=2;i[1]=2
        curve=risk_coverage(c,i)
        point=checkpoint(curve,.75)
        self.assertEqual(point['actual_histogram_coverage'],.5)
        self.assertEqual(point['risk'],0)
        self.assertEqual(checkpoint(curve,.9)['actual_histogram_coverage'],1)

    def test_empty_histogram_rejected(self):
        c=np.zeros(1000,dtype=np.int64)
        with self.assertRaises(AssertionError):risk_coverage(c,c)


if __name__=='__main__':unittest.main(verbosity=2)
