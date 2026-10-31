"""Small analytical checks; no dataset, inference, or output files."""
import json
import unittest

import numpy as np

from experiments.paired_calibration_v1.metrics import (
    MetricAccumulator, sample_indices, weighted_risk_coverage,
)


class AccumulatorTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([[0,0,1],[1,2,255]],dtype=np.uint8)
        self.pred = np.array([[0,1,1],[2,2,0]],dtype=np.uint8)
        self.conf = np.array([[.95,.95,.8],[.4,.6,.99]])
        self.nll = np.array([[.1,3,.3],[2,.5,np.nan]])

    def test_counts_denominators_and_classes(self):
        a = MetricAccumulator(tau=.3).update(self.conf,self.nll,self.pred,self.gt)
        r = a.finalize()
        self.assertEqual((r['num_all'],r['num_valid'],r['num_correct']),(6,5,3))
        self.assertEqual((r['num_accepted'],r['num_accepted_errors']),(3,1))
        self.assertAlmostEqual(r['coverage_all'],4/6)
        self.assertAlmostEqual(r['coverage_valid'],3/5)
        self.assertAlmostEqual(r['accepted_risk'],1/3)
        self.assertAlmostEqual(r['high_conf_error_rate'],1/2)
        self.assertAlmostEqual(r['error_high_conf_fraction'],1/2)
        self.assertAlmostEqual(r['nll'],5.9/5)
        self.assertAlmostEqual(r['signed_gap'],3.7/5-3/5)
        self.assertEqual(a.confusion[:3,:3].tolist(),[[1,1,0],[0,1,1],[0,0,1]])
        rows = a.class_rows()
        self.assertAlmostEqual(rows[1]['pred_accepted_risk'],.5)
        self.assertAlmostEqual(rows[1]['gt_coverage'],.5)
        self.assertEqual(rows[2]['pred_accepted_risk'],None)
        self.assertEqual(rows[18]['gt_recall'],None)
        json.dumps(r,allow_nan=False)
        json.dumps(rows,allow_nan=False)

    def test_bins_edges_and_ece_by_definition(self):
        c = np.array([[0,1/15,14/15,1]],dtype=np.float64)
        y = np.zeros((1,4),dtype=np.uint8)
        p = np.array([[1,1,0,0]],dtype=np.uint8)
        a = MetricAccumulator().update(c,np.ones((1,4)),p,y)
        self.assertEqual(a.bin_count[[0,1,14]].tolist(),[1,1,2])
        self.assertEqual(int(a.bin_count.sum()),4)
        # Bin 0: zero gap; bin 1: 1/15; bin 14: mean(14/15,1) vs 1.
        self.assertAlmostEqual(a.finalize()['ece'],(1/15+2*(1-29/30))/4)

    def test_region_intersects_semantic_validity(self):
        region = np.array([[False,True,False],[True,False,True]])
        a = MetricAccumulator(tau=.3).update(self.conf,self.nll,self.pred,self.gt,region)
        r = a.finalize()
        self.assertEqual((r['num_all'],r['num_valid'],r['num_errors']),(3,2,2))
        self.assertAlmostEqual(r['coverage_all'],2/3)
        self.assertAlmostEqual(r['coverage_valid'],1/2)
        self.assertEqual(r['accepted_risk'],1)

    def test_frozen_stage3_calibration_parity_both_precisions(self):
        from stage3.evaluate_calibration import CalibrationAccumulator
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                c = np.array([[0, 1/15, .9], [14/15, 1, .55]],dtype=dtype)
                modern = MetricAccumulator().update(c,self.nll,self.pred,self.gt)
                valid = self.gt < 19
                legacy = CalibrationAccumulator()
                legacy.update(c[valid],self.pred[valid]==self.gt[valid],
                              float(self.nll[valid].sum(dtype=np.float64)))
                old, bins = legacy.finalize('synthetic',1)
                new = modern.finalize()
                for current,historical in (('nll','nll'),('ece','ece'),
                    ('accuracy','pixel_accuracy'),('high_conf_coverage','high_conf_coverage'),
                    ('high_conf_error_rate','high_conf_error_rate'),
                    ('error_high_conf_fraction','fraction_incorrect_that_are_high_conf')):
                    self.assertAlmostEqual(new[current],old[historical],places=14)
                self.assertEqual(modern.bin_count.tolist(),[b['count'] for b in bins])

    def test_merge_matches_pool_and_empty_is_undefined(self):
        pooled = MetricAccumulator(tau=.3).update(self.conf,self.nll,self.pred,self.gt)
        left = MetricAccumulator(tau=.3).update(self.conf[:1],self.nll[:1],self.pred[:1],self.gt[:1])
        right = MetricAccumulator(tau=.3).update(self.conf[1:],self.nll[1:],self.pred[1:],self.gt[1:])
        left.merge(right)
        for key,value in pooled.finalize().items():
            if key == 'num_images':continue
            if isinstance(value,float):self.assertAlmostEqual(left.finalize()[key],value)
            else:self.assertEqual(left.finalize()[key],value)
        np.testing.assert_array_equal(left.confusion,pooled.confusion)
        empty = MetricAccumulator().update(np.ones((1,1)),np.array([[np.nan]]),
                                         np.zeros((1,1),dtype=np.uint8),np.full((1,1),255,dtype=np.uint8))
        self.assertIsNone(empty.finalize()['ece'])
        self.assertEqual(empty.finalize()['coverage_all'],1)
        self.assertIsNone(empty.finalize()['coverage_valid'])
        with self.assertRaises(ValueError):left.merge(MetricAccumulator(tau=.2))


class SampledCurveTests(unittest.TestCase):
    def test_weighted_curve_manual_algebra(self):
        r = weighted_risk_coverage(np.array([.3,.1,.2]),np.array([1,0,1]),
                                  np.array([2.,1.,3.]),np.array([30,10,20]),targets=(.5,1.))
        np.testing.assert_allclose(r['coverage'],[1/6,4/6,1])
        np.testing.assert_allclose(r['risk'],[0,3/4,5/6])
        self.assertAlmostEqual(r['aurc'],.5*.75+(1/3)*(5/6))
        self.assertEqual(r['checkpoints'][1]['actual_coverage'],1)
        self.assertAlmostEqual(r['checkpoints'][0]['coverage_error'],1/6)
        self.assertAlmostEqual(r['risk'][-1],5/6)

    def test_ties_same_scores_and_permutation_invariance(self):
        s = np.array([.1,.1,.1,.2])
        e = np.array([1,0,1,0])
        w = np.ones(4)
        k = np.array([30,10,20,40],dtype=np.uint64)
        first = weighted_risk_coverage(s,e,w,k)
        idx = [3,1,0,2]
        second = weighted_risk_coverage(s[idx],e[idx],w[idx],k[idx])
        np.testing.assert_array_equal(first['risk'],[0,.5,2/3,.5])
        np.testing.assert_array_equal(first['risk'],second['risk'])
        self.assertEqual(first['aurc'],second['aurc'])
        duplicate_method = weighted_risk_coverage(s.copy(),e,w,k)
        np.testing.assert_array_equal(first['risk'],duplicate_method['risk'])
        nearest = weighted_risk_coverage(np.arange(4),e,w,k,targets=(.625,))
        self.assertEqual(nearest['checkpoints'][0]['actual_coverage'],.5)

    def test_sampling_shared_label_independent_and_weighted(self):
        gt = np.arange(100).reshape(10,10)%19
        gt[0,0] = 255
        a = sample_indices(gt,'normal/scene1',n=20)
        changed = np.where(gt==255,255,(gt+3)%19)
        b = sample_indices(changed,'normal/scene1',n=20)
        for key in ('indices','weights','tie_keys'):
            np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(len(set(a['indices'])),20)
        self.assertTrue(np.all(gt.ravel()[a['indices']]<19))
        self.assertAlmostEqual(a['weights'].sum(),99)
        c = sample_indices(gt,'normal/scene1',n=1000)
        self.assertEqual(len(c['indices']),99)
        self.assertTrue(np.all(c['weights']==1))
        self.assertFalse(np.array_equal(a['tie_keys'],sample_indices(gt,'night/scene1',n=20)['tie_keys']))

    def test_empty_and_invalid(self):
        empty = sample_indices(np.full((1,1),255,dtype=np.uint8),'empty')
        r = weighted_risk_coverage(np.array([]),np.array([]),empty['weights'],empty['tie_keys'])
        self.assertIsNone(r['aurc'])
        self.assertIsNone(r['checkpoints'][0]['risk'])
        with self.assertRaises(ValueError):
            weighted_risk_coverage(np.array([0.]),np.array([1]),np.array([0.]),np.array([1]))


if __name__ == '__main__':
    unittest.main()
