"""Small synthetic checks of precision, own-GT evaluation and paired aggregation."""
import unittest
import numpy as np
import torch
from experiments.paired_calibration_v1.evaluation import score_maps, paired_statistics, PAIR_METRICS


class EvaluationTests(unittest.TestCase):
    def test_score_maps_canonical_and_stable_nll(self):
        logits = torch.full((1, 19, 1, 3), -10., dtype=torch.float32)
        logits[0, 0] = torch.tensor([1000., 2., 3.])
        logits[0, 1] = torch.tensor([-1000., 3., 2.])
        gt = np.array([[1, 0, 255]], dtype=np.uint8)
        for raw, temperature in ((True, 1.), (False, 2.124186626456894)):
            confidence, loss, changes = score_maps(logits, gt, temperature, raw=raw)
            self.assertEqual(changes, 0)
            self.assertTrue(np.isfinite(loss).all())
            self.assertGreater(float(loss[0, 0]), 900)
            self.assertEqual(confidence.dtype, np.float32 if raw else np.float64)
            expected = torch.log_softmax(logits if raw else logits.double()/temperature, dim=1)
            self.assertAlmostEqual(float(loss[0, 1]), float(-expected[0,0,0,1]), places=12 if not raw else 5)

    def test_pair_differences_are_equal_image_not_pixel_weighted(self):
        rows=[]
        for i, delta in enumerate((.1,.5)):
            for domain in ("normal", "night"):
                rows.append(dict(pair_id=str(i),sequence_id="GOPR0351",temperature_name="raw",domain=domain,
                                 **{m: .2+(delta if domain=="night" else 0) for m in PAIR_METRICS}))
        differences, summaries=paired_statistics(rows)
        self.assertEqual(len(differences),2)
        selected=[r for r in summaries if r["sequence_id"]=="pooled" and r["metric"]=="nll"][0]
        self.assertAlmostEqual(selected["mean_delta"],.3)
        self.assertEqual(selected["num_pairs"],2)
        with self.assertRaises(RuntimeError): paired_statistics(rows[:-1])


if __name__ == "__main__":
    unittest.main()
