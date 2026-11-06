import unittest
import numpy as np
import torch
import torch.nn.functional as F

from .features import feature_block


class FeaturesTest(unittest.TestCase):
    def test_probability_margin_and_canonical_class(self):
        z = torch.zeros((1, 19, 2, 3), dtype=torch.float32)
        z[0, 7, 0, 1] = 3
        X, discrepancy = feature_block(z)
        p = z.softmax(1)
        self.assertLess(discrepancy, 2e-6)
        self.assertEqual(X[1, 3], 7)
        self.assertAlmostEqual(float(X[0, 0]), 1/19, places=6)
        self.assertAlmostEqual(float(X[0, 1]), 1, places=6)
        self.assertEqual(X[0, 2], 0)
        self.assertAlmostEqual(float(X[1, 2]), float(p[0, 7, 0, 1] - p[0, 0, 0, 1]), places=6)

    def test_chunk_alignment_after_interpolation(self):
        generator = torch.Generator().manual_seed(805)
        z = torch.randn((1, 19, 3, 4), generator=generator)
        aligned = F.interpolate(z, (7, 11), mode='bilinear', align_corners=False)
        full, _ = feature_block(aligned)
        chunks = [feature_block(aligned[:, :, first:first+2])[0] for first in range(0, 7, 2)]
        np.testing.assert_allclose(np.concatenate(chunks), full, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(full[:, 3], aligned.argmax(1).numpy().ravel())

    def test_invalid_logits_fail(self):
        with self.assertRaises(ValueError):
            feature_block(torch.full((1, 19, 2, 2), float('nan')))


if __name__ == '__main__':
    unittest.main()
