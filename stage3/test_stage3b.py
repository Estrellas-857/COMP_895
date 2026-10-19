"""Synthetic numerical checks; no validation or reserved image data are read."""

import math
import unittest

import torch
import torch.nn.functional as F

from stage3.fit_temperature import pixel_terms, solve_temperature


class TemperatureTests(unittest.TestCase):
    def test_loss_derivatives_ignore_mask_and_all_pixels(self):
        generator = torch.Generator().manual_seed(42)
        logits = torch.randn(1, 19, 3, 4, dtype=torch.float64, generator=generator)
        target = torch.tensor([[0, 1, 2, 255], [18, 4, 5, 6], [7, 8, 9, 255]])
        valid = target != 255
        beta = torch.tensor(0.43, dtype=torch.float64, requires_grad=True)
        independent = F.cross_entropy(logits * beta, target[None], ignore_index=255, reduction="sum")
        gradient = torch.autograd.grad(independent, beta, create_graph=True)[0]
        hessian = torch.autograd.grad(gradient, beta)[0]
        nll, g, h, g0 = pixel_terms(logits, target, valid, 1 / beta.item())
        self.assertAlmostEqual(nll, independent.item(), places=12)
        self.assertAlmostEqual(g, gradient.item(), places=12)
        self.assertAlmostEqual(h, hessian.item(), places=12)
        eps = 1e-5
        plus = pixel_terms(logits, target, valid, 1 / (beta.item() + eps))[0]
        minus = pixel_terms(logits, target, valid, 1 / (beta.item() - eps))[0]
        self.assertAlmostEqual(g, (plus-minus)/(2*eps), places=7)
        self.assertTrue(math.isfinite(g0))

    def test_known_binomial_optimum_and_expanding_bracket(self):
        for true_probability in (0.75, 0.95):
            def objective(beta, phase=None):
                p = 1 / (1 + math.exp(-2 * beta))
                return dict(beta=beta, nll=math.log1p(math.exp(2*beta))-2*true_probability*beta,
                            gradient=2*(p-true_probability), hessian=4*p*(1-p),
                            gradient0=2*(0.5-true_probability))
            raw, fitted, iterations, reason = solve_temperature(objective)
            expected_beta = math.log(true_probability / (1-true_probability)) / 2
            self.assertAlmostEqual(fitted["beta"], expected_beta, places=6)
            self.assertLessEqual(fitted["nll"], raw["nll"])
            self.assertLess(iterations, 50)
            self.assertTrue(reason)

    def test_extreme_logits_and_positive_scaling(self):
        logits32 = torch.tensor([1000., -1000., 0.] + [1.]*16).reshape(1, 19, 1, 1)
        target = torch.tensor([[1]])
        nll, g, h, _ = pixel_terms(logits32.double(), target, target != 255, 2.)
        self.assertAlmostEqual(nll, 1000.)
        self.assertTrue(all(math.isfinite(v) for v in (nll, g, h)))
        self.assertTrue(torch.equal(logits32.argmax(1), (logits32.double()/2.31).argmax(1)))


if __name__ == "__main__":
    unittest.main()
