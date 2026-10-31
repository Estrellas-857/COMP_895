"""CPU-only numerical and population guards; no dataset inference."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from experiments.paired_calibration_v1.engine import (
    _write_once, combine_domain_terms, pixel_terms, solve_temperature, validate_fit_population,
)


class EngineTests(unittest.TestCase):
    def rows(self):
        return [dict(split="fit", domain=domain, pair_id=f"scene_{i}", sequence_id="sequence")
                for domain in ("normal", "night") for i in range(101)]

    def test_matched_population_and_rejections(self):
        self.assertEqual(len(validate_fit_population(self.rows())["night"]), 101)
        for mutation in ("validation", "missing", "mismatched", "duplicate", "sequence"):
            rows = self.rows()
            if mutation == "validation":
                rows[0]["split"] = "val"
            elif mutation == "missing":
                rows.pop()
            elif mutation == "mismatched":
                rows[0]["pair_id"] = "unpaired"
            elif mutation == "duplicate":
                rows[0]["pair_id"] = rows[1]["pair_id"]
            else:
                rows[0]["sequence_id"] = "wrong_sequence"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_fit_population(rows)

    def test_mix_weights_domain_means_with_unequal_pixel_support(self):
        sums = {"normal": np.array([20., 4., 6., -8.]),
                "night": np.array([900., 600., 300., -300.])}
        counts = {"normal": 2, "night": 300}
        actual = combine_domain_terms(sums, counts, {"normal": .5, "night": .5})
        np.testing.assert_allclose(actual, [6.5, 2., 2., -2.5], rtol=0, atol=0)
        self.assertNotEqual(actual[0], (20 + 900) / 302)

    def test_stable_pixel_terms_match_autograd_derivatives(self):
        logits = torch.tensor([[[[2., -.5, 5.]], [[0., 1., -1.]], [[-1., .2, 2.]]]], dtype=torch.float64)
        target = torch.tensor([[0, 2, 0]])
        valid = torch.tensor([[True, True, False]])
        beta = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        logp = torch.log_softmax(logits * beta, dim=1)
        loss = -logp[0].gather(0, target[None])[0][valid].sum()
        grad = torch.autograd.grad(loss, beta, create_graph=True)[0]
        hess = torch.autograd.grad(grad, beta)[0]
        actual = pixel_terms(logits, target, valid, 1 / float(beta.detach()))
        np.testing.assert_allclose(actual[:3], [loss.item(), grad.item(), hess.item()], rtol=1e-13)

    def test_solver_finite_optimum_and_infinite_optimum_guard(self):
        def objective(beta, phase="optimization"):
            return dict(beta=beta, nll=1 + (beta - .6)**2,
                        gradient=2 * (beta - .6), hessian=2., gradient0=-1.2)
        raw, result, _, _ = solve_temperature(objective)
        self.assertAlmostEqual(result["beta"], .6, places=12)
        self.assertLess(result["nll"], raw["nll"])
        with self.assertRaisesRegex(RuntimeError, "infinity"):
            solve_temperature(lambda beta, phase="optimization":
                              dict(beta=beta, nll=beta, gradient=1., hessian=0., gradient0=1.))

    def test_immutable_json_creation_and_identical_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            _write_once(path, {"protocol": "frozen", "value": 1})
            before = path.read_bytes()
            _write_once(path, {"value": 1, "protocol": "frozen"})
            self.assertEqual(before, path.read_bytes())
            with self.assertRaises(RuntimeError):
                _write_once(path, {"protocol": "changed", "value": 1})
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
