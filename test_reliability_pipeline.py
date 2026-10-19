"""Focused Stage 4A engineering checks; synthetic logits only, no dataset labels."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from reliability_pipeline import (
    ReliabilityPipeline, METHODS, TEMPERATURE_PATH, EXPECTED_TEMPERATURE,
    apply_reliability_threshold, load_frozen_temperature,
)


class ReliabilityTests(unittest.TestCase):
    def pipeline(self):
        # Exercise postprocessing on CPU without loading network weights for unit tests.
        pipe = ReliabilityPipeline.__new__(ReliabilityPipeline)
        pipe.threshold = 0.10
        pipe.uncertainty_method = "calibrated_msp"
        pipe.device = torch.device("cpu")
        pipe.temperature_path = TEMPERATURE_PATH
        pipe.temperature, pipe.temperature_sha256 = load_frozen_temperature()
        return pipe

    def test_inclusive_threshold_and_invalid_inputs(self):
        values = np.array([[np.nextafter(.1, 0), .1, np.nextafter(.1, 1)]])
        np.testing.assert_array_equal(apply_reliability_threshold(values, .1), [[True, True, False]])
        for threshold in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                apply_reliability_threshold(values, threshold)
        for values in (np.array([[np.nan]]), np.array([[-.01]]), np.array([[1.01]])):
            with self.assertRaises(ValueError):
                apply_reliability_threshold(values, .1)

    def test_formulas_precision_and_segmentation_independence(self):
        pipe = self.pipeline()
        logits = torch.linspace(-5, 5, 19 * 6, dtype=torch.float32).reshape(1, 19, 2, 3)
        before = logits.clone()
        for method in METHODS:
            result = pipe.predict_from_logits(logits, uncertainty_method=method)
            scaled = logits.double() / pipe.temperature if method.startswith("calibrated") else logits
            probs = scaled.softmax(dim=1)
            expected = (1 - probs.max(dim=1).values if method.endswith("msp") else
                        -(probs * probs.clamp_min(1e-12).log()).sum(dim=1) / math.log(19))
            np.testing.assert_array_equal(result["uncertainty"], expected[0].numpy())
            np.testing.assert_array_equal(result["segmentation"], logits.argmax(dim=1)[0].numpy())
            self.assertEqual(result["uncertainty"].dtype, np.float64 if method.startswith("calibrated") else np.float32)
            self.assertIs(result["accepted_mask"], result["reliability_mask"])
            self.assertEqual(result["metadata"]["accepted_pixels"] + result["metadata"]["rejected_pixels"], 6)
        torch.testing.assert_close(logits, before, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            pipe.compute_uncertainty(logits, "unsupported")

    def test_uniform_distribution_entropy(self):
        pipe = self.pipeline()
        logits = torch.zeros(1, 19, 2, 3)
        for method in METHODS:
            uncertainty, confidence = pipe.compute_uncertainty(logits, method)
            np.testing.assert_allclose(confidence, 1 / 19, atol=1e-7)
            np.testing.assert_allclose(uncertainty, 1 - 1 / 19 if method.endswith("msp") else 1, atol=1e-7)

    def test_frozen_temperature_rejects_changed_artifact(self):
        self.assertEqual(load_frozen_temperature()[0], EXPECTED_TEMPERATURE)
        artifact = json.loads(TEMPERATURE_PATH.read_text())
        artifact["temperature"] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):
                load_frozen_temperature(path)

    def test_public_path_pil_and_original_size_without_dataset_naming(self):
        pipe = self.pipeline()
        calls = []
        def processor(**kwargs):
            calls.append(kwargs)
            return {"pixel_values": torch.zeros(1, 3, 512, 1024)}
        class FrozenStub(torch.nn.Module):
            def forward(self, **kwargs):
                assert not self.training and torch.is_inference_mode_enabled()
                return SimpleNamespace(logits=torch.zeros(1, 19, 128, 256))
        pipe.processor, pipe.model = processor, FrozenStub()
        rgb = Image.new("L", (31, 17), color=100)
        pil_result = pipe.predict(rgb)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unseen-picture.png"
            rgb.save(path)
            path_result = pipe.predict(path)
        self.assertEqual(pil_result["segmentation"].shape, (17, 31))
        np.testing.assert_array_equal(pil_result["segmentation"], path_result["segmentation"])
        for call in calls:
            self.assertEqual(call["images"].mode, "RGB")
            self.assertEqual(call["size"], {"height": 512, "width": 1024})


if __name__ == "__main__":
    unittest.main(verbosity=2)
