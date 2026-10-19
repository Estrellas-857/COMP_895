"""Focused final-configuration integrity checks; no model inference or dataset I/O."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_final_reliability as final


class FinalConfigurationTests(unittest.TestCase):
    def test_source_artifacts_match_config(self):
        config, digest = final.load_final_config()
        self.assertEqual(config["temperature"], 2.124186626456894)
        self.assertEqual(config["threshold"], 0.386413580715649)
        self.assertEqual(config["uncertainty_method"], "calibrated_msp")
        self.assertFalse(config["gt_required_at_inference"])
        self.assertEqual(len(digest), 64)

    def test_threshold_override_not_exposed(self):
        with self.assertRaises(TypeError):
            final.FinalReliabilityPipeline(threshold=.10)

    def test_modified_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(final.CONFIG_PATH.read_text())
            config["threshold"] = .10
            changed = root / "final_reliability_config.json"
            changed.write_text(json.dumps(config))
            (root / "final_config_freeze_record.json").write_bytes(
                (final.ROOT / "final_config_freeze_record.json").read_bytes())
            with patch.object(final, "ROOT", root), patch.object(final, "CONFIG_PATH", changed):
                with self.assertRaisesRegex(ValueError, "frozen hash"):
                    final.load_final_config()


if __name__ == "__main__":
    unittest.main(verbosity=2)
