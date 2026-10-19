"""Dataset access guards must cover ACDC outside the project directory."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage1 import dataset
from stage4 import select_operating_threshold as selection
from stage5 import benchmark_final_reliability as benchmark
from stage5 import evaluate_frozen_operating_point as evaluation


class ExternalDatasetGuardTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(temporary) / "external/ACDC"
        for module in (dataset, selection, benchmark, evaluation):
            self.stack.enter_context(patch.object(module, "ACDC_ROOT", self.root))
        # Exercise hook logic without installing irreversible process-wide hooks.
        self.stack.enter_context(patch("sys.addaudithook"))
        self.rgb = self.root / "rgb_anon_trainvaltest/rgb_anon/fog/val/seq/image.png"
        self.gt = self.root / "gt_trainval/gt/fog/val/seq/label.png"

    def test_gt_is_blocked_until_inference_completes(self):
        guard = evaluation.InferenceBeforeGTAudit()
        guard.begin(self.rgb, self.gt)
        guard.audit("open", (str(self.rgb),))
        with self.assertRaises(RuntimeError):
            guard.audit("open", (str(self.gt),))
        guard.inference_completed()
        guard.audit("open", (str(self.gt),))
        guard.finish()
        self.assertEqual(guard.events[0]["rgb_path"], dataset.dataset_reference(self.rgb))
        self.assertTrue(guard.events[0]["gt_opened_only_after_inference"])
        with self.assertRaises(RuntimeError):
            guard.audit("open", (str(self.rgb),))

    def test_selection_allowlist_resolves_frozen_paths(self):
        row = {"rgb_path": dataset.dataset_reference(self.rgb),
               "gt_path": dataset.dataset_reference(self.gt)}
        guard = selection.DatasetAccessGuard([row], "select")
        guard.audit("open", (str(self.rgb),))
        self.assertTrue(guard.report()["only_manifest_files_opened"])
        with self.assertRaises(RuntimeError):
            guard.audit("open", (str(self.gt),))

    def test_benchmark_blocks_labels_in_external_dataset(self):
        guard = benchmark.RGBOnlyAudit([self.rgb])
        guard.hook("open", (str(self.rgb),))
        self.assertEqual(guard.opened, {dataset.dataset_reference(self.rgb)})
        with self.assertRaises(RuntimeError):
            guard.hook("open", (str(self.gt),))


if __name__ == "__main__":
    unittest.main()
