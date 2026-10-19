"""Location overrides and frozen-manifest path compatibility."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage1 import dataset


class DatasetPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "Documents/ACDC"
        self.project = self.base / "project"
        patches = patch.multiple(dataset, ACDC_ROOT=self.root, PROJECT_ROOT=self.project)
        patches.start()
        self.addCleanup(patches.stop)
        self.relative = Path("rgb_anon_trainvaltest/rgb_anon/fog/train/sequence/image.png")

    def test_default_location_is_in_documents(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "home", return_value=self.base):
            self.assertEqual(dataset.get_acdc_root(), self.root)

    def test_environment_override(self):
        custom = self.base / "shared/datasets/ACDC"
        with patch.dict(os.environ, {"ACDC_ROOT": str(custom)}):
            self.assertEqual(dataset.get_acdc_root(), custom)
        with patch.dict(os.environ, {"ACDC_ROOT": "relative/dataset"}):
            with self.assertRaises(ValueError):
                dataset.get_acdc_root()

    def test_legacy_and_current_references_resolve_to_same_file(self):
        expected = self.root / self.relative
        for reference in (self.relative, Path("data/ACDC") / self.relative,
                          self.project / "data/ACDC" / self.relative, expected):
            self.assertEqual(dataset.acdc_path(reference), expected)

    def test_manifest_identifier_round_trip(self):
        path = self.root / self.relative
        reference = dataset.dataset_reference(path)
        self.assertEqual(reference, "data/ACDC/" + self.relative.as_posix())
        self.assertEqual(dataset.acdc_path(reference), path)

    def test_unrelated_paths_and_directory_escape_are_rejected(self):
        for reference in ("results/table.csv", self.base / "unrelated.png",
                          "data/ACDC/rgb_anon_trainvaltest/../../outside.png"):
            with self.assertRaises(ValueError, msg=str(reference)):
                dataset.acdc_path(reference)


if __name__ == "__main__":
    unittest.main()
