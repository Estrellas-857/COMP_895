"""Regression checks for relocated source paths and strict artifact hashes."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage1 import _layout


class SourceLayoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.current = self.root / "stage4/pipeline.py"
        self.current.parent.mkdir()
        self.current.write_bytes(b"from stage1 import example\n")
        self.original_hash = hashlib.sha256(b"import example\n").hexdigest()
        self.current_hash = hashlib.sha256(self.current.read_bytes()).hexdigest()
        self.manifest = self.root / "source_layout.json"
        self.manifest.write_text(json.dumps({"files": {"pipeline.py": {
            "path": "stage4/pipeline.py",
            "original_sha256": self.original_hash,
            "sha256": self.current_hash,
        }}}))
        patches = patch.multiple(_layout, PROJECT_ROOT=self.root, LAYOUT_PATH=self.manifest)
        patches.start()
        self.addCleanup(patches.stop)

    def test_original_and_current_paths_accept_only_recorded_source(self):
        for path in ("pipeline.py", "stage4/pipeline.py", self.current):
            self.assertTrue(_layout.matches_frozen_hash(path, self.original_hash))
        self.assertTrue(_layout.matches_frozen_hash(self.current, self.current_hash))
        manifest = {"pipeline.py": self.original_hash}
        self.assertEqual(_layout.manifest_key(self.current, manifest), "pipeline.py")
        self.assertEqual(_layout.manifest_key("pipeline.py", manifest), "pipeline.py")

    def test_modified_source_is_rejected_at_both_paths(self):
        self.current.write_bytes(b"unexpected implementation change\n")
        for path in ("pipeline.py", self.current):
            self.assertFalse(_layout.matches_frozen_hash(path, self.original_hash))
        self.assertFalse(_layout.matches_frozen_hash(self.current, self.current_hash))

    def test_unknown_hash_or_unrelated_source_is_rejected(self):
        wrong_hash = "0" * 64
        self.assertFalse(_layout.matches_frozen_hash(self.current, wrong_hash))
        self.assertFalse(_layout.matches_frozen_hash("another/pipeline.py", self.original_hash))
        with self.assertRaises(KeyError):
            _layout.manifest_key("another/pipeline.py", {"pipeline.py": self.original_hash})

    def test_data_and_configuration_require_exact_bytes(self):
        artifact = self.root / "temperature.json"
        artifact.write_bytes(b'{"temperature": 2.0}\n')
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        self.assertTrue(_layout.matches_frozen_hash(artifact, digest))
        artifact.write_bytes(b'{"temperature": 3.0}\n')
        self.assertFalse(_layout.matches_frozen_hash(artifact, digest))
        artifact.unlink()
        self.assertFalse(_layout.matches_frozen_hash(artifact, digest))


if __name__ == "__main__":
    unittest.main()
