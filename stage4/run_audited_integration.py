"""One unlabeled RGB integration test; block every other dataset file open."""
import json
import os
from pathlib import Path
import runpy
import sys

from stage1.dataset import ACDC_ROOT, RGB_ROOT

project = Path(__file__).resolve().parents[1]
os.chdir(project)
sys.path.insert(0, str(project))
root = project / "results/stage4_reliability/stage4c"
# Filename-only choice: lexicographically first Night validation RGB.
rgb = sorted((RGB_ROOT / "night/val").rglob("*_rgb_anon.png"))[0]
dataset = ACDC_ROOT
opened, blocked = set(), []


def guard(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.is_relative_to(dataset):
        if path != rgb:
            blocked.append(str(path))
            raise RuntimeError(f"Stage 4C forbids opening this dataset file: {path}")
        opened.add(str(path))


sys.addaudithook(guard)
sys.argv = ["run_final_reliability.py", "--image", str(rgb), "--device", "cuda",
            "--local-files-only", "--output-dir", str(root / "integration_test")]
try:
    runpy.run_module("stage4.run_final_reliability", run_name="__main__")
finally:
    (root / "integration_data_access_audit.json").write_text(json.dumps(dict(
        deterministic_selection_rule="Lexicographically first *_rgb_anon.png under night/val",
        selected_rgb=str(rgb), opened_dataset_files=sorted(opened), blocked_open_attempts=blocked,
        gt_opened=False, reliability_selection_data_opened=False,
        scope="Python file-open audit during integration; opaque preservation hashing is separate",
    ), indent=2) + "\n")
assert opened == {str(rgb)} and not blocked
print("PASS: one unlabeled RGB only; no GT or training-data reads; no parameter selection.")
