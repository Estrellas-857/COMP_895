"""Reproducible Stage 4A smoke run with a dataset file-access guard.

Run from the project root with .venv/bin/python. Refuses existing demo outputs.
Only the single declared validation RGB may be opened under data/ACDC.
The guard prevents GT and all training-image access during this run.
"""
import json
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
output = root / "results/stage4_reliability/stage4a"
rgb = root / "data/ACDC/rgb_anon_trainvaltest/rgb_anon/night/val/GOPR0351/GOPR0351_frame_000033_rgb_anon.png"
dataset = root / "data/ACDC"
opened = set()
blocked = []


def guard(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.is_relative_to(dataset):
        if path != rgb:
            blocked.append(str(path))
            raise RuntimeError(f"Stage 4A disallows opening this dataset file: {path}")
        opened.add(str(path))


sys.addaudithook(guard)
sys.argv = ["run_reliability_pipeline.py", "--image", str(rgb),
            "--uncertainty-method", "calibrated_msp", "--threshold", "0.10",
            "--device", "cuda", "--local-files-only", "--check-all-methods",
            "--threshold-purpose", "Engineering smoke-test threshold only; NOT a selected operating threshold",
            "--output-dir", str(output / "smoke_test")]
try:
    runpy.run_module("stage4.run_reliability_pipeline", run_name="__main__")
finally:
    audit = dict(allowed_dataset_rgb=str(rgb), opened_dataset_files=sorted(opened),
                 blocked_dataset_open_attempts=blocked,
                 ground_truth_opened=False, training_images_opened=False,
                 scope="Python file-open audit during smoke inference; preservation hashing is separate")
    (output / "smoke_test_data_access_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
assert opened == {str(rgb)} and not blocked
print("Dataset file-access audit PASS: one validation RGB only; no GT or training data.")
