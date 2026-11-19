"""Seal sources and evaluation choices before inspecting interaction results."""
import importlib.metadata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.cse805_error_prediction_v1.interpretation_v1.shared import ROWS, verify_execution
from experiments.cse805_error_prediction_v1.protocol import load_protocol
from .model import BASE, DATA, OUT, read, sha256, write


def main():
    load_protocol()
    verify_execution()
    folds = read(DATA / "assignments.json")
    groups = {r["recording_group"] for r in ROWS}
    assert len(groups) == len(folds) == 17 and len(ROWS) == 1600
    assert dict(Counter(r["condition"] for r in ROWS)) == dict(fog=400, night=400, rain=400, snow=400)
    assert {f["eval_group"] for f in folds} == groups
    for f in folds:
        training = groups - {f["eval_group"]}
        assert set(f["train_groups"]) == training and len(f["inner_folds"]) == 3
        devs = []
        for i in f["inner_folds"]:
            dev = set(i["dev_groups"])
            assert set(i["fit_groups"]) == training-dev and dev <= training
            devs.extend(dev)
        assert sorted(devs) == sorted(training)
    complete = read(DATA / "features/complete.json")
    manifest = DATA / "features/manifest.json"
    assert sha256(manifest) == complete["manifest_sha256"]
    records = read(manifest)["records"]
    assert {r["image_id"] for r in records} == {r["image_id"] for r in ROWS} and len(records) == 1600
    assert all(Path(r["path"]).is_file() and Path(r["native_cache"]["path"]).is_file() for r in records)
    assert all(r["device"] == "cuda" for r in records)
    assert sum(r["num_valid"] for r in records) == 3059663748
    inputs = [DATA/n for n in ("metadata_train.csv", "groups_train.csv", "assignments.json",
              "protocol.json", "metric_amendment_001.json", "features/manifest.json")]
    inputs += [BASE/n for n in ("results/report/group_metrics.csv", "interpretation_v1/tables/pixel_class_acceptance.csv",
              "interpretation_v1/tables/metrics_policy_coverage.csv", "logs/completion.json")]
    versions = {p:importlib.metadata.version(p) for p in read(BASE/"logs/environment_evaluate.json")["versions"]}
    assert versions == read(BASE/"logs/environment_evaluate.json")["versions"]
    seal = dict(analysis_plan_sha256=sha256(OUT/"protocol.json"), model_sha256=sha256(OUT/"model.py"),
                train_sha256=sha256(OUT/"train.py"), inputs={str(p.relative_to(BASE)):sha256(p) for p in inputs},
                versions=versions, outer_groups=[f["eval_group"] for f in folds], num_images=len(records),
                train_seed=805, qa_seed=806, native_cache_count=len(records))
    path = OUT/"run_manifest.json"
    if path.exists():
        previous = read(path)
        previous.pop("frozen_utc")
        if seal != previous:
            raise RuntimeError("Existing experiment manifest differs")
    else:
        write(path, dict(**seal, frozen_utc=datetime.now(timezone.utc).isoformat()))
    print("PASS: frozen 17-fold assignments, 1600 cache records, original sources and package versions.")


if __name__ == "__main__":
    main()
