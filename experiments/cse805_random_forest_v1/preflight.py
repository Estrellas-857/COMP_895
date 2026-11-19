"""Bind the original splits, feature caches, comparators and RF design."""
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

from experiments.cse805_error_prediction_v1.interpretation_v1.shared import ROWS, verify_execution
from experiments.cse805_error_prediction_v1.protocol import load_protocol
from .model import BASE, DATA, INTERACTION, OUT, read, sha256, write


def main():
    load_protocol()
    verify_execution()
    seal = read(INTERACTION/"run_manifest.json")
    prior = read(INTERACTION/"results/verification.json")
    assert prior["status"]=="PASS" and prior["manifest_sha256"]==sha256(INTERACTION/"run_manifest.json")
    for name,digest in prior["artifact_sha256"].items():
        assert sha256(INTERACTION/name)==digest,name
    for name,digest in seal["inputs"].items():
        assert sha256(BASE/name)==digest,name
    assert seal["versions"]=={p:importlib.metadata.version(p) for p in seal["versions"]}
    rows=read(DATA/"features/manifest.json")["records"]
    assert len(rows)==len(ROWS)==1600 and {r["image_id"] for r in rows}=={r["image_id"] for r in ROWS}
    assert all(Path(r["path"]).is_file() and Path(r["native_cache"]["path"]).is_file() and r["device"]=="cuda" for r in rows)
    folds=read(DATA/"assignments.json")
    assert len(folds)==17 and {r["condition"] for r in ROWS}=={"fog","rain","snow","night"}
    for fold in folds:
        outer=set(fold["train_groups"])
        assert len(outer)==16 and fold["eval_group"] not in outer
        seen=[]
        for inner in fold["inner_folds"]:
            dev=set(inner["dev_groups"])
            assert dev<=outer and set(inner["fit_groups"])==outer-dev
            seen.extend(dev)
        assert sorted(seen)==sorted(outer) and len(fold["inner_folds"])==3
    new_seal=dict(frozen_sources=seal["inputs"],interaction_receipt_sha256=sha256(INTERACTION/"results/verification.json"),
                  rf_source_sha256={p:sha256(OUT/p) for p in ("protocol.json","model.py","train.py")},
                  versions=seal["versions"],outer_groups=[f["eval_group"] for f in folds],images=1600,
                  training_pixels_per_image=512,train_seed=805,qa_seed=806,forest_seed=805)
    path=OUT/"run_manifest.json"
    if path.exists():
        prior_seal=read(path)
        prior_seal.pop("frozen_utc")
        if prior_seal!=new_seal:
            raise RuntimeError("RF configuration or bound inputs have changed")
    else:
        write(path,dict(**new_seal,frozen_utc=datetime.now(timezone.utc).isoformat()))
    print("Sealed RF settings and matched original 1600 images, 17 nested splits, cache manifest and previous results.")


if __name__=="__main__":
    main()
