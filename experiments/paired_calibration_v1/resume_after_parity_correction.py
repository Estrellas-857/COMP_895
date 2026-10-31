"""Resume the registered experiment with historical validation CUDA settings.

All originally frozen sources and artifacts remain unchanged. The amendment
is written before corrected validation inference, and binds this source, the
one-image diagnostic, the failed attempt, and the completed training fits.
"""
from datetime import datetime, timezone
from pathlib import Path

from . import run  # Establish the original offline/CUBLAS environment first.
from . import engine
from .common import OUTPUT_ROOT, PROJECT_ROOT, read_json, write_json, sha256, load_protocol
from .evaluation import evaluate
from .report import generate_report
import torch

CORRECTED = OUTPUT_ROOT / "corrected_validation"
AMENDMENT = OUTPUT_ROOT / "protocol_amendment_001.json"


def verify_amendment(protocol_digest):
    record = read_json(AMENDMENT)
    if record["original_protocol_sha256"] != protocol_digest:
        raise RuntimeError("Amendment protocol mismatch")
    for relative, expected in record["bound_files"].items():
        if sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"Amendment-bound file changed: {relative}")
    return sha256(AMENDMENT)


def freeze_amendment(protocol_digest):
    if AMENDMENT.exists():
        return verify_amendment(protocol_digest)
    if (OUTPUT_ROOT / "evaluation_complete.json").exists():
        raise RuntimeError("Correction must be registered before completed outcomes")
    probe = read_json(OUTPUT_ROOT / "parity_probe.json")
    old, new = probe
    if (old["deterministic"] is not False or new["deterministic"] is not True
            or old["direct"]["errors"] != 730438 or new["direct"]["errors"] != 730440
            or new["max_native_difference_from_cache"] != 0
            or old["native_channels_last"] or new["native_channels_last"]):
        raise RuntimeError("The correction is not supported by the recorded diagnostic")
    paths = [Path(__file__).resolve(), OUTPUT_ROOT / "parity_probe.json",
             OUTPUT_ROOT / "parity_probe_source.txt", OUTPUT_ROOT / "execution.log",
             OUTPUT_ROOT / "cache_index_val.json", OUTPUT_ROOT / "evaluation_start.json"]
    paths += [OUTPUT_ROOT / "temperatures" / f"{name}.json"
              for name in ("NightPair", "NormalPair", "Mix")]
    record = dict(
        amendment=1, created_utc=datetime.now(timezone.utc).isoformat(),
        original_protocol_sha256=protocol_digest,
        reason="Exact historical parity failed before scientific tables were published: GOPR0351_frame_000059 had 730440 versus 730438 wrong valid pixels.",
        cause="Deterministic CUDA algorithms alter native model logits relative to the historical validation pipeline. A one-image diagnostic with deterministic algorithms disabled reproduces the historical counts; native layout is contiguous in both cases.",
        correction="Validation inference only uses torch.use_deterministic_algorithms(False), matching Stage1/Stage4/Stage5. The engine device helper is temporarily wrapped for this one cache call; all original source files stay unchanged.",
        unchanged=["101 matched fitting pairs", "all three already fitted temperatures and traces",
                   "training inference and fitting numerical conventions", "model and revision",
                   "53 validation pairs", "own-GT masks", "metrics, samples, ties and coverage checkpoints",
                   "exact historical parity gates", "frozen global T and tau"],
        no_scientific_outcome_used_for_tuning=True,
        original_validation_cache_role="aborted numerical attempt, retained and excluded from scientific evaluation",
        corrected_validation_directory=str(CORRECTED),
        inference_repetition="106 validation caches regenerated once to correct the diagnosed defect; original caches retained. A single image was additionally inferred twice for diagnosis.",
        bound_files={str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in paths},
    )
    write_json(AMENDMENT, record)
    return verify_amendment(protocol_digest)


def corrected_cache(protocol_digest):
    index = CORRECTED / "cache_index_val.json"
    if index.exists():
        record = read_json(index)
        if record["protocol_sha256"] != protocol_digest:
            raise RuntimeError("Corrected cache protocol mismatch")
        rows = record["rows"]
    else:
        original_device = engine._device

        def historical_validation_device(value):
            device = original_device(value)
            torch.use_deterministic_algorithms(False)
            assert not torch.are_deterministic_algorithms_enabled()
            return device

        try:
            engine._device = historical_validation_device
            rows = engine.cache_images(run.records("val"), CORRECTED, "cuda", protocol_digest)
        finally:
            engine._device = original_device
    expected = {(r["domain"], r["pair_id"]): r for r in run.records("val")}
    if len(rows) != 106 or len({(r["domain"], r["pair_id"]) for r in rows}) != 106:
        raise RuntimeError("Unexpected corrected validation population")
    for row in rows:
        original = expected.get((row["domain"], row["pair_id"]))
        if not original or any(row.get(k) != v for k, v in original.items()):
            raise RuntimeError("Corrected cache differs from registered manifest")
        if not Path(row["cache_path"]).is_relative_to(CORRECTED):
            raise RuntimeError("Wrong validation cache location")
        if sha256(row["cache_path"]) != row["cache_sha256"]:
            raise RuntimeError("Corrected native cache hash mismatch")
    return rows


def resources():
    rows = [dict(source=str(p.relative_to(OUTPUT_ROOT)), **read_json(p))
            for root in (OUTPUT_ROOT / "resources", CORRECTED / "resources")
            for p in sorted(root.glob("*.json"))]
    for name in ("NightPair", "NormalPair", "Mix"):
        rows.append(dict(phase=name, **read_json(OUTPUT_ROOT / "temperatures" / f"{name}.json")["resources"]))
    native = list((OUTPUT_ROOT / "logits").rglob("*.npy")) + list((CORRECTED / "logits").rglob("*.npy"))
    discarded = list((OUTPUT_ROOT / "logits" / "val").rglob("*.npy"))
    maxima = {key: max((r.get(key) or 0 for r in rows), default=0) for key in
              ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "peak_process_rss_bytes")}
    write_json(OUTPUT_ROOT / "resource_summary.json", dict(
        measured_phase_seconds=sum(r.get("elapsed_seconds", 0) for r in rows), **maxima,
        new_native_cache_files=len(native), new_native_cache_bytes=sum(p.stat().st_size for p in native),
        useful_new_native_cache_files=len(native)-len(discarded),
        retained_aborted_native_cache_files=len(discarded),
        retained_aborted_native_cache_bytes=sum(p.stat().st_size for p in discarded),
        reused_adverse_fitting_logits=101, phases=rows,
        caveats=["GPU shared with existing workloads; no uncontended latency claim",
                 "includes the aborted and corrected validation cache-generation phases",
                 "excludes source inspection, unit tests, initial failed partial evaluation, two diagnostic forward passes, protocol/preservation hashing and reporting",
                 "RSS is lifetime process high-water mark; GPU reserved differs from allocated"] ))


def main():
    torch.set_num_threads(8)
    protocol, digest = load_protocol()
    amendment_digest = freeze_amendment(digest)
    if (OUTPUT_ROOT / "run_complete.json").exists():
        completion = read_json(OUTPUT_ROOT / "run_complete.json")
        if completion.get("protocol_amendment_sha256") != amendment_digest or completion["status"] != "PASS":
            raise RuntimeError("Completed correction provenance mismatch")
        corrected_cache(digest)
        if sha256(CORRECTED / "cache_index_val.json") != completion["corrected_validation_index_sha256"]:
            raise RuntimeError("Corrected index changed")
        if sha256(OUTPUT_ROOT / "evaluation_complete.json") != completion["evaluation_complete_sha256"]:
            raise RuntimeError("Evaluation completion record changed")
        for record in (completion, read_json(OUTPUT_ROOT / "evaluation_complete.json")):
            for relative, expected in {**record.get("report_hashes", {}), **record.get("output_hashes", {})}.items():
                if sha256(OUTPUT_ROOT / relative) != expected:
                    raise RuntimeError(f"Output changed: {relative}")
        run.verify_preservation()
        print("Corrected completed run verified; no inference or fitting repeated", flush=True)
        return
    run.preflight("cuda")
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    rows = corrected_cache(digest)
    assert not torch.are_deterministic_algorithms_enabled()
    verify_amendment(digest)
    start = OUTPUT_ROOT / "corrected_evaluation_start.json"
    if not start.exists():
        write_json(start, dict(started_utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=digest, protocol_amendment_sha256=amendment_digest,
            validation_deterministic_algorithms=False, cudnn_benchmark=False,
            corrected_validation_index_sha256=sha256(CORRECTED / "cache_index_val.json")))
    evaluate(rows, run.temperatures(digest), OUTPUT_ROOT, "cuda", protocol, digest)
    resources()
    generate_report(OUTPUT_ROOT)
    run.verify_preservation()
    names = ("findings.txt", "summary.json", "resource_summary.json", "calibration_comparison.png", "risk_coverage.png",
             "protocol_amendment_001.json", "corrected_evaluation_start.json")
    write_json(OUTPUT_ROOT / "run_complete.json", dict(status="PASS", protocol_sha256=digest,
        protocol_amendment_sha256=amendment_digest,
        corrected_validation_index_sha256=sha256(CORRECTED / "cache_index_val.json"),
        evaluation_complete_sha256=sha256(OUTPUT_ROOT / "evaluation_complete.json"),
        completed_utc=datetime.now(timezone.utc).isoformat(), existing_work_preserved=True,
        report_hashes={name: sha256(OUTPUT_ROOT / name) for name in names}))
    print("Completed corrected paired calibration experiment; all exact parity and preservation gates PASS", flush=True)


if __name__ == "__main__":
    main()
