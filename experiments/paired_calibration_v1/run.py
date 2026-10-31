"""Run the isolated extension; no historical experimental main is invoked.

Example: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m
experiments.paired_calibration_v1.run all --device cuda
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from .common import PROJECT_ROOT, OUTPUT_ROOT, GLOBAL_T, read_csv, read_json, write_csv, write_json, sha256, load_protocol
from .protocol import freeze


def preflight(device):
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
                             "--format=csv,noheader"], capture_output=True, text=True, check=False)
    data = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
                nvidia_smi_returncode=result.returncode, nvidia_smi=result.stdout.strip(),
                nvidia_smi_stderr=result.stderr.strip(), torch_version=torch.__version__,
                numpy_version=np.__version__, cuda_available=torch.cuda.is_available(),
                cuda_device_count=torch.cuda.device_count(), requested_device=device,
                other_gpu_workloads_observed=True,
                timing_limitation="GPU shared with existing workloads; do not infer uncontended latency")
    if device == "cuda":
        if result.returncode or not torch.cuda.is_available():
            raise RuntimeError("NVIDIA driver/PyTorch CUDA unavailable; no automatic dependency or driver changes")
        free, total = torch.cuda.mem_get_info()
        data.update(free_cuda_bytes=free, total_cuda_bytes=total, gpu=torch.cuda.get_device_name())
        if free < 3 * 2**30:
            raise RuntimeError("Less than 3 GiB GPU headroom; wait for existing workload without interrupting it")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "resources" / f"preflight_{time.time_ns()}.json", data)
    if not (OUTPUT_ROOT / "environment_preflight.json").exists():
        write_json(OUTPUT_ROOT / "environment_preflight.json", data)
    print(f"Preflight: CUDA={data['cuda_available']}; {data['nvidia_smi']}", flush=True)
    return data


def records(split):
    # CSV optional fields are blank on rows which do not reuse original caches.
    rows = [{k: v for k, v in row.items() if v != ""} for row in read_csv(OUTPUT_ROOT / f"{split}_images.csv")]
    for row in rows:
        if "num_valid_pixels" in row:
            row["num_valid_pixels"] = int(row["num_valid_pixels"])
    return rows


def load_cache(split, protocol_sha256):
    data = read_json(OUTPUT_ROOT / f"cache_index_{split}.json")
    if data["protocol_sha256"] != protocol_sha256:
        raise RuntimeError("Cache index belongs to another protocol")
    rows = data["rows"]
    expected = {(r["domain"], r["pair_id"]): r for r in records(split)}
    if len(rows) != len(expected) or len({(r["domain"], r["pair_id"]) for r in rows}) != len(rows):
        raise RuntimeError("Duplicate or missing cached records")
    for row in rows:
        original = expected.get((row["domain"], row["pair_id"]))
        if not original or any(row.get(k) != v for k, v in original.items()):
            raise RuntimeError("Cache population/provenance differs from manifest")
    return rows


def temperatures(protocol_sha256):
    values = {"raw": 1., "global": GLOBAL_T}
    rows = []
    for name in ("NightPair", "NormalPair", "Mix"):
        data = read_json(OUTPUT_ROOT / "temperatures" / f"{name}.json")
        if data["protocol_sha256"] != protocol_sha256 or not data["converged"] or data["validation_used"]:
            raise RuntimeError("Unverified fitted temperature")
        if sha256(data["trace_path"]) != data["trace_sha256"]:
            raise RuntimeError("Temperature trace hash mismatch")
        values[name] = data["temperature"]
        rows.append(dict(name=name, temperature=data["temperature"], num_pairs=data["num_pairs"],
                         num_images=data["num_images"], raw_fit_nll=data["raw_nll"],
                         fitted_nll=data["calibrated_nll"], converged=data["converged"],
                         domain_weights=str(data["domain_weights"]), validation_used=False))
    write_csv(OUTPUT_ROOT / "fitted_temperatures.csv", rows)
    return values


def verify_preservation():
    before_path = OUTPUT_ROOT / "preservation_before.json"
    if not before_path.exists():
        raise RuntimeError("Missing before-run full protected-file snapshot")
    before = read_json(before_path)["files"]
    started = time.perf_counter()
    problems = []
    for relative, record in before.items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            problems.append(relative)
    after_names = {str(p.relative_to(PROJECT_ROOT)) for i in range(1, 6)
                   for p in (PROJECT_ROOT / f"stage{i}").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts}
    after_names |= {str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "results").rglob("*")
                    if p.is_file() and "research_extension" not in p.parts}
    added = sorted(after_names - set(before))
    if problems or added:
        raise RuntimeError(f"Protected tree changed: {problems}; added={added}")
    summary = dict(status="PASS", protected_files=len(before),
                   protected_bytes=sum(r["size"] for r in before.values()),
                   changed_files=[], added_protected_files=[], elapsed_seconds=time.perf_counter()-started,
                   scope="all pre-existing Stage1–5 source/support files excluding Python bytecode, all old results including caches, README/requirements/git configuration")
    path = OUTPUT_ROOT / "preservation_after.json"
    if path.exists():
        return read_json(path)
    write_json(path, summary)
    return summary


def resource_summary():
    rows = [read_json(p) for p in sorted((OUTPUT_ROOT / "resources").glob("*.json"))]
    for p in sorted((OUTPUT_ROOT / "temperatures").glob("*.json")):
        data = read_json(p)
        if "resources" in data:
            rows.append(dict(phase=p.stem, **data["resources"]))
    elapsed = sum(r.get("elapsed_seconds", 0) for r in rows)
    allocated = [r["peak_cuda_allocated_bytes"] for r in rows if r.get("peak_cuda_allocated_bytes") is not None]
    reserved = [r["peak_cuda_reserved_bytes"] for r in rows if r.get("peak_cuda_reserved_bytes") is not None]
    rss = [r["peak_process_rss_bytes"] for r in rows if r.get("peak_process_rss_bytes") is not None]
    native = list((OUTPUT_ROOT / "logits").rglob("*.npy"))
    write_json(OUTPUT_ROOT / "resource_summary.json", dict(
        measured_phase_seconds=elapsed, peak_cuda_allocated_bytes=max(allocated, default=0),
        peak_cuda_reserved_bytes=max(reserved, default=0), peak_process_rss_bytes=max(rss, default=0),
        new_native_cache_files=len(native), new_native_cache_bytes=sum(p.stat().st_size for p in native),
        reused_adverse_fitting_logits=101, phases=rows,
        caveats=["single session with concurrent GPU workload", "no steady-state latency claim",
                 "RSS is process lifetime high-water mark; GPU reserved differs from allocated",
                 "phase sums exclude source inspection/unit tests/protocol hashing/preservation hashes"] ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "cache-fit", "fit", "cache-val", "evaluate", "report", "preserve", "all"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.phase in ("all", "report") and (OUTPUT_ROOT / "run_complete.json").exists():
        _, digest = load_protocol()
        record = read_json(OUTPUT_ROOT / "run_complete.json")
        if record["protocol_sha256"] != digest or record["status"] != "PASS":
            raise RuntimeError("Completed run provenance mismatch")
        evaluation = read_json(OUTPUT_ROOT / "evaluation_complete.json")
        for relative, expected in evaluation["output_hashes"].items():
            if sha256(OUTPUT_ROOT / relative) != expected:
                raise RuntimeError(f"Completed evaluation changed: {relative}")
        for relative, expected in record.get("report_hashes", {}).items():
            if sha256(OUTPUT_ROOT / relative) != expected:
                raise RuntimeError(f"Completed report changed: {relative}")
        verify_preservation()
        print("Completed run verified; no inference or fitting repeated", flush=True)
        return
    if args.phase == "freeze":
        protocol, digest = freeze()
        print(f"Frozen protocol {digest}:101 fitting pairs,53 validation pairs; Mix preregistered", flush=True)
        return
    if args.phase == "preserve":
        print(verify_preservation(), flush=True)
        return
    if args.phase == "all":
        preflight(args.device)
        protocol, digest = freeze()
    else:
        protocol, digest = load_protocol()
        if args.phase != "report":
            preflight(args.device)
    from .engine import cache_images, fit_temperatures
    from .evaluation import evaluate
    if args.phase in ("cache-fit", "all"):
        cache_images(records("fit"), OUTPUT_ROOT, args.device, digest)
    if args.phase in ("fit", "all"):
        fit_temperatures(load_cache("fit", digest), OUTPUT_ROOT, args.device, digest)
    if args.phase in ("cache-val", "all"):
        temperatures(digest)  # All training-only fits must finish before validation is accessed.
        load_protocol()
        cache_images(records("val"), OUTPUT_ROOT, args.device, digest)
    if args.phase in ("evaluate", "all"):
        load_protocol()
        evaluate(load_cache("val", digest), temperatures(digest), OUTPUT_ROOT, args.device, protocol, digest)
    if args.phase in ("report", "all"):
        completion = read_json(OUTPUT_ROOT / "evaluation_complete.json")
        if completion["status"] != "PASS" or completion["protocol_sha256"] != digest:
            raise RuntimeError("Scientific report requires a completed verified evaluation")
        resource_summary()
        from .report import generate_report
        generate_report(OUTPUT_ROOT)
        verify_preservation()
        write_json(OUTPUT_ROOT / "run_complete.json", dict(status="PASS", protocol_sha256=digest,
                   completed_utc=datetime.now(timezone.utc).isoformat(), existing_work_preserved=True,
                   report_hashes={name: sha256(OUTPUT_ROOT / name) for name in
                                  ("findings.txt", "summary.json", "resource_summary.json",
                                   "calibration_comparison.png", "risk_coverage.png")}))
    print(f"Completed phase {args.phase}; outputs:{OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
