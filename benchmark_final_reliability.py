"""Stage 5D: fixed RGB-only timing boundaries, no fitting or implementation tuning.

Reuse the frozen Stage 4C instance and Stage 4A inference/uncertainty primitives.
Exclude wrapper provenance I/O and metadata summaries from the measured region.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from run_final_reliability import FinalReliabilityPipeline, load_final_config
from reliability_pipeline import apply_reliability_threshold
from evaluate_condition import RGB_ROOT

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / "results/stage5_final_evaluation/stage5d"
CONDITIONS = ("fog", "rain", "snow", "night")
REPEATS = 3
WARMUPS = 5


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RGBOnlyAudit:
    def __init__(self, paths):
        self.allowed = {p.resolve() for p in paths}
        self.dataset = (PROJECT / "data").resolve()
        self.opened, self.blocked = set(), []
        sys.addaudithook(self.hook)

    def hook(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(self.dataset):
            if path not in self.allowed:
                self.blocked.append(str(path))
                raise RuntimeError(f"Stage 5D permits only selected validation RGB reads: {path}")
            self.opened.add(str(path.relative_to(PROJECT)))


@torch.inference_mode()
def inference_region(core, rgb, kind):
    """Measured computation: RGB preprocessing through CPU-resident output maps.

    No artifact reads, RGB decoding, coverage summaries, GT, or file writing.
    Both branches use the same full-resolution raw-logit argmax and transfer.
    Frozen primitive validation checks remain included, without optimization.
    """
    logits = core.infer_logits(rgb)
    if kind == "baseline":
        return {"segmentation": logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)}
    uncertainty, confidence = core.compute_uncertainty(logits)
    segmentation = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    accepted = apply_reliability_threshold(uncertainty, core.threshold)
    return dict(segmentation=segmentation, uncertainty=uncertainty, confidence=confidence,
                reliability_mask=accepted, rejected_mask=~accepted)


def statistics(values):
    v = np.array(values, dtype=np.float64)
    return dict(mean_latency_ms=float(v.mean()), median_latency_ms=float(np.median(v)),
        std_latency_ms=float(v.std(ddof=0)), min_latency_ms=float(v.min()), max_latency_ms=float(v.max()),
        approximate_fps=float(1000 / v.mean()))


def wrap_up(runtime, memory, config):
    sources = {
        "baseline": "results/baseline_analysis/condition_summary.csv",
        "paired": "results/paired_normal_vs_adverse/paired_condition_summary.csv",
        "failure_characterization": "results/stage2_uncertainty/stage2b_analysis/stage2b_summary.csv",
        "raw_discrimination": "results/stage2_uncertainty/failure_detection/stage2d_analysis/stage2d_summary.csv",
        "calibration": "results/stage3_calibration/stage3c/stage3c_summary/stage3c_combined_summary.csv",
        "calibrated_discrimination": "results/stage3_calibration/stage3d/stage3d_summary/stage3d_combined_summary.csv",
        "selective_prediction": "results/stage5_final_evaluation/stage5a/stage5a_combined_summary.csv",
        "risk_coverage": "results/stage5_final_evaluation/stage5b/risk_at_target_coverages.csv",
        "target_audit": "results/stage5_final_evaluation/stage5c/stage5c_combined_summary.csv",
    }
    data = {key: read_csv(PROJECT / path) for key, path in sources.items()}
    lines = ["COMP 895 — final measured experimental results", "",
        "All prior-stage results below are read from frozen artifacts; no experiment was rerun.", "",
        "A. Segmentation baseline (full validation; mIoU and accuracy are percentages)",
        "mIoU_union averages classes with nonzero prediction/GT union; mIoU_GT averages classes present in GT.",
        "Condition | images | mIoU_union (%) | mIoU_GT (%) | pixel accuracy (%)"]
    for r in data["baseline"]:
        lines.append(f"{r['condition']} | {r['images']} | {100*float(r['miou_union']):.4f} | {100*float(r['miou_gt']):.4f} | {100*float(r['pixel_accuracy']):.4f}")
    lines += ["", "Paired normal/adverse evaluation (separate paired subset; delta = adverse minus normal)",
              "Condition | pairs | normal mIoU_union (%) | adverse mIoU_union (%) | delta (percentage points)"]
    for r in data["paired"]:
        lines.append(f"{r['condition']} | {r['pairs']} | {100*float(r['normal_miou_union']):.4f} | {100*float(r['adverse_miou_union']):.4f} | {100*float(r['miou_union_delta']):+.4f}")
    lines += ["", "B. Failure characterization",
              "Historical Stage 2 P(raw confidence >= 0.90 | incorrect), NOT P(incorrect | high confidence):"]
    for r in data["failure_characterization"]:
        lines.append(f"{r['condition']}: {100*float(r['high_conf_wrong_fraction']):.4f}%")
    lines += ["", "Stage 2D raw failure discrimination (1000-bin histogram approximation; incorrect = positive)",
              "Condition | method | AUROC | AUPRC | random PR baseline (error prevalence)"]
    for r in data["raw_discrimination"]:
        lines.append(f"{r['condition']} | {r['method']} | {float(r['auroc']):.6f} | {float(r['auprc']):.6f} | {float(r['positive_prevalence']):.6f}")
    lines += ["", "C. Calibration",
        f"One shared temperature T={config['temperature']} was fitted only on train/calibration_fit.",
        "Condition | raw ECE | calibrated ECE | raw NLL (nats) | calibrated NLL (nats)"]
    for r in data["calibration"]:
        lines.append(r['condition'] + " | " + " | ".join(f"{float(r[k]):.6f}" for k in ("raw_ece", "calibrated_ece", "raw_nll", "calibrated_nll")))
    lines += ["NLL improved for all four conditions; ECE improved for Snow/Night and increased for Fog/Rain.",
              "Semantic predictions did not change. Calibration and segmentation accuracy are different properties.", "",
              "Stage 3D changes in failure discrimination (calibrated minus frozen raw; absolute units)",
              "Condition | MSP AUROC delta | MSP AUPRC delta | entropy AUROC delta | entropy AUPRC delta"]
    for r in data["calibrated_discrimination"]:
        lines.append(r['condition'] + " | " + " | ".join(f"{float(r[k]):+.6f}" for k in ("msp_auroc_change", "msp_auprc_change", "entropy_auroc_change", "entropy_auprc_change")))
    lines += ["Calibration did not uniformly improve discrimination; calibrated entropy decreased both metrics in all four conditions.",
        "Historical Stage 2 used softmax argmax: one Rain and one Snow float32 tie differed from canonical logit argmax.",
        "Frozen historical values remain unchanged; Stages 5A/B use canonical logit-argmax correctness. Small differences are not significance claims.", "",
        "D. Final frozen reliability system",
        f"calibrated_msp; T={config['temperature']}; tau={config['threshold']}; accept uncertainty <= tau.",
        "Threshold selected on reserved train data before validation; Stage 5 did not retune it."]
    pooled = next(r for r in data["selective_prediction"] if r['condition'] == 'pooled')
    for k in ("valid_coverage", "baseline_risk", "accepted_risk", "risk_reduction_absolute", "risk_reduction_relative", "error_recall", "rejection_precision"):
        suffix = " percentage points" if k == 'risk_reduction_absolute' else "%"
        lines.append(f"Pooled {k}: {100*float(pooled[k]):.6f}{suffix}")
    lines += ["", "E. Pooled calibrated-MSP risk–coverage (1000-bin descriptive approximation)",
              "Requested coverage (%) | actual histogram coverage (%) | accepted risk (%)"]
    for r in data["risk_coverage"]:
        if r['condition'] == 'pooled' and r['method'] == 'calibrated_msp':
            lines.append(" | ".join(f"{100*float(r[k]):.6f}" for k in ("requested_coverage", "actual_histogram_coverage", "risk")))
    lines += ["These descriptive points do not select or replace the deployed threshold.", "",
              "F. Representative systematic failures", "Case | target errors rejected (%) | target errors still accepted (%)"]
    for r in data["target_audit"]:
        lines.append(f"{r['case']} | {100*float(r['target_error_recall']):.6f} | {100*float(r['target_error_remaining_fraction']):.6f}")
    lines += ["Both target classes retain zero recall and 100% accepted target-region risk; rejected pixels were not corrected.", "",
              "G. Computational cost (Stage 5D, this RTX 3060 environment)"]
    lines += runtime.splitlines()[1:] + [""] + memory.splitlines()
    lines += ["", "Final interpretation",
        "Adverse conditions reduce segmentation performance in the measured paired data; Night remains the hardest evaluated condition.",
        "Uncertainty carries failure-discrimination information, but systematic high-confidence errors remain.",
        "Probability calibration changes confidence agreement; it does not change semantic predictions and does not uniformly improve failure ranking.",
        "The fixed rejection mechanism retains a lower-error subset at the cost of coverage; rejected predictions are abstentions.",
        "The two representative systematic failures are only partly rejected. These results do not establish real-world autonomous-driving safety.",
        "No model, temperature, uncertainty definition, final method, or operating point was changed. Experimental work stops after Stage 5D.", "",
        "Source artifacts:"]
    lines.extend(f"{key}: {path}" for key, path in sources.items())
    (ROOT / "final_experimental_summary.txt").write_text("\n".join(lines) + "\n")
    (ROOT / "summary_sources.json").write_text(json.dumps({key: dict(path=path, sha256=sha256(PROJECT/path)) for key,path in sources.items()},indent=2)+"\n")


def main():
    assert (ROOT / "protected_files_before.json").is_file()
    assert not (ROOT / "runtime_per_image.csv").exists(), "Do not overwrite completed measurements"
    config, config_hash = load_final_config()
    selected = []
    for condition in CONDITIONS:
        paths = sorted((PROJECT / RGB_ROOT / condition / "val").rglob("*_rgb_anon.png"), key=lambda p:(p.name,str(p)))[:5]
        assert len(paths) == 5
        selected.extend((condition,p) for p in paths)
    audit = RGBOnlyAudit([p for _,p in selected])
    images, subset = [], []
    for condition, path in selected:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
        assert rgb.size == (1920,1080)
        images.append((condition,path,rgb))
        subset.append(dict(condition=condition,rgb_path=str(path.relative_to(PROJECT)),width=rgb.width,height=rgb.height,rgb_sha256=sha256(path)))
    write_csv(ROOT / "test_subset.csv", subset)
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0)
    assert gpu == "NVIDIA GeForce RTX 3060", gpu
    pipeline = FinalReliabilityPipeline(device="cuda",local_files_only=True)
    core = pipeline._pipeline
    assert core.device.type == "cuda" and next(core.model.parameters()).is_cuda and not core.model.training
    print(f"GPU: {gpu}; batch=1; 20 RGB images, 5/condition; T={config['temperature']}; tau={config['threshold']}",flush=True)
    # Verify the timed core against the unchanged final wrapper outside timing.
    expected = pipeline.predict(images[0][2])
    actual = inference_region(core, images[0][2], "final_reliability")
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
    del actual, expected
    for _ in range(WARMUPS):
        for kind in ("baseline","final_reliability"):
            output = inference_region(core, images[0][2], kind)
            del output
    torch.cuda.synchronize()
    print("Parity with frozen wrapper PASS. Five warm-ups per pipeline complete and excluded.",flush=True)
    records = []
    # Repeats traverse all images; alternating AB/BA reduces fixed ordering bias.
    for repeat in range(REPEATS):
        for index,(condition,path,rgb) in enumerate(images):
            order = ("baseline","final_reliability") if (repeat+index)%2==0 else ("final_reliability","baseline")
            pair = {}
            for position,kind in enumerate(order):
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = perf_counter()
                result = inference_region(core,rgb,kind)
                torch.cuda.synchronize()
                latency = (perf_counter()-start)*1000
                peak = torch.cuda.max_memory_allocated()
                assert latency > 0 and np.isfinite(latency)
                pair[kind] = result
                records.append(dict(condition=condition,image=path.name,repeat=repeat+1,pipeline=kind,
                    pair_position=position+1,latency_ms=latency,peak_allocated_bytes=peak,
                    resident_allocated_bytes_before_run=start_memory,batch_size=1,width=rgb.width,height=rgb.height))
            np.testing.assert_array_equal(pair['baseline']['segmentation'],pair['final_reliability']['segmentation'])
            del result,pair
        print(f"Timed repeat {repeat+1}/{REPEATS} complete: all 20 RGB images, both pipelines.",flush=True)
    write_csv(ROOT/'runtime_per_image.csv',records)
    summaries, memories = [], []
    for kind in ("baseline","final_reliability"):
        rows = [r for r in records if r['pipeline']==kind]
        summaries.append(dict(pipeline=kind,num_images=20,repeats_per_image=REPEATS,num_measurements=len(rows),**statistics([r['latency_ms'] for r in rows])))
        peak = max(r['peak_allocated_bytes'] for r in rows)
        memories.append(dict(pipeline=kind,peak_allocated_bytes=peak,peak_allocated_mib=peak/1024**2,
            mean_run_peak_allocated_mib=float(np.mean([r['peak_allocated_bytes'] for r in rows]))/1024**2,
            min_resident_allocated_bytes=min(r['resident_allocated_bytes_before_run'] for r in rows),
            max_resident_allocated_bytes=max(r['resident_allocated_bytes_before_run'] for r in rows)))
    write_csv(ROOT/'runtime_summary.csv',summaries)
    write_csv(ROOT/'memory_summary.csv',memories)
    baseline,final = summaries
    delta = final['mean_latency_ms']-baseline['mean_latency_ms']
    memory_delta = memories[1]['peak_allocated_bytes']-memories[0]['peak_allocated_bytes']
    overhead = dict(latency_overhead_ms=delta,latency_overhead_fraction=delta/baseline['mean_latency_ms'],
        memory_overhead_bytes=memory_delta,memory_overhead_mib=memory_delta/1024**2,
        memory_overhead_fraction=memory_delta/memories[0]['peak_allocated_bytes'])
    (ROOT/'overhead_summary.json').write_text(json.dumps(overhead,indent=2)+'\n')
    runtime_text = "Stage 5D runtime measurement\n"
    for r in summaries:
        runtime_text += (f"{r['pipeline']}: mean={r['mean_latency_ms']:.6f} ms; median={r['median_latency_ms']:.6f} ms; "
            f"std={r['std_latency_ms']:.6f} ms; min={r['min_latency_ms']:.6f} ms; max={r['max_latency_ms']:.6f} ms; FPS={r['approximate_fps']:.6f}.\n")
    runtime_text += (f"Mean latency overhead={delta:.6f} ms ({100*overhead['latency_overhead_fraction']:.6f}%).\n"
        "20 RGBs: lexicographically first 5 filenames per Fog/Rain/Snow/Night validation subset (ties by path); exact manifest in test_subset.csv.\n"
        "Five warm-up runs PER pipeline; three timed repeats per image; 60 observations per pipeline, paired alternating AB/BA order.\n"
        "perf_counter wall-clock timing with CUDA synchronization immediately before/after each measured region; batch=1, original 1920x1080.\n"
        "Statistics pool all 60 observations equally; population standard deviation (ddof=0). FPS=1000/mean_ms, sequential batch-one approximation.\n"
        "Both: predecoded PIL RGB -> frozen processor 512x1024 -> input GPU transfer -> model -> float32 bilinear logits to original size -> raw-logit argmax -> CPU uint8 map.\n"
        "Final adds frozen float64 temperature scaling/softmax, MSP/confidence CPU transfers, CPU <=tau comparison and accepted/rejected masks.\n"
        "Frozen primitive validation checks included; no reduced-precision substitution or inference optimization.\n"
        "Excluded: RGB file decoding, model loading/downloads, wrapper artifact hashing, metadata summaries, GT, evaluation metrics, plotting and serialization.\n"
        "These are the requested computational pipeline regions, not complete wrapper-call/file-I/O latency. Outputs match the unchanged final wrapper exactly.\n"
        "Stage 1 reported approximately 20–21 ms under its earlier inference timing definition; Stage 5D uses the explicitly defined end-to-end regions above.\n")
    memory_text = ("GPU memory is peak PyTorch allocated memory (includes resident model), not reserved allocator memory or total GPU/system usage.\n"
        + "\n".join(f"{r['pipeline']}: {r['peak_allocated_mib']:.6f} MiB ({r['peak_allocated_bytes']} bytes), maximum over the 60 measured runs." for r in memories)
        + f"\nIncrease={overhead['memory_overhead_mib']:.6f} MiB ({100*overhead['memory_overhead_fraction']:.6f}%).\n"
        "Peak counters reset before every run outside timing; previous GPU outputs released; CPU outputs retained only until pair parity check.\n")
    (ROOT/'runtime_summary.txt').write_text(runtime_text+"\n"+memory_text)
    wrap_up(runtime_text,memory_text,config)
    manifest=json.loads((ROOT/'protected_files_before.json').read_text())
    changed=[p for p,h in manifest.items() if not (PROJECT/p).is_file() or sha256(PROJECT/p)!=h]
    assert not changed, changed
    assert load_final_config()==(config,config_hash)
    assert len(audit.opened)==20 and not audit.blocked and len(records)==120
    report=dict(status="PASS",gpu=gpu,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),cpu_threads=torch.get_num_threads(),
        model_eval=not core.model.training,inference_mode=True,batch_size=1,processor_size=[512,1024],
        original_image_size=[1080,1920],temperature=config['temperature'],threshold=config['threshold'],
        uncertainty_method=config['uncertainty_method'],config_hash=config_hash,
        temperature_artifact_hash=config['temperature_artifact_hash'],threshold_artifact_hash=config['threshold_artifact_hash'],
        rgb_files_opened=sorted(audit.opened),blocked_dataset_opens=audit.blocked,gt_loaded=False,
        reserved_train_used=False,full_wrapper_output_parity=True,paired_canonical_segmentation_identical=True,
        warmups_per_pipeline=WARMUPS,timed_repeats_per_image=REPEATS,observations_per_pipeline=60,
        synchronization_before_and_after=True,frozen_precision_preserved=True,
        no_fitting=True,no_parameter_changes=True,
        limitations="Single machine/session, serial batch-one throughput estimate; wall-clock includes CPU preprocessing and output transfers. No statistical significance or real-time safety claim.")
    (ROOT/'evaluation_audit.txt').write_text(json.dumps(report,indent=2)+'\n')
    preservation=f"PASS: all {len(manifest)} pre-existing protected files retain SHA-256 hashes.\nStages 1–5C results, frozen implementation/config, temperature, threshold and train-split artifacts unchanged.\nExisting cached arrays hashed as opaque bytes only; no train RGB/GT loaded.\n"
    (ROOT/'preservation_check.txt').write_text(preservation)
    (ROOT/'stage5d_summary.txt').write_text(runtime_text+"\n"+memory_text+"\n"+preservation+"\nGT-free timing checks PASS. Prior results summarized in final_experimental_summary.txt. No fitting/tuning/new experimental method; project experimental pipeline complete.\n")
    print(runtime_text+"\n"+memory_text+"\n"+preservation,flush=True)


if __name__ == '__main__':
    main()
