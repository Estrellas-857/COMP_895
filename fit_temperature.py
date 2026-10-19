"""Stage 3B only: cache train-fit native logits and fit one shared temperature.

The NLL is convex in beta=1/T. Its first two derivatives are
E_p[z]-z_GT and Var_p[z]. A bracketed Newton root solve minimizes pooled NLL.
No validation dataset or calibration metric is consulted.
"""

import argparse
import csv
import io
import json
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from evaluate_condition import MODEL_NAME, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES, IGNORE_INDEX
from prepare_stage3b_split import ROOT, CONDITIONS, SEED, SPLIT_METHOD, sha256, write_csv, inventory, check_split

GRADIENT_TOLERANCE = 1e-7
BETA_INTERVAL_TOLERANCE = 1e-8
MAX_ITERATIONS = 50


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_splits():
    manifest = json.loads((ROOT / "split_manifest.json").read_text())
    for path, digest in manifest.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Frozen split changed: {path}")
    fit = read_csv(ROOT / "splits/calibration_fit.csv")
    reserved = read_csv(ROOT / "splits/reliability_selection.csv")
    check_split(fit + reserved, inventory())
    return fit


def preservation_check():
    manifest = json.loads((ROOT / "protected_files_before.json").read_text())
    changed = [path for path, digest in manifest.items() if sha256(path) != digest]
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")
    verify_splits()
    return (f"PASS: all {len(manifest)} pre-existing protected script/result files have unchanged SHA-256 hashes.\n"
            "PASS: frozen split CSVs and summaries have unchanged hashes.\n"
            "Stage 1, Stage 2 and Stage 3A results were not modified.\n"
            "Existing results were accessed only as opaque bytes for preservation hashes, not interpreted for fitting.\n")


def load_gt(path, expected_hash=None):
    import hashlib
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_hash and expected_hash != digest:
        raise RuntimeError(f"Calibration-fit GT changed: {path}")
    with Image.open(io.BytesIO(data)) as source:
        gt = np.array(source)
    if gt.shape != (1080, 1920):
        raise ValueError(f"Unexpected GT size: {path}, {gt.shape}")
    valid = (gt >= 0) & (gt < NUM_CLASSES) & (gt != IGNORE_INDEX)
    if np.any(~valid & (gt != IGNORE_INDEX)):
        raise ValueError(f"Unexpected GT trainIds: {path}")
    return gt, int(valid.sum()), digest


def cache_phase(device):
    rows = verify_splits()
    print(preservation_check(), flush=True)
    cache_dir = ROOT / "logit_cache"
    if cache_dir.exists():
        raise RuntimeError("Cache directory exists; refuse to overwrite cached inference")
    projected = len(rows) * (NUM_CLASSES * 128 * 256 * 4 + 128)
    free = shutil.disk_usage(ROOT).free
    print(f"Cache preflight: {len(rows)} images, {projected:,} projected bytes; {free:,} bytes free", flush=True)
    if projected > 10 * 2**30 or free < projected * 2 + 2**30:
        raise RuntimeError("Unexpected cache size or insufficient disk space")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    model.requires_grad_(False)
    cache_dir.mkdir()
    index = []
    started = time.perf_counter()
    for i, row in enumerate(rows, 1):
        with Image.open(row["rgb_path"]) as source:
            image = source.convert("RGB")
        if image.size != (1920, 1080):
            raise ValueError(f"Unexpected image size: {row['rgb_path']}")
        _, count, gt_hash = load_gt(row["gt_path"])
        inputs = processor(images=image, return_tensors="pt",
                           size={"height": INPUT_HEIGHT, "width": INPUT_WIDTH})
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            if logits.shape != (1, NUM_CLASSES, 128, 256) or logits.dtype != torch.float32:
                raise ValueError("Expected native float32 logits of shape (1, 19, 128, 256)")
            native = logits.cpu().numpy()
        if not np.isfinite(native).all():
            raise ValueError("Nonfinite native logits")
        path = cache_dir / row["condition"] / row["sequence_id"] / (Path(row["rgb_path"]).stem + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.save(handle, native, allow_pickle=False)
        if i == 1:
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), native)
        index.append({**row, "cache_path": str(path), "cache_sha256": sha256(path),
                      "rgb_sha256": sha256(row["rgb_path"]), "gt_sha256": gt_hash,
                      "num_valid_pixels": count})
        del inputs, logits, native, image
        if i == 1 or i % 50 == 0 or i == len(rows):
            print(f"Cache [{i}/{len(rows)}] elapsed={time.perf_counter()-started:.1f}s", flush=True)
    write_csv(ROOT / "cache_index.csv", index)
    metadata = dict(model=MODEL_NAME, model_revision=getattr(model.config, "_commit_hash", None),
                    processor_class=type(processor).__name__, torch=torch.__version__,
                    transformers=transformers.__version__, numpy=np.__version__,
                    num_images=len(rows), num_valid_pixels=sum(int(r["num_valid_pixels"]) for r in index),
                    cache_dtype="float32", cache_shape=[1, 19, 128, 256],
                    cache_bytes=sum(Path(r["cache_path"]).stat().st_size for r in index),
                    cache_index_sha256=sha256(ROOT / "cache_index.csv"),
                    projected_cache_bytes=projected, free_bytes_before_cache=free)
    write_json(ROOT / "cache_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    print(preservation_check(), flush=True)


def pixel_terms(logits64, target, valid, temperature):
    """All pixels in one image; no subsampling. Logits were interpolated in float32.

    Float64 scalar scaling/log-softmax/moments and reductions avoid precision loss
    in the 1-D optimizer and new argmax ties when dividing float32 logits by T.
    """
    safe = target.masked_fill(~valid, 0)
    gt_logits = logits64[0].gather(0, safe[None])[0]
    scaled = logits64 / temperature
    logp = F.log_softmax(scaled, dim=1)
    nll = -logp[0].gather(0, safe[None])[0][valid].sum()
    p = logp.exp()
    mean = (p * logits64).sum(dim=1)[0]
    gradient = (mean - gt_logits)[valid].sum()
    variance = (p * (logits64 - mean[None, None]).square()).sum(dim=1)[0]
    hessian = variance[valid].sum()
    gradient_at_zero = (logits64.mean(dim=1)[0] - gt_logits)[valid].sum()
    return tuple(float(v.item()) for v in (nll, gradient, hessian, gradient_at_zero))


class StreamingObjective:
    def __init__(self, device):
        self.device = device
        fit = verify_splits()
        self.rows = read_csv(ROOT / "cache_index.csv")
        self.metadata = json.loads((ROOT / "cache_metadata.json").read_text())
        assert sha256(ROOT / "cache_index.csv") == self.metadata["cache_index_sha256"]
        assert [{k: r[k] for k in fit[0]} for r in self.rows] == fit
        assert all(r["subset"] == "calibration_fit" for r in self.rows)
        self.evaluations = 0
        self.trace_path = ROOT / "optimization_trace.csv"
        self.fields = ["iteration_or_evaluation", "phase", "temperature", "inverse_temperature",
                       "pooled_nll", "gradient_beta", "hessian_beta", "num_valid_pixels", "elapsed_seconds"]
        with self.trace_path.open("x", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    @torch.inference_mode()
    def __call__(self, beta, phase="optimization", check_predictions=False):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("Inverse temperature must be finite and positive")
        temperature = 1 / beta
        sums = np.zeros(4, dtype=np.float64)
        total, changes = 0, 0
        by_condition = {c: {"num_images": 0, "num_valid_pixels": 0, "nll_sum": 0.0} for c in CONDITIONS}
        started = time.perf_counter()
        for i, row in enumerate(self.rows, 1):
            if sha256(row["cache_path"]) != row["cache_sha256"]:
                raise RuntimeError(f"Cached logits changed: {row['cache_path']}")
            native = np.load(row["cache_path"], allow_pickle=False)
            assert native.dtype == np.float32 and native.shape == (1, NUM_CLASSES, 128, 256)
            gt, count, _ = load_gt(row["gt_path"], row["gt_sha256"])
            assert count == int(row["num_valid_pixels"])
            logits32 = F.interpolate(torch.from_numpy(native).to(self.device),
                                     size=(1080, 1920), mode="bilinear", align_corners=False)
            # Preserve frozen interpolation precision, then evaluate the scalar
            # optimization in float64. Cache is still unmodified native float32.
            logits64 = logits32.to(torch.float64)
            target = torch.as_tensor(gt.astype(np.int64), device=self.device)
            valid = (target != IGNORE_INDEX) & (target >= 0) & (target < NUM_CLASSES)
            terms = pixel_terms(logits64, target, valid, temperature)
            if not np.isfinite(terms).all() or terms[0] < 0 or terms[2] < 0:
                raise RuntimeError("Nonfinite loss/derivatives or negative NLL/curvature")
            sums += terms
            total += count
            d = by_condition[row["condition"]]
            d["num_images"] += 1
            d["num_valid_pixels"] += count
            d["nll_sum"] += terms[0]
            if check_predictions:
                before = logits32.argmax(dim=1)[0]
                after = (logits64 / temperature).argmax(dim=1)[0]
                changes += int(((before != after) & valid).sum().item())
                del before, after
            del native, gt, logits32, logits64, target, valid
            if i % 200 == 0:
                print(f"  eval {self.evaluations+1} [{i}/{len(self.rows)}] "
                      f"T={temperature:.9f}, elapsed={time.perf_counter()-started:.1f}s", flush=True)
        assert total == self.metadata["num_valid_pixels"]
        self.evaluations += 1
        nll, gradient, hessian, gradient0 = sums / total
        row = dict(iteration_or_evaluation=self.evaluations, phase=phase, temperature=temperature,
                   inverse_temperature=beta, pooled_nll=nll, gradient_beta=gradient,
                   hessian_beta=hessian, num_valid_pixels=total, elapsed_seconds=time.perf_counter()-started)
        with self.trace_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)
        print(f"Evaluation {self.evaluations}: T={temperature:.12f}, NLL={nll:.12f}, "
              f"gradient(beta)={gradient:+.3e}, curvature={hessian:.6f}, changes={changes if check_predictions else 'not checked'}", flush=True)
        return dict(nll=float(nll), gradient=float(gradient), hessian=float(hessian),
                    gradient0=float(gradient0), num_valid_pixels=total, by_condition=by_condition,
                    prediction_changes=changes, beta=beta)


def solve_temperature(objective):
    """Safeguarded Newton on convex NLL in beta>0, starting from T=1.

    Stop at |dNLL/dbeta| <= 1e-7 or a bracket of width <= 1e-8*(1+beta).
    beta=0 is only an analytic bracketing limit (T=infinity), never a fitted T.
    """
    current = raw = objective(1.0, phase="raw_reference")
    if current["gradient0"] >= 0:
        raise RuntimeError("Optimum is at T=infinity; no finite positive temperature optimum")
    low, high = 0.0, 1.0
    while current["gradient"] < 0 and abs(current["gradient"]) > GRADIENT_TOLERANCE:
        low, high = high, high * 2
        if high > 1024:
            raise RuntimeError("Could not bracket a finite positive optimum")
        current = objective(high)
    for iteration in range(1, MAX_ITERATIONS + 1):
        beta = current["beta"]
        if abs(current["gradient"]) <= GRADIENT_TOLERANCE:
            return raw, current, iteration, "absolute beta-gradient tolerance"
        if current["gradient"] > 0:
            high = beta
        else:
            low = beta
        if high - low <= BETA_INTERVAL_TOLERANCE * (1 + beta):
            return raw, current, iteration, "inverse-temperature bracket tolerance"
        proposal = beta - current["gradient"] / current["hessian"] if current["hessian"] > 0 else float("nan")
        if not math.isfinite(proposal) or not low < proposal < high:
            proposal = (low + high) / 2
        current = objective(proposal)
    raise RuntimeError("Scalar optimizer did not converge")


def fit_phase(device):
    if (ROOT / "fitted_temperature.json").exists():
        raise RuntimeError("A fitted temperature already exists; refuse to overwrite")
    objective = StreamingObjective(device)
    raw, optimum, iterations, reason = solve_temperature(objective)
    temperature = 1 / optimum["beta"]
    final = objective(optimum["beta"], phase="final_verification", check_predictions=True)
    assert temperature > 0 and math.isfinite(temperature)
    assert final["nll"] <= raw["nll"] + 1e-10
    assert abs(final["nll"] - optimum["nll"]) <= 1e-12
    if final["prediction_changes"]:
        raise RuntimeError(f"Prediction changes require investigation: {final['prediction_changes']}")
    reduction = raw["nll"] - final["nll"]
    result = dict(temperature=temperature, fit_objective="pixel-weighted multiclass NLL (natural-log nats)",
                  fit_dataset="ACDC train/calibration_fit only, pooled fog/rain/snow/night; all valid pixels",
                  seed=SEED, split_method=SPLIT_METHOD, num_calibration_images=len(objective.rows),
                  num_calibration_valid_pixels=final["num_valid_pixels"], raw_nll=raw["nll"],
                  calibrated_nll=final["nll"], absolute_nll_reduction=reduction,
                  relative_nll_reduction=reduction/raw["nll"],
                  optimizer="safeguarded Newton root solve of convex pooled NLL derivative in beta=1/T",
                  initial_temperature=1.0, converged=True, convergence_reason=reason,
                  gradient_tolerance=GRADIENT_TOLERANCE, beta_interval_tolerance=BETA_INTERVAL_TOLERANCE,
                  max_iterations=MAX_ITERATIONS, optimizer_iterations=iterations,
                  objective_evaluations=objective.evaluations, final_gradient_beta=final["gradient"],
                  prediction_changes=final["prediction_changes"],
                  prediction_pixels_checked=final["num_valid_pixels"], validation_used=False,
                  precision="native cache and full-resolution bilinear interpolation float32; scalar scaling, log-softmax, moments and reductions float64",
                  model=MODEL_NAME, model_revision=objective.metadata["model_revision"],
                  split_manifest_sha256=sha256(ROOT / "split_manifest.json"),
                  cache_index_sha256=sha256(ROOT / "cache_index.csv"))
    report = preservation_check()
    write_json(ROOT / "fitted_temperature.json", result)
    write_csv(ROOT / "stage3b_fit_summary.csv", [result])
    diagnostics = []
    for c in CONDITIONS:
        a, b = raw["by_condition"][c], final["by_condition"][c]
        assert a["num_valid_pixels"] == b["num_valid_pixels"]
        diagnostics.append(dict(condition=c, num_images=a["num_images"],
                                num_valid_pixels=a["num_valid_pixels"], shared_temperature=temperature,
                                raw_nll=a["nll_sum"]/a["num_valid_pixels"],
                                calibrated_nll=b["nll_sum"]/b["num_valid_pixels"]))
    write_csv(ROOT / "calibration_fit_condition_diagnostics.csv", diagnostics)
    lines = ["STAGE 3B - ONE SHARED TEMPERATURE FIT ON ACDC TRAIN", ""]
    lines += [f"{key}: {value}" for key, value in result.items()]
    lines += ["", "Condition diagnostics (not used to select T):"]
    lines += [str(r) for r in diagnostics]
    lines += ["", "All fit pixels are included on every objective evaluation. No pixel subsampling.",
              "Full-resolution tensors exist for only one image at a time; native raw float32 logits are cached.",
              "Interpolation: bilinear, 1080x1920, align_corners=False, before scaling.",
              "NLL form is stable -log_softmax(z_full/T)[GT]; float64 improves scalar optimizer numerical precision.",
              "Prediction check compares original float32-logit argmax with scaled float64-logit argmax on ALL valid calibration-fit pixels.",
              "Reliability-selection data are reserved; only their inventory/path metadata are used for split integrity.",
              "No validation images, labels, logits or interpreted metrics were loaded for fitting or evaluation.",
              "No validation calibrated evaluation, threshold selection, Stage 3C/3D or Stage 4 work performed.",
              "Fitting-set NLL reduction is an optimization result, not evidence of validation generalization or improved segmentation.",
              "SciPy is not installed; the deterministic analytic-derivative safeguarded Newton solver is implemented locally.",
              "", report]
    if device.type == "cuda":
        lines.append(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated()/2**30:.3f} GiB")
    (ROOT / "stage3b_fit_summary.txt").write_text("\n".join(lines) + "\n")
    (ROOT / "preservation_check.txt").write_text(report)
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("cache", "fit"))
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 3B phase={args.phase}, device={device}, seed={SEED}", flush=True)
    (cache_phase if args.phase == "cache" else fit_phase)(device)


if __name__ == "__main__":
    main()
