# COMP_895 — Semantic Segmentation Reliability in Adverse Conditions

This project evaluates a pretrained SegFormer-B0 model on the ACDC dataset,
studies pixel-level uncertainty and temperature calibration, and implements a
frozen reliability threshold for selective semantic segmentation.

The final pipeline uses
`nvidia/segformer-b0-finetuned-cityscapes-512-1024` at revision
`7a91500a7086b805eeb868719ea5542a3e0bdeb3`. It produces segmentation,
uncertainty, and accepted/rejected pixel masks. An accepted pixel is not a
guarantee of correctness.

## Project structure

Python scripts are organized into five stage packages. Tests and small run
drivers live alongside the stage they exercise.

```text
COMP_895/
├── stage1/              # Baseline segmentation and paired comparisons
├── stage2/              # Uncertainty and failure detection
├── stage3/              # Temperature calibration
├── stage4/              # Reliability pipeline and frozen operating point
├── stage5/              # Final evaluation, risk/coverage, failures, and runtime
├── results/             # Small results and frozen experiment artifacts
├── README.md
└── requirements.txt
```

Run scripts as modules from the project root, for example
`python -m stage1.evaluate_condition --condition fog`. This makes imports
between stages work without setting `PYTHONPATH`. Results stay in the project;
the ACDC dataset lives separately under `~/Documents/ACDC` by default.

## Setup

The existing experiment environment uses Python 3.12.3. Direct dependency
versions are recorded in `requirements.txt`; this is not a complete environment
lockfile. A fresh installation on another machine has not been verified.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The original environment used PyTorch 2.11.0 and torchvision 0.26.0 with CUDA
12.8 builds. Choose compatible PyTorch wheels for your machine. CPU execution
is also supported by the final inference interface.

## Run inference on your own image

Run commands from the project root. Replace the example image path with a real
image. The first inference run downloads the pretrained model from Hugging
Face; model weights are not stored in this repository.

```bash
python -m stage4.run_final_reliability \
  --image /path/to/your/image.png \
  --device cpu
```

Use `--device cuda` for a compatible GPU, or omit `--device` for automatic
selection. Use `--local-files-only` after the model is cached if offline
execution is required. Outputs default to
`results/stage4_reliability/stage4c/inference_outputs/<image stem>/`.
The output directory must not already exist. This interface does not require
ACDC or ground-truth labels.

Five committed artifacts are required by the frozen final interface:

- `results/stage3_calibration/stage3b/fitted_temperature.json`
- `results/stage4_reliability/stage4b/frozen_operating_point.json`
- `results/stage4_reliability/stage4b/threshold_freeze_record.json`
- `results/stage4_reliability/stage4c/final_reliability_config.json`
- `results/stage4_reliability/stage4c/final_config_freeze_record.json`

Source and artifact hashes are checked during loading. Historical experiment
records retain their original bytes, including references to the former flat
source layout. [The relocation record](stage1/source_layout.json) records the
original commit, old paths/hashes, reviewed new paths/hashes, and subsequent
dataset-location changes. Source checks accept only those recorded changes;
data and configuration still require
exact hash matches. Inference metadata records the relocation manifest's hash.
When changing a relocated implementation again, review and update its recorded
hash intentionally. `.gitattributes` disables automatic newline conversion.

## Dataset and evaluation

Obtain ACDC separately from the [official download page](https://acdc.vision.ee.ethz.ch/download)
and follow the [dataset license](https://acdc.vision.ee.ethz.ch/license).
The dataset is not included. Dataset paths are configured once in
[`stage1/dataset.py`](stage1/dataset.py). The default location is independent
of the project checkout:

```text
~/Documents/
├── nuScenes/                       # Independent dataset, if installed
└── ACDC/
    ├── rgb_anon_trainvaltest/rgb_anon/
    ├── gt_trainval/gt/
    ├── gt_trainval_ref/gt/
    └── archives/                   # Optional original download ZIP files
```

To use a different dataset location, set an absolute path before running any
stage (no source edits are needed):

```bash
export ACDC_ROOT="/path/to/ACDC"
python -m stage1.evaluate_condition --condition fog
```

`ACDC_ROOT` must point to the directory containing these three extracted
subdirectories:

```text
ACDC/
├── rgb_anon_trainvaltest/rgb_anon/
├── gt_trainval/gt/
└── gt_trainval_ref/gt/
```

Frozen split/cache CSVs and historical audit records retain `data/ACDC/...`
identifiers so their hashes and selected images stay unchanged. When a script
opens an image or label from one of those records, the shared path resolver
maps the identifier to the configured external ACDC directory. No dataset copy
or symlink is required inside the project. New split/audit identifiers use the
same portable format. Dataset access guards monitor the external directory.

The reference labels are used for paired normal/adverse comparisons. For
example, baseline evaluation for fog is:

```bash
python -m stage1.evaluate_condition --condition fog
```

Project stages are represented by the following scripts and saved results:

| Stage | Main scripts | Results |
| --- | --- | --- |
| [Stage 1: baseline and paired evaluation](stage1/) | `evaluate_condition.py`, `evaluate_paired_reference.py`, `analyze_baseline.py`, `finalize_stage1.py` | `results/*_val/`, `results/stage1_wrapup/` |
| [Stage 2: uncertainty and failure detection](stage2/) | `evaluate_uncertainty.py`, `evaluate_paired_uncertainty.py`, `evaluate_failure_detection.py`, `analyze_stage2*.py` | `results/stage2_uncertainty/` |
| [Stage 3: calibration](stage3/) | `prepare_stage3b_split.py`, `fit_temperature.py`, `evaluate_calibrated.py`, `summarize_stage3*.py` | `results/stage3_calibration/` |
| [Stage 4: reliability pipeline](stage4/) | `reliability_pipeline.py`, `select_operating_threshold.py`, `run_final_reliability.py` | `results/stage4_reliability/` |
| [Stage 5: final evaluation](stage5/) | `evaluate_frozen_operating_point.py`, `evaluate_risk_coverage.py`, `audit_systematic_failures.py`, `benchmark_final_reliability.py` | `results/stage5_final_evaluation/` |

Small numerical summaries, aggregate arrays, statistical plots, and frozen
configuration files are retained. Raw data, model weights, large intermediate
arrays, logs, and dataset-image visualizations are excluded by `.gitignore`.
Historical metadata may refer to excluded caches or original machine paths.
Scripts that consume these caches require regenerating them first. Some stages
refuse to overwrite frozen artifacts; use a separate experiment copy when
reproducing fitting or threshold selection.

## Checks without the dataset or model downloads

```bash
python -m unittest \
  stage1.test_layout stage1.test_dataset stage3.test_stage3b \
  stage4.test_reliability_pipeline stage4.test_stage4b \
  stage4.test_final_reliability stage5.test_risk_coverage \
  stage5.test_dataset_access
```

`python -m stage1.test_segformer` is a separate model smoke test that loads
pretrained weights. The two dataset access audit drivers are now
`python -m stage4.run_audited_smoke` and
`python -m stage4.run_audited_integration`; they require the local dataset,
cached model weights, and CUDA.

## Data and model attribution

- [ACDC dataset](https://acdc.vision.ee.ethz.ch/): Christos Sakaridis, Dengxin Dai,
  and Luc Van Gool, *ACDC: The Adverse Conditions Dataset with Correspondences
  for Semantic Driving Scene Understanding*, ICCV 2021.
- [Pretrained SegFormer-B0 model](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-512-1024).

Dataset and pretrained-model terms apply separately to those resources. A
license for this project's original code has not yet been selected.
