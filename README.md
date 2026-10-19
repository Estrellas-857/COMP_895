# COMP_895 — Semantic Segmentation Reliability in Adverse Conditions

This project evaluates a pretrained SegFormer-B0 model on the ACDC dataset,
studies pixel-level uncertainty and temperature calibration, and implements a
frozen reliability threshold for selective semantic segmentation.

The final pipeline uses
`nvidia/segformer-b0-finetuned-cityscapes-512-1024` at revision
`7a91500a7086b805eeb868719ea5542a3e0bdeb3`. It produces segmentation,
uncertainty, and accepted/rejected pixel masks. An accepted pixel is not a
guarantee of correctness.

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
python run_final_reliability.py \
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

Source and artifact hashes are checked during loading. Preserve the original
bytes of frozen files; `.gitattributes` disables automatic newline conversion.

## Dataset and evaluation

Obtain ACDC separately from the [official download page](https://acdc.vision.ee.ethz.ch/download)
and follow the [dataset license](https://acdc.vision.ee.ethz.ch/license).
The dataset is not included. Extract the downloads into the layout expected by
the evaluation scripts:

```text
data/ACDC/
├── rgb_anon_trainvaltest/rgb_anon/
├── gt_trainval/gt/
└── gt_trainval_ref/gt/
```

The reference labels are used for paired normal/adverse comparisons. For
example, baseline evaluation for fog is:

```bash
python evaluate_condition.py --condition fog
```

Project stages are represented by the following scripts and saved results:

| Stage | Main scripts | Results |
| --- | --- | --- |
| Baseline and paired evaluation | `evaluate_condition.py`, `evaluate_paired_reference.py` | `results/*_val/`, `results/stage1_wrapup/` |
| Uncertainty and failure detection | `evaluate_uncertainty.py`, `evaluate_failure_detection.py` | `results/stage2_uncertainty/` |
| Calibration | `prepare_stage3b_split.py`, `fit_temperature.py`, `evaluate_calibrated.py` | `results/stage3_calibration/` |
| Reliability pipeline | `reliability_pipeline.py`, `select_operating_threshold.py`, `run_final_reliability.py` | `results/stage4_reliability/` |
| Final evaluation | `evaluate_frozen_operating_point.py`, `evaluate_risk_coverage.py`, `audit_systematic_failures.py`, `benchmark_final_reliability.py` | `results/stage5_final_evaluation/` |

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
  test_stage3b test_reliability_pipeline test_stage4b \
  test_final_reliability test_risk_coverage
```

`test_segformer.py` is a separate model smoke test that loads pretrained weights.

## Data and model attribution

- [ACDC dataset](https://acdc.vision.ee.ethz.ch/): Christos Sakaridis, Dengxin Dai,
  and Luc Van Gool, *ACDC: The Adverse Conditions Dataset with Correspondences
  for Semantic Driving Scene Understanding*, ICCV 2021.
- [Pretrained SegFormer-B0 model](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-512-1024).

Dataset and pretrained-model terms apply separately to those resources. A
license for this project's original code has not yet been selected.
