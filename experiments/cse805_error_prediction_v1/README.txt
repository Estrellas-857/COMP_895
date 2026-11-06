CSE 805: Learning-Based Semantic Segmentation Error Prediction

This isolated extension trains binary error classifiers over a frozen SegFormer-B0.
The segmentation backbone is never trained. The original COMP 895 artifacts are
read-only inputs. Primary evaluation uses all 1,600 official adverse train images;
the 406 official validation images are inventoried but excluded from primary work.

Read these artifacts together:
  data/metadata_train.csv              Image/GT hashes, weather, groups, old caches
  data/groups_train.csv                Counts and weather for every proxy group
  data/leakage_audit.json               Grouping ambiguities and historical overlaps
  data/assignments.json                 All outer and inner group assignments
  data/protocol.json                    Frozen original protocol
  data/metric_amendment_001.json        Before-training numerical metric amendment
  data/features/manifest.json          Reproducible samples and cache provenance
  data/models/<group>/selection.json   Every tuning candidate, failures, thresholds
  results/folds/<group>/result.json    All four methods on identical held-out pixels
  results/report/report.txt            Complete results and interpretation
  results/report/*.csv                 Group/weather/pooled metrics, deltas, QA, costs
  results/report/cases/                Post-hoc success/failure illustrations
  logs/experiment.jsonl                Append-only execution events
  logs/environment_*.json              Package, device, source and protocol hashes
  logs/preservation_after.json         Original COMP 895 preservation verification

The effective metrics are protocol.json PLUS metric_amendment_001.json. The base
protocol remains unchanged: its 10,000 uniform bins were superseded by 100,000
fixed arcsine-sqrt bins after smoke testing and before formal outer model training.
The original smoke outputs are preserved in smoke_uniform_10000. No model choice
or feature was changed by the amendment. See its recorded rationale and QA results.

Method
  Raw 1-MSP; raw entropy; logistic regression with MSP/entropy/margin; and the same
  logistic regression with predicted-class one-hot encoding. Scores increase with
  predicted error. GT supplies only targets and the semantic-valid evaluation mask.
  Bilinear float32 native logits are resized before softmax and class prediction.
  Class IDs are 0..18; ignored GT 255 is excluded. Entropy is normalized by log(19).

  Leave one of 17 conservative GoPro filename families out. Within each outer
  training population, use three grouped development folds to choose C from
  {0.01, 0.1, 1.0} by mean AP. All scaler/encoder/model fitting is inside sklearn
  pipelines and uses the corresponding training partition only. A filename family
  is a conservative chapter proxy, NOT proof of an independent drive/session.

  Train on 512 uniform valid pixels per image, without replacement (seed 805 plus
  image-ID hash). Use N_valid/512 weights and mean-one fitting weights. The separate
  8,192-pixel/image QA sample uses seed 806. Primary evaluation uses EVERY valid
  full-resolution pixel, streamed from native logits, preserving error prevalence.
  The same held-out pixels are used for all methods. No full probability volumes
  are persisted. Existing native caches are reused; only 798 missing frozen forward
  passes are generated. Outputs include 34 fitted outer-fold classifiers.

  AUROC/AP/AURC and retrospective coverage curves use fixed histogram bins; exact
  QA sample ranks document approximation discrepancies. RC discrepancies also
  include differing handling of true score ties. Report actual retrospective
  coverage, which may differ slightly from 80%. The training-derived 80% score
  quantile uses final outer-training samples only; exact held-out acceptance counts
  show actual coverage and error, with no guarantee of 80% unseen-group coverage.

Reproduction (from the project root, using its existing .venv)
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$PWD/experiments/cse805_error_prediction_v1/runtime/packages:$PWD"
  export OPENBLAS_NUM_THREADS=4
  export OMP_NUM_THREADS=4
  export CUDA_CACHE_PATH="$PWD/experiments/cse805_error_prediction_v1/runtime/cuda-cache"

  .venv/bin/python -B -m experiments.cse805_error_prediction_v1.protocol verify
  .venv/bin/python -B -m unittest experiments.cse805_error_prediction_v1.test_protocol experiments.cse805_error_prediction_v1.test_features experiments.cse805_error_prediction_v1.test_evaluation experiments.cse805_error_prediction_v1.test_run experiments.cse805_error_prediction_v1.test_report
  .venv/bin/python -B -m experiments.cse805_error_prediction_v1.run snapshot
  .venv/bin/python -B -m experiments.cse805_error_prediction_v1.run smoke --device cuda
  .venv/bin/python -B -m experiments.cse805_error_prediction_v1.run extract --device cuda
  .venv/bin/python -B -m experiments.cse805_error_prediction_v1.run evaluate --device cuda
  .venv/bin/python -B -m experiments.cse805_error_prediction_v1.report
  .venv/bin/python -B -m experiments.cse805_error_prediction_v1.visualize --device cuda
  .venv/bin/python -B -m experiments.cse805_error_prediction_v1.run preserve

The extraction phase after smoke restores the full 1,600-image feature manifest.
Hash and phase gates reject incompatible cached artifacts. Do not regenerate the
frozen metadata/protocol in place to change experiments; use a separate version.
Local ACDC data and the pinned locally cached Hugging Face checkpoint are required.
CUDA was used for this run; changing the numeric backend requires new validation.
Dependencies added for this extension are in runtime/packages, not the shared venv.
Exact versions are in logs/environment_evaluate.json. Reconstruct those isolated
packages with pip --target if needed; do not silently upgrade the shared environment.

Interpretation
  The primary endpoint is the mean paired outer-family change in accepted error
  near retrospective 80% coverage, learned method minus MSP. Negative is better.
  Also compare entropy, class ablation, every weather and every family. Report
  wins/losses and leave-one-family-out sensitivity; correlated pixels do not supply
  independent statistical observations. These data were explored in earlier work,
  and the group proxies are unverified, so this is not a pristine confirmatory test.
  Pooled cross-fitted scores combine different models and scales: descriptive only.
  This project cannot claim a new segmentation model or improved segmentation mIoU.

No RF, boosting, MLP, backbone fine-tuning, or extra model search is included.
