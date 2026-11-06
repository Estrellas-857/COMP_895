Post-hoc interpretation pass for the completed CSE805 Option B experiment

Read interpretation_report.txt for the scientific findings and presentation_plan.txt
for the seven-part story and minimal presentation evidence. Original experiment
files are read-only inputs; all new artifacts live in this interpretation_v1 tree.

Completed analyses
* coefficient_analysis.py: inspect all 34 pipelines; weighted training/held-out
  class profiles; within-class ranking; exploratory high-confidence subsets.
* metric_sensitivity.py: consistent tie/bin policies on identical saved QA pixels,
  including a within-/cross-group Night AUROC pair decomposition.
* metric_largest_bins.py: locate and characterize the largest Night class-aware
  100k score bins; dense road-associated clusters are not zero-score saturation.
* pixel_diagnostics.py: all 800 Night/Fog images, 216 existing operating points;
  full-pixel accepted-class and GT->prediction confusion counts. Native caches
  are required and backbone construction is explicitly forbidden.
* figures_and_tables.py, coefficient_figure.py: source-linked tables and two
  scientific figures. The original primary per-group figure is reused unchanged.
* verify_preservation.py: rehash 5,094 original extension files and 3,635 protected
  COMP895 files against pre-analysis/frozen snapshots.

Reproduction from the project directory
Use the existing virtual environment and local dependency directory:

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/experiments/cse805_error_prediction_v1/runtime/packages:$PWD"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

Run a module, for example:
.venv/bin/python -B -m experiments.cse805_error_prediction_v1.interpretation_v1.metric_sensitivity

Substitute coefficient_analysis, metric_largest_bins, pixel_diagnostics,
figures_and_tables, coefficient_figure, or verify_preservation as needed.
pixel_diagnostics uses the frozen CUDA extraction backend by default; it requires
access to the same GPU, cached logits, labels, and existing model artifacts.
Run figures_and_tables after pixel_diagnostics and coefficient_figure after
coefficient_analysis. No interpretation module fits a model. Do not invoke the
parent experiment's training/evaluation runner to reproduce this pass.

Focused verification:
.venv/bin/python -B -m unittest experiments.cse805_error_prediction_v1.interpretation_v1.test_metric_sensitivity experiments.cse805_error_prediction_v1.interpretation_v1.test_coefficient_analysis experiments.cse805_error_prediction_v1.interpretation_v1.test_pixel_diagnostics -v

Logs include input/output and source hashes, exact operating-point count parity,
coefficient reconstruction checks, numerical-policy definitions, and timings.
logs/completion.json binds this delivered interpretation bundle. Re-running an
analysis updates its new interpretation outputs only; it makes the old completion
manifest stale, so regenerate that manifest before treating a rerun as this bundle.

Limits
Class-profile and metric-policy tables are weighted sample estimates; full-pixel
tables are named pixel_*. Both are descriptive post-hoc analyses, not new primary
results or causal tests. Ground-truth classes in confusion tables are analysis
labels only. Filename-derived groups are not proven independent sessions. No model
or deployment threshold was fitted, and no segmentation forward pass was run.
New retrospective diagnostic points do not replace frozen baselines or metrics.
