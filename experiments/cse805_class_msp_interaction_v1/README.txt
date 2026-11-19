CSE 805 | Class-by-MSP interaction experiment

Read results/report_zh.txt for the findings. The model, fold assignments, training
records, full-population histograms and tables are separate from the original
cse805_error_prediction_v1 experiment.

Comparison
  The frozen backbone produces MSP, normalized entropy, the top-two margin and
  a predicted Cityscapes class (19 possibilities). Both LR models use the same
  three continuous features and 19 class indicators. The interaction model adds
  one term per class: its indicator times the training-standardized MSP. Each
  pixel activates just one interaction. At fixed entropy and margin the error
  log-odds slope on standardized MSP is the global coefficient plus the active
  class's interaction coefficient. The combined slope is the interpretable
  quantity under the full one-hot and L2 parameterization.

  Data and splits follow the original experiment. Each of 17 conservative
  filename groups is held out once. A scaler fitted on the corresponding
  training pixels and inverse inclusion weights (normalized for model fitting)
  reproduce the existing baseline design. Only inner-fold weighted AP chooses
  C. The class feature is a predicted class at inference. GT defines the error
  label and excludes ignored pixels in evaluation. See protocol.json for the
  comparisons chosen before fitting and run_manifest.json for input hashes.

Reading the results
  results/group_metrics.csv            Full-pixel metrics and train-threshold results, 3 methods × 17 groups.
  results/paired_comparisons.csv       Overall, by weather and by group. MSP, old=LR+Class and new=interaction.
  results/night_low_coverage.csv       1%, 5%, 10%, 20% for each Night group and method; definitions below.
  results/night_budget_summary.csv     Equal-group Night risks plus total expected/observed counts across groups.
  results/night_fog.png                Matched 10% Night risk and per-group Fog risk80 differences vs MSP.
  results/verification.json            Independent count checks and input/output fingerprints.
  folds/<group>/selection.json         Training/development memberships, all candidates and chosen C.
  folds/<group>/interaction.joblib     Fitted preprocessing and classifier. Load using the run environment.
  folds/<group>/evaluation.json        Per-group primary metrics, QA checks, provenance and runtime.
  folds/<group>/histograms.npz         Full-pixel counts in the original score bins and Night vegetation counts.

  All CSV risk, coverage and prevalence columns use proportions unless named
  "_pp" (percentage points). Lower error and AURC, higher AUROC and AP are better.
  Aggregate differences weight each held-out group equally. Primary risk80
  uses the nearest attainable whole-bin coverage, as in the original study.
  The training-derived threshold is computed from outer-training scores alone.

  At low coverage whole-bin ties can make accepted counts very different.
  "accepted_errors" is the observed count for "accepted_pixels" at actual_coverage.
  The matched diagnostic uses K = floor(target * N_valid + 0.5). Accept every
  lower bin, then uniformly choose the remaining pixels in the boundary bin.
  "matched_expected_errors" is the expectation for this rule, and "matched_risk"
  divides it by K. A model may have no single cutoff that achieves K. Vegetation
  counts describe the actual whole-bin set only. "qa_raw_secondary_risk" instead
  ranks the shared 8192-pixel/image sample by unbinned scores, then by saved
  label-independent tie keys. These two diagnostics preserve the original
  whole-bin primary endpoint.

Reproduction from the project root
  bash experiments/cse805_class_msp_interaction_v1/run.sh preflight
  bash experiments/cse805_class_msp_interaction_v1/run.sh train
  bash experiments/cse805_class_msp_interaction_v1/run.sh evaluate
  bash experiments/cse805_class_msp_interaction_v1/run.sh report
  bash experiments/cse805_class_msp_interaction_v1/run.sh verify

  Use the existing project .venv and original isolated packages, cached logits,
  local GT, and original CUDA backend. Completed model/evaluation records with
  matching fingerprints are reused. train and evaluate also accept explicit
  group IDs; train jobs on disjoint groups may run in separate processes.
  Data regeneration and model selection are not part of report or verify.
  A fresh fit with different inputs or code requires a separate output version.

Interpretation
  This is an exploratory follow-up to previously observed Night failures. The
  three Night filename groups provide descriptive evidence about transfer.
  Pixels within scenes are correlated, and retrospective coverage does not
  establish a deployment operating point.
