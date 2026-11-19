CSE 805 exploratory Random Forest comparison

The six-method results and interpretation are in results/final_report_zh.txt.
The original error-prediction study and class-by-MSP extension supply read-only
comparators. New forests, selection logs, histogram counts and summaries live here.

Design
  Every method uses the same 1,600 images, 17 outer filename groups and all
  valid pixels. The four weather conditions each contribute 400 images. RF
  uses the original three confidence features in their native units and 19
  fixed one-hot indicators of predicted semantic class. Weather does not
  enter the score. Frozen protocol.json and run_manifest.json record the
  choices and input hashes before RF training.

  Within each outer training partition, the original three inner folds choose
  between depths 8 and 14 by mean weighted AP. Each candidate has 32 trees,
  Gini splits, sqrt feature subsampling, minimum leaf size 64 and seed 805.
  All 512 seed-805 sample pixels per training image are available to each
  forest. Each tree draws about half its partition size with replacement;
  scikit-learn 1.9.1 implements the provided inverse-inclusion weights through
  bootstrap draw probabilities. The outer fit computes its weighted training
  score quantile; exact attained held-out coverage and error are recorded.

  Full evaluation streams frozen native logits on CUDA and uses the original
  feature extractor. gpu_forest.py traverses the fitted sklearn trees in their
  stored order, comparing float32 inputs with the stored float64 thresholds
  and averaging float64 leaf error fractions. Every image's saved QA indices
  check the actual full-stream features, binary labels and scores against
  sklearn. The original 406 validation images stay separate.

Outputs
  results/final_comparison.csv       Six methods, averaged equally across held-out groups overall or within each weather.
  results/group_metrics.csv          All 102 group/method rows with thresholds, counts, attained coverage and ranking metrics.
  results/rf_paired_differences.csv  RF minus each of five baselines, group wins/losses and leave-one-group-out sensitivity.
  results/night_low_coverage.csv     1%, 5%, 10%, 20%; actual whole-bin sets, equal-budget expectation, and independent tie-key QA.
  results/night_summary.csv          Both group-equal risks and sums of expected/observed errors across the three Night budgets.
  results/night_qa10_classes.csv     Weighted shared QA sample; exact raw-score tie mass, road acceptance and vegetation errors at 10%.
  results/risk_coverage.csv          1%–100% grid for each method and group, both whole-bin and matched-budget risk.
  results/qa_aurc_sensitivity.csv    AURC on the same QA sample under five original binning and tie policies, all six methods and 17 groups.
  results/training_and_runtime.csv   Fold-level validation APs, chosen depth, fitted-model size and measured times.
  results/final_comparison.png       Weather contrasts at the original endpoint plus all six early Night risk–coverage curves.
  results/verification.json          Split, numerical, score-parity, comparator and preservation checks and output hashes.
  folds/<group>/                    Fitted forest, complete inner selection, full histograms and evaluation record.

  Risk, prevalence and coverage in the CSVs are proportions unless a column
  ends in "_pp" (percentage points). Smaller risk/AURC and larger AUROC/AP
  are favorable. The primary retrospective endpoint takes the nearest occupied
  whole-bin cumulative coverage with the original 100,000 arcsine-sqrt bins.
  Fixed-budget sensitivity uses K=floor(target*N_valid+0.5), fills the final
  bin uniformly in expectation and reports the expected error count separately
  from the observed whole-bin count. The check on unbinned QA predictions uses
  the stored seed-806, label-independent tie keys. group-equal and pixel-weighted
  summaries answer different questions. All comparisons use error=1.

Reproduction (from the project root, using the existing environment)
  bash experiments/cse805_random_forest_v1/run.sh preflight
  bash experiments/cse805_random_forest_v1/run.sh train
  bash experiments/cse805_random_forest_v1/run.sh evaluate
  bash experiments/cse805_random_forest_v1/run.sh report
  bash experiments/cse805_random_forest_v1/run.sh verify

  Matched completed training/evaluation folds are reused. Each phase places new
  artifacts only in this extension; a different RF design needs a new version.
  The CUDA evaluation requires the project's GPU, original .venv and bundled
  Triton. Fit and evaluation times exclude segmentation training/forward passes;
  the RF scoring timer includes transfers and any first-use compilation.
  The Chinese interpretation is a written reading of the generated tables.

Interpretation is exploratory: the previous Night results informed this follow-up.
Groups are conservative filename proxies, with only three in Night. Retrospective
coverage and boundary-bin expectations do not guarantee deployment coverage.
