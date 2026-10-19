# Repository execution conventions

- Server commands run from the repository root; do not prepend a `cd` command.
- On the server use `.venv/bin/python` for scripts, modules, training and tests.

# Pruning experiment requirements

- New pruning experiments keep label_smoothing=0 in both search and recovery,
  per the team's decision. Do not tune it without a new explicit instruction.
- Existing ResNet50/CIFAR10 sweeps keep min_keep_ratio=0.08. The follow-up
  keep001_target4p5m/keep001_target14m and keep001_quarterpp_grid8 configurations
  use min_keep_ratio=0.001
  by the user's explicit 2026-10-19 request to restore the old red-circle floor.
  This is the minimum surviving channel fraction, not the gate probability
  threshold (which stays 0.5). For the 60-epoch search, both feasible and fallback
  selection use only epochs 31-60. New follow-ups score each physical candidate
  after 200 train-only BN batches; recovery must preserve this selection policy.
- Adaptive lambda is a core requirement. Do not disable it or replace it with
  fixed-lambda experiments without explicit user approval. The historical fixed
  pilot remains only for reproduction, not the default next experiment.
- A new model may consume at most 150 total training epochs, counting all search,
  intermediate recovery and final fine-tuning. Never add training to a completed
  150-epoch result; allocate phases within the same budget on a new model.
- Physical recovery has no gates or gate penalty. Hold the adaptive controller
  state and restore it for the next search; do not reset it silently.
- Primary competitor: DepGraph (also called "depthgraph" by the user).
- Final comparisons and plots use official test metrics, never validation as a
  substitute. Select checkpoints and tune settings on validation, then evaluate
  frozen deployments on test without training or BatchNorm updates.
- Iteration informed by previously seen test results is exploratory, not a fresh
  untouched holdout confirmation. Keep physical parameters distinct from active
  parameter estimates and account for training/pretraining budgets explicitly.
- In comparisons with different adaptive-search horizons, do not reuse one fixed
  `log_step` merely to keep the numeric configuration identical. Derive the
  controller warmup and automatic step from each search horizon, and verify the
  resolved values before launching, unless the user explicitly requests a
  fixed-step ablation.
