# Repository execution conventions

- Server commands run from the repository root; do not prepend a `cd` command.
- On the server use `.venv/bin/python` for scripts, modules, training and tests.

# Pruning experiment requirements

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
