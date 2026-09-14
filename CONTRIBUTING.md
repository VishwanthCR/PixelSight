# Contributing to PixelSight

## Branches

Use descriptive feature branches:

- `feature/plan1-v2`
- `feature/dataset-pipeline`
- `feature/plan2-sentinel2`
- `feature/downstream-evaluation`
- `feature/visualization-docs`

## Pull Requests

1. Create a feature branch from `develop`.
2. Make focused commits.
3. Run relevant tests before opening a PR.
4. Open a pull request into `develop`.
5. Request at least one teammate review.
6. Merge only after review and checks pass.

`main` is reserved for stable, reviewed work.

## Research Rules

- Keep Plan 1 and Plan 2 experiments clearly separated.
- Never use the Region 3 final test set to tune model or hyperparameters.
- Record experiment configuration and evaluation metrics.
- Do not commit raw datasets or model checkpoints.
- Preserve reproducibility: document dependencies, seeds and preprocessing choices when applicable.
