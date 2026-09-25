# Task 2.3 — Hyperparameter Experimentation

## Overview
Runs controlled experiments on the feed-forward network from task 2.1, varying
optimizer, learning rate, batch size, and epoch count, then compares the outcomes
in a table.

## Experiments run
| Optimizer | LR | Batch Size | Epochs |
|---|---|---|---|
| SGD | 0.01 | 16 | 10 |
| SGD | 0.1 | 16 | 10 |
| Adam | 0.001 | 16 | 10 |
| Adam | 0.001 | 64 | 10 |
| Adam | 0.001 | 16 | 20 |

## What the notebook does
1. Defines a reusable `run_experiment()` function that trains a fresh model with
   given hyperparameters and returns test accuracy + training time
2. Runs all 5 experiments listed above
3. Builds a comparison table and saves it as `experiment_log.csv`
4. Summarizes observations about how each hyperparameter affected results

## Files
- `hyperparameter_experimentation.ipynb` — the full notebook, code + outputs
- `experiment_log.csv` — generated automatically when the notebook runs

## How to run
```bash
pip install torch scikit-learn pandas
jupyter notebook hyperparameter_experimentation.ipynb
```
Then Run All Cells.

## Deliverable
Experiment log with comparison table, as required by the Skill Set Go EduTech
AI/ML track, Week 2, Task 2.3.
