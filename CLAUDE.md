# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NNI-pred is a machine learning project for predicting neonicotinoid insecticide (NNI) concentrations in environmental samples (soil and water). It uses nested spatial cross-validation with multiple model types (Elastic Net, Random Forest, XGBoost) and automated seed selection.

## Commands

### Running Experiments

```bash
# Full training with seed selection (recommended)
uv run scripts/train_all.py --max-attempts 5 --size small

# Train specific targets only
uv run scripts/train_all.py --max-attempts 2 --targets THIA --size small

# Choose dataset type (merged water-soil or soil-only)
uv run scripts/train_all.py --cls merged  # or --cls soil

# Simple single-model training (for exploration/debugging)
uv run scripts/train_simplest.py xgb --size small --targets THIA
```

### Key Arguments
- `--size`: Hyperparameter grid size (`small`, `medium`, `large`)
- `--targets`: Pollutant targets (e.g., `THIA IMI CLO` or `all`)
- `--cls`: Dataset class (`merged` for water with soil features, `soil` for soil-only)
- `--max-attempts`: Number of random seeds to try
- `--indicator`: Metric for seed selection (default: `NSE_log`)

## Architecture

### Core Package (`src/nni_pred/`)

- **data.py**: Dataset classes (`MergedTabularDataset`, `SoilTabularDataset`) and variable group definitions (`MergedVariableGroups`, `SoilVariableGroups`). Variable groups define feature categories: metadata, categorical, targets (parent/metabolites), natural, agro, and socio-economic features.

- **models.py**: Model builders (`ElasticNetBuilder`, `RandomForestBuilder`, `XGBoostBuilder`) that create sklearn pipelines with preprocessing + regressor. Each builder provides hyperparameter grids at different scales.

- **trainer.py**: `Trainer` handles nested cross-validation with GridSearchCV. `SeedSelector` runs experiments across multiple random seeds and selects the best seed/model combination.

- **evaluation.py**: `Metrics` computes NSE, RSR, PBIAS, KGE and detection rates. `Evaluator` tracks fold-level results. `Comparator` selects best models across seeds based on coefficient of variation thresholds.

- **transformers.py**: Feature engineering pipelines. `GroupedPCA` applies separate PCA to agro and socio feature groups. `TargetTransformer` applies log transform with offset. `SkewnessTransformer` handles high-skew features for linear models.

- **visualization.py**: `Visualizer` generates metrics bar charts, measured vs predicted plots, and SHAP analysis plots.

### Data Flow

1. Dataset loads CSV, creates spatial groups from (Lon, Lat) for GroupKFold
2. Preprocessing pipeline: feature selection -> encoding -> optional skewness transform -> GroupedPCA
3. Target transformed with log(y + offset) where offset = min(positive_values) / 2
4. Nested CV: outer 5-fold (spatial) for evaluation, inner 4-fold for hyperparameter tuning
5. Results saved as JSON metrics files; best seed/model selected by `Comparator`

### Output Structure

```
output/exp_{cls}_{seed}_{timestamp}/
├── {target}/
│   ├── seed_{n}/
│   │   ├── oof_metrics_of_{model_type}.json
│   │   ├── model_comparison.json
│   │   └── {model_type}_model_for_{target}.joblib
│   └── seed_comparison.json
└── metrics_summary.csv
```

## Key Evaluation Metrics

- **NSE_log**: Nash-Sutcliffe Efficiency on log-transformed values (primary metric)
- **RSR**: RMSE-observations Standard Deviation Ratio
- **PBIAS**: Percent Bias
- **KGE**: Kling-Gupta Efficiency
- Detection-based variants (e.g., `NSE_detected`) computed only on samples above detection threshold
