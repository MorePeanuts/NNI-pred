# NNI-pred

Predicting neonicotinoid insecticide (NNI) concentrations in soil and water using machine learning.

## Background

Neonicotinoids are widely used insecticides that can persist in the environment. This project builds predictive models to estimate NNI concentrations based on environmental, agricultural, and socio-economic factors.

## Features

- Supports soil-only and water-soil merged datasets
- Three model types: Elastic Net, Random Forest, XGBoost
- Nested spatial cross-validation to prevent data leakage
- SHAP-based feature importance analysis

## Usage

```bash
uv run scripts/train_all.py --cls merged --targets THIA IMI CLO
```

