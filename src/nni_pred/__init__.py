"""
NNI-pred: Neonicotinoid Insecticide Prediction with Machine Learning

This package provides tools for predicting neonicotinoid pollutant concentrations
in water bodies using spatial machine learning with nested cross-validation.

Main modules:
- preprocessing: CV-compatible transformers and feature engineering
- models: Model wrappers (Elastic Net, Random Forest, XGBoost)
- validation: Spatial cross-validation utilities
- training: Batch training and final model training
"""

__version__ = "0.1.0"

# Import main classes for convenient access
from .preprocessing import (
    CVCompatiblePreprocessingPipeline,
    get_feature_groups,
)
from .models import (
    ElasticNetModel,
    RandomForestModel,
    XGBoostModel,
)
from .validation import (
    SpatialGroupGenerator,
    NestedSpatialCV,
)
from .training import (
    BatchTrainer,
    FinalModelTrainer,
)

__all__ = [
    'CVCompatiblePreprocessingPipeline',
    'get_feature_groups',
    'ElasticNetModel',
    'RandomForestModel',
    'XGBoostModel',
    'SpatialGroupGenerator',
    'NestedSpatialCV',
    'BatchTrainer',
    'FinalModelTrainer',
]


def hello() -> str:
    """Legacy hello function for compatibility."""
    return "Hello from nni-pred!"
