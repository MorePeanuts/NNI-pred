"""
Model wrappers for NNI prediction.

This module provides unified interfaces for different machine learning models.
"""

from .base import BaseModel
from .elastic_net import ElasticNetModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel

__all__ = [
    'BaseModel',
    'ElasticNetModel',
    'RandomForestModel',
    'XGBoostModel',
]
