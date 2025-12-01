"""
Preprocessing module for NNI prediction.

This module provides CV-compatible transformers for use in cross-validation pipelines.
"""

from .transformers import (
    CVCompatibleSkewnessTransformer,
    CVCompatibleGroupedPCA,
    CVCompatiblePreprocessingPipeline,
)
from .feature_groups import get_feature_groups, validate_feature_groups

__all__ = [
    'CVCompatibleSkewnessTransformer',
    'CVCompatibleGroupedPCA',
    'CVCompatiblePreprocessingPipeline',
    'get_feature_groups',
    'validate_feature_groups',
]
