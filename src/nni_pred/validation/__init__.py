"""
Validation module for NNI prediction.

This module provides spatial cross-validation utilities.
"""

from .spatial_cv import SpatialGroupGenerator
from .nested_cv import NestedSpatialCV

__all__ = [
    'SpatialGroupGenerator',
    'NestedSpatialCV',
]
