"""
This script defines several candidate model constructors and a hyperparameter grid.

Candidate models include linear models, random forests, and XGBoost.
"""

import pickle
import pandas as pd
import numpy as np
from typing import Literal
from xgboost import XGBRegressor
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from dataclasses import dataclass
from .transformers import get_preprocessing_pipeline, TargetTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class ElasticNetBuilder:
    model_name = 'Elastic Net'
    model_type = 'linear'

    def __init__(self):
        super().__init__()

    def get_default_param_grid(self, scale: Literal['small', 'medium', 'large']):
        match scale:
            case 'small':
                return {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.3, 0.5, 0.7],
                    'max_iter': [10000],
                }
            case 'medium':
                return {
                    'alpha': [0.0005, 0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
                    'max_iter': [10000],
                }
            case 'large':
                return {
                    'alpha': [0.0005, 0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
                    'max_iter': [10000],
                }

    def get_regressor(self, random_state: int = 42):
        feature_engineering = get_preprocessing_pipeline(self.model_type, random_state=random_state)
        regressor = TransformedTargetRegressor(
            ElasticNet(random_state=random_state),
            transformer=TargetTransformer(0),
        )
        pipeline = Pipeline([('prep', feature_engineering), ('model', regressor)])
        return pipeline


class RandomForestBuilder:
    model_name = 'Random Forest'
    model_type = 'rf'

    def __init__(
        self,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.n_jobs = n_jobs

    def get_default_param_grid(self, scale: Literal['small', 'medium', 'large']):
        match scale:
            case 'small':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [8, 15],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4],
                    'max_features': ['sqrt', 'log2'],
                }
            case 'medium':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [8, 15, 20],
                    'min_samples_split': [5, 8, 10],
                    'min_samples_leaf': [2, 4],
                    'max_features': ['sqrt', 'log2'],
                }
            case 'large':
                return {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [8, 15, 20],
                    'min_samples_split': [5, 8, 10],
                    'min_samples_leaf': [2, 4, 6],
                    'max_features': ['sqrt', 'log2'],
                }

    def get_regressor(self, random_state: int = 42):
        feature_engineering = get_preprocessing_pipeline(self.model_type, random_state=random_state)
        regressor = TransformedTargetRegressor(
            RandomForestRegressor(random_state=random_state, n_jobs=self.n_jobs),
            transformer=TargetTransformer(0),
        )
        pipeline = Pipeline([('prep', feature_engineering), ('model', regressor)])
        return pipeline


class XGBoostBuilder:
    model_name = 'XGBoost'
    model_type = 'xgb'

    def __init__(
        self,
        objective='reg:tweedie',
        tree_method='hist',
        **kwargs,
    ):
        self.objective = objective
        self.tree_method = tree_method

    def get_default_param_grid(self, scale: Literal['small', 'medium', 'large']):
        match scale:
            case 'small':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'reg_alpha': [0],
                    'reg_lambda': [1],
                    'tweedie_variance_power': [1.2],
                }
            case 'medium':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'reg_alpha': [0, 0.1],
                    'reg_lambda': [1, 5],
                    'tweedie_variance_power': [1.2, 1.8],
                }
            case 'large':
                return {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 5, 10],
                    'tweedie_variance_power': [1.2, 1.5, 1.8],
                }

    def get_regressor(self, random_state: int = 42):
        feature_engineering = get_preprocessing_pipeline(self.model_type, random_state=random_state)
        regressor = TransformedTargetRegressor(
            XGBRegressor(
                # objective=self.objective,
                tree_method=self.tree_method,
                random_state=random_state,
            ),
            transformer=TargetTransformer(0),
        )
        pipeline = Pipeline([('prep', feature_engineering), ('model', regressor)])
        return pipeline
