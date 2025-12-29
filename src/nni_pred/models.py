import pickle
import pandas as pd
import numpy as np
from typing import Literal
from xgboost import XGBRegressor
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from dataclasses import dataclass


@dataclass
class RandomForestConfig:
    # TODO: 使用自定义超参数训练模型
    pass


class ElasticNetBuilder:
    model_name = 'Elastic Net'
    model_type = 'linear'

    def __init__(
        self,
        random_state: int = 42,
        **kwargs,
    ):
        self.random_state = random_state

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

    def get_regressor(self):
        return ElasticNet(random_state=self.random_state)


class RandomForestBuilder:
    model_name = 'Random Forest'
    model_type = 'rf'

    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.random_state = random_state
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

    def get_regressor(self):
        return RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


class XGBoostBuilder:
    model_name = 'XGBoost'
    model_type = 'xgb'

    def __init__(
        self,
        objective='reg:tweedie',
        tree_method='hist',
        random_state: int = 42,
        **kwargs,
    ):
        self.objective = objective
        self.tree_method = tree_method
        self.random_state = random_state

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

    def get_regressor(self):
        return XGBRegressor(
            objective=self.objective,
            tree_method=self.tree_method,
            random_state=self.random_state,
        )
