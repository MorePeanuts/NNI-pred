import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Literal
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, make_scorer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from nni_pred.data import MergedTabularDataset
from nni_pred.evaluation import Metrics, Evaluator
from nni_pred.transformers import get_feature_engineering, TargetTransformer
from nni_pred.models import RandomForestBuilder, XGBoostBuilder, ElasticNetBuilder


class Trainer:
    def __init__(
        self,
        dataset=None,
        scoring=None,
        outer_k_fold: int = 5,
        inner_k_fold: int = 4,
        output_path: Path | None = None,
        param_size: Literal['small', 'medium', 'large'] = 'small',
    ):
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        if output_path is None:
            self.output_path = Path(__file__).parents[2] / f'output/train_{now}'
        else:
            self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.param_size = param_size

        self.log_path = self.output_path / f'full_trace_{now}.log'
        logger.remove()
        logger.add(sys.stderr, level='INFO')
        logger.add(self.log_path, level='TRACE')

        if dataset is None:
            self.dataset = MergedTabularDataset()
        else:
            assert hasattr(dataset, 'prepare_data')
            self.dataset = dataset

        if scoring is None:
            self.scoring = make_scorer(Metrics.calc_kge, greater_is_better=True)
            logger.info('Using default scoring function: KGE')
        else:
            self.scoring = scoring

        self.outer_k_fold = outer_k_fold
        self.inner_k_fold = inner_k_fold

        self.model_builder = {
            'linear': ElasticNetBuilder(),
            'rf': RandomForestBuilder(),
            'xgb': XGBoostBuilder(),
        }

    def train(
        self,
        target: str,
        model_type: Literal['linear', 'rf', 'xgb', 'all'] = 'all',
        random_state: int = 42,
    ):
        if model_type == 'all':
            model_list = ['linear', 'rf', 'xgb']
        else:
            model_list = [model_type]

        X, y_dict, groups = self.dataset.prepare_data()
        y = y_dict[target]
        output_path = self.output_path / f'{target}'
        output_path.mkdir(parents=True, exist_ok=True)

        for model_tp in model_list:
            self.run_nested_cv(target, model_tp, X, y, groups, random_state, output_path=target)
            self.run_final_train(target, model_tp, X, y, groups, random_state, output_path=target)

        logger.info('Done.')

    def run_nested_cv(self, target, model_type, X, y, groups, random_state=42, output_path=None):
        if output_path is None:
            output_path = self.output_path / f'{target}/seed_{random_state}'
        else:
            output_path = self.output_path / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        model_builder = self.model_builder[model_type]
        model_name = model_builder.model_name
        param_grid = model_builder.get_default_param_grid(self.param_size)  # type: ignore
        param_grid = {f'model__regressor__{k}': v for k, v in param_grid.items()}

        logger.info(f'(Seed={random_state}) Training {model_name} for {target}...')
        outer_cv = GroupKFold(self.outer_k_fold, shuffle=True, random_state=random_state)
        evaluator = Evaluator(target, model_name, model_type, self.outer_k_fold)  # type: ignore

        for idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            train_val_X = X.iloc[train_val_idx]
            train_val_y = y.iloc[train_val_idx]
            test_X = X.iloc[test_idx]
            test_y = y.iloc[test_idx]
            train_groups = groups[train_val_idx]

            feature_engineering = get_feature_engineering(model_type, random_state)  # type: ignore
            regressor = TransformedTargetRegressor(
                model_builder.get_regressor(),  # type: ignore
                transformer=TargetTransformer(1),
            )
            pipeline = Pipeline(
                [
                    ('prep', feature_engineering),
                    ('model', regressor),
                ]
            )

            inner_cv = GroupKFold(self.inner_k_fold, shuffle=True, random_state=random_state)
            grid_search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=inner_cv)

            # Inner cross-validation
            grid_search.fit(train_val_X, train_val_y, groups=train_groups)
            test_y_pred = grid_search.predict(test_X)
            offset = grid_search.best_estimator_.named_steps['model'].transformer_.offset_
            best_param = {
                k.replace('model__regressor__', ''): v for k, v in grid_search.best_params_.items()
            }

            # Inner evaluation
            evaluator.update(
                X=test_X,
                y_true=test_y,
                y_pred=test_y_pred,
                fold=idx + 1,
                offset=offset,
                best_param=best_param,
                best_inner_score=grid_search.best_score_,
            )

        logger.info('Finished.')
        evaluator.save_result(output_path)

    def run_final_train(self, target, model_type, X, y, groups, random_state=42, output_path=None):
        if output_path is None:
            output_path = self.output_path / f'{target}/seed_{random_state}'
        else:
            output_path = self.output_path / output_path

        model_builder = self.model_builder[model_type]
        model_name = model_builder.model_name
        param_grid = model_builder.get_default_param_grid(self.param_size)  # type: ignore
        param_grid = {f'model__regressor__{k}': v for k, v in param_grid.items()}

        logger.info(f'(Seed={random_state}) Training {model_name} for {target} on all data...')
        feature_engineering = get_feature_engineering(model_type, random_state)  # type: ignore
        regressor = TransformedTargetRegressor(
            model_builder.get_regressor(),  # type: ignore
            transformer=TargetTransformer(1),
        )
        pipeline = Pipeline(
            [
                ('prep', feature_engineering),
                ('model', regressor),
            ]
        )
        final_cv = GroupKFold(self.outer_k_fold, shuffle=True, random_state=random_state)
        final_grid_search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=final_cv)

        final_grid_search.fit(X, y, groups=groups)
        best_model = final_grid_search.best_estimator_
        model_path = output_path / f'{model_name.replace(" ", "_")}_for_{target}.joblib'  # type: ignore
        joblib.dump(best_model, model_path)
        logger.info(f'Model has been saved to {model_path}')


class SeedSelector:
    # TODO: 种子选择器
    pass
