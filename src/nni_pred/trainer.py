"""
This script includes a trainer and a seed selector. The former defines the basic training loop (including
 the invocation of data processing functions), while the latter defines the method for selecting the best
 results from experiments with multiple random seeds.
"""

import json
import random
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Literal
from datetime import datetime
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import GroupKFold, GridSearchCV
from nni_pred.data import (
    MergedTabularDataset,
    MergedVariableGroups,
    SoilVariableGroups,
    SoilTabularDataset,
)
from nni_pred.evaluation import Metrics, Evaluator, Comparator, OOFMetrics
from nni_pred.models import RandomForestBuilder, XGBoostBuilder, ElasticNetBuilder


class Trainer:
    def __init__(
        self,
        var_cls: type[MergedVariableGroups | SoilVariableGroups],
        dataset=None,
        scoring=None,
        outer_k_fold: int = 5,
        inner_k_fold: int = 4,
        output_path: Path | None = None,
        n_jobs: int = -1,
        param_size: Literal['small', 'medium', 'large'] = 'small',
    ):
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        if output_path is None:
            self.output_path = Path(__file__).parents[2] / f'output/train_{now}'
        else:
            self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.param_size = param_size
        self.n_jobs = n_jobs

        self.log_path = self.output_path / f'full_trace_{now}.log'
        logger.add(self.log_path, level='TRACE')

        if dataset is None:
            if var_cls is MergedVariableGroups:
                self.dataset = MergedTabularDataset()
            elif var_cls is SoilVariableGroups:
                self.dataset = SoilTabularDataset()
        else:
            assert hasattr(dataset, 'prepare_data')
            self.dataset = dataset

        if scoring is None:
            self.scoring = 'r2'
        else:
            self.scoring = scoring

        self.outer_k_fold = outer_k_fold
        self.inner_k_fold = inner_k_fold

        self.model_builder = {
            'linear': ElasticNetBuilder(var_cls=var_cls),
            'rf': RandomForestBuilder(var_cls=var_cls),
            'xgb': XGBoostBuilder(var_cls=var_cls),
        }

    def train(
        self,
        target: str,
        model_type: Literal['linear', 'rf', 'xgb', 'all'] = 'all',
        random_state: int = 42,
        run_nested_cv: bool = True,
        run_final_train: bool = True,
    ):
        if model_type == 'all':
            model_list = ['linear', 'rf', 'xgb']
        else:
            model_list = [model_type]

        X, y_dict, groups = self.dataset.prepare_data()
        y = y_dict[target]

        for model_tp in model_list:
            if run_nested_cv:
                output_path = self.run_nested_cv(target, model_tp, X, y, groups, random_state)
            else:
                output_path = None
            if run_final_train:
                self.run_final_train(target, model_tp, X, y, groups, random_state)

        if output_path:
            logger.info(
                f'Finish training {model_list} on {target} with random_state {random_state}.'
            )
        return output_path

    def run_nested_cv(self, target, model_type, X, y, groups, random_state=42, output_path=None):
        if output_path is None:
            output_path = self.output_path / f'{target}/seed_{random_state}'
        else:
            output_path = self.output_path / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        model_builder = self.model_builder[model_type]
        model_name = model_builder.model_name
        param_grid = model_builder.get_default_param_grid(self.param_size)
        param_grid = {f'model__regressor__{k}': v for k, v in param_grid.items()}

        logger.info(f'(Seed={random_state}) Training {model_name} for {target}...')
        outer_cv = GroupKFold(self.outer_k_fold, shuffle=True, random_state=random_state)
        evaluator = Evaluator(target, model_name, model_type, self.outer_k_fold)

        for idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            train_val_X = X.iloc[train_val_idx]
            train_val_y = y.iloc[train_val_idx]
            test_X = X.iloc[test_idx]
            test_y = y.iloc[test_idx]
            train_groups = groups[train_val_idx]

            pipeline = model_builder.get_regressor(random_state)

            inner_cv = GroupKFold(self.inner_k_fold, shuffle=True, random_state=random_state)
            grid_search = GridSearchCV(
                pipeline, param_grid, scoring=self.scoring, cv=inner_cv, n_jobs=self.n_jobs
            )

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

        logger.info(
            f'Finish run nested cv of {model_type} on {target} with random_state {random_state}'
        )
        evaluator.save_result(output_path)

        return output_path

    def run_final_train(self, target, model_type, X, y, groups, random_state=42, output_path=None):
        if output_path is None:
            output_path = self.output_path / f'{target}/seed_{random_state}'
        else:
            output_path = self.output_path / output_path

        model_builder = self.model_builder[model_type]
        model_name = model_builder.model_name
        param_grid = model_builder.get_default_param_grid(self.param_size)
        param_grid = {f'model__regressor__{k}': v for k, v in param_grid.items()}

        logger.info(f'(Seed={random_state}) Training {model_name} for {target} on all data...')
        pipeline = model_builder.get_regressor(random_state)

        final_cv = GroupKFold(self.outer_k_fold, shuffle=True, random_state=random_state)
        final_grid_search = GridSearchCV(
            pipeline, param_grid, scoring=self.scoring, cv=final_cv, n_jobs=self.n_jobs
        )

        final_grid_search.fit(X, y, groups=groups)
        best_model = final_grid_search.best_estimator_
        model_path = output_path / f'{model_type}_model_for_{target}.joblib'
        joblib.dump(best_model, model_path)
        logger.info(f'Model has been saved to {model_path}')

        return model_path


class SeedSelector:
    def __init__(self, trainer: Trainer, comparator: Comparator, max_attempts=10, seed=42):
        self.trainer = trainer
        self.exp_root = self.trainer.output_path
        self.max_attempts = max_attempts
        self.seed = seed
        self.comparator = comparator
        self.rng = random.Random(self.seed)
        self.seed_set = set()

        while len(self.seed_set) < max_attempts:
            self.seed_set.add(self.rng.randint(1, 128000))

    def run_exp(self, targets: list[str]):
        total_targets = len(targets)
        pbar = tqdm(total=total_targets * len(self.seed_set))
        logger.add(
            lambda msg: pbar.write(msg.strip()),
            colorize=True,
        )
        logger.info(f'All targets: {targets}')
        for idx, target in enumerate(targets):
            pbar.set_description(f'Run experiment on {target}, progress: {idx + 1}/{total_targets}')
            for seed in self.seed_set:
                output_path = self.trainer.train(target, random_state=seed, run_final_train=False)
                pbar.update(1)
                self.comparator.compare_model(output_path)
            target_path = output_path.parent
            self.comparator.compare_seed(target_path)

            # Log
            logger.info(f'{target} model training and seed selection are finished.')
            seed_comparison_path = target_path / 'seed_comparison.json'
            if not seed_comparison_path.exists():
                continue
            with seed_comparison_path.open() as f:
                info = json.load(f)
                best_seed = info['best_seed']
                best_model_type = info['best_model_type']
                best_metrics = OOFMetrics.from_json(info['best_metrics'])
            logger.info(
                f'Best seed: {best_seed}\tBest model type: {best_model_type} {best_metrics}'
            )

            # Final train
            self.trainer.train(
                target, model_type=best_model_type, random_state=best_seed, run_nested_cv=False
            )

        # experiment summary
        rows = []
        for target in targets:
            target_path = self.exp_root / target / 'seed_comparison.json'
            if not target_path.exists():
                logger.warning(f'No best seed found in {target_path.parent}.')
                continue
            with target_path.open() as f:
                s = json.load(f)
                best_metrics = OOFMetrics.from_json(s['best_metrics'])
                row = {'target': target, 'seed': s['best_seed'], 'model': s['best_model_type']}
                row.update(best_metrics.to_format_dict())
                rows.append(row)

        if len(rows) > 0:
            metrics_summary = pd.DataFrame(rows)
            metrics_summary = metrics_summary.set_index('target')
            logger.info(
                f'Summary all {total_targets} targets.\n{Comparator.format_summary_table(metrics_summary)}'
            )
            summary_path = self.exp_root / 'metrics_summary.csv'
            metrics_summary.to_csv(summary_path, index=True)
        else:
            logger.warning(f'All targets failed to find best seed in {self.exp_root}')

        pbar.close()
