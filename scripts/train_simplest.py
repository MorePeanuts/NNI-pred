"""
Perform a simplest nested spatial cross-validation using the random forest model.
"""

import sys
import joblib
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from nni_pred.models import RandomForestBuilder, XGBoostBuilder, ElasticNetBuilder
from nni_pred.data import MergedTabularDataset
from nni_pred.transformers import GroupedPCA, TargetTransformer, get_feature_engineering
from nni_pred.evaluation import Metrics, Evaluator
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, make_scorer
from loguru import logger


def main():
    match args.model_type:
        case 'linear':
            builder = ElasticNetBuilder()
        case 'rf':
            builder = RandomForestBuilder()
        case 'xgb':
            builder = XGBoostBuilder()

    model_name = builder.model_name
    model_type = builder.model_type
    dataset = MergedTabularDataset()
    X, y_dict, groups = dataset.prepare_data()
    param_grid = builder.get_default_param_grid(args.size)
    custom_scoring = make_scorer(Metrics.calc_kge, greater_is_better=True)
    param_grid_pipeline = {f'model__regressor__{k}': v for k, v in param_grid.items()}

    for _, (target, y) in enumerate(y_dict.items()):
        if 'all' not in args.targets and target not in args.targets:
            continue
        target_output_path = output_path / f'{target}'
        target_output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Nested CV
        logger.info(f'Training {model_name} Predictor for {target}...')
        outer_cv = GroupKFold(5, shuffle=True, random_state=args.seed)
        evaluator = Evaluator(target, model_name=model_name, model_type=model_type, k_fold=5)

        for idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            train_val_X = X.iloc[train_val_idx]
            train_val_y = y.iloc[train_val_idx]
            test_X = X.iloc[test_idx]
            test_y = y.iloc[test_idx]
            train_groups = groups[train_val_idx]

            feature_engineering = get_feature_engineering(args.model_type, random_state=args.seed)
            model = TransformedTargetRegressor(
                builder.get_regressor(), transformer=TargetTransformer(1)
            )
            pipeline = Pipeline(
                [
                    ('prep', feature_engineering),
                    ('model', model),
                ]
            )

            inner_cv = GroupKFold(4, shuffle=True, random_state=args.seed)
            grid_search_cv = GridSearchCV(
                pipeline, param_grid_pipeline, scoring=custom_scoring, cv=inner_cv
            )

            grid_search_cv.fit(train_val_X, train_val_y, groups=train_groups)
            test_y_pred = grid_search_cv.predict(test_X)
            offset = grid_search_cv.best_estimator_.named_steps['model'].transformer_.offset_
            best_param = {
                k.replace('model__regressor__', ''): v
                for k, v in grid_search_cv.best_params_.items()
            }

            evaluator.update(
                X=test_X,
                y_true=test_y,
                y_pred=test_y_pred,
                fold=idx + 1,
                offset=offset,
                best_param=best_param,
                best_inner_score=grid_search_cv.best_score_,
            )

        logger.info('Finished!')
        evaluator.save_result(target_output_path)

        # Step 2: Final model training
        logger.info(f'Training {model_name} ({target}) on all data...')
        feature_engineering = get_feature_engineering(args.model_type, random_state=args.seed)
        model = TransformedTargetRegressor(
            builder.get_regressor(), transformer=TargetTransformer(1)
        )
        pipeline = Pipeline(
            [
                ('prep', feature_engineering),
                ('model', model),
            ]
        )
        final_cv = GroupKFold(5, shuffle=True, random_state=args.seed)
        final_grid_search = GridSearchCV(
            pipeline, param_grid_pipeline, scoring=custom_scoring, cv=final_cv
        )
        final_grid_search.fit(X, y, groups=groups)
        best_model = final_grid_search.best_estimator_
        model_path = target_output_path / f'{model_name.replace(" ", "_")}_for_{target}.joblib'
        joblib.dump(best_model, model_path)
        logger.info(f'Model has been saved to {model_path}')

    logger.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=['linear', 'rf', 'xgb'])
    parser.add_argument(
        '--size',
        default='small',
        choices=['small', 'medium', 'large'],
        help='Hyperparameter grid size: small (fast), medium (moderate), full (exhaustive)',
    )
    parser.add_argument(
        '--seed', default=42, type=int, help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        default=['all'],
        help='Pollutants to train (space-separated). Use "all" for all 11 pollutants. '
        'Available: THIA IMI CLO ACE DIN parentNNIs IMI-UREA DN-IMI DM-ACE CLO-UREA mNNIs',
    )
    parser.add_argument('--output', type=str, help='Output path for models and evaluation results.')
    args = parser.parse_args()
    if args.output:
        output_path = Path(args.output)
    else:
        now = datetime.now().strftime('%m%d_%H%M%S')
        output_path = (
            Path(__file__).parents[1]
            / f'output/simplest_{args.model_type}_{args.seed}_{"_".join(args.targets)}_{now}'
        )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / f'trace_{now}.log'
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    logger.add(log_path, level='TRACE')

    main()
