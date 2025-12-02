"""
Nested Spatial Cross-Validation for NNI Prediction

This module implements nested cross-validation with spatial awareness.
The double-loop structure ensures unbiased performance estimation while
properly tuning hyperparameters.

Outer loop (5-fold): Generalization evaluation
Inner loop (4-fold): Hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..preprocessing import CVCompatiblePreprocessingPipeline
from ..models.base import BaseModel


class NestedSpatialCV:
    """
    Orchestrate nested spatial cross-validation.

    This class manages the double-loop cross-validation:
    - Outer loop: 5-fold GroupKFold for unbiased performance estimation
    - Inner loop: 4-fold GroupKFold for hyperparameter selection

    All preprocessing (skewness, scaling, PCA) fits only on training folds.
    """

    def __init__(
        self,
        n_outer: int = 5,
        n_inner: int = 4,
        random_state: int = 42,
        verbose: int = 1,
    ):
        """
        Initialize nested CV orchestrator.

        Args:
            n_outer: Number of outer CV folds (default: 5)
            n_inner: Number of inner CV folds (default: 4)
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.random_state = random_state
        self.verbose = verbose

    def run_nested_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseModel,
        feature_groups: dict,
        groups: np.ndarray,
        param_grid: dict | None = None,
    ) -> dict:
        """
        Run nested CV for a single pollutant-model combination.

        Args:
            X: Feature dataframe
            y: Target series (single pollutant)
            model: Model instance (ElasticNetModel, RandomForestModel, etc.)
            feature_groups: Feature group definitions
            groups: Spatial group IDs
            param_grid: Optional custom parameter grid (if None, uses model's default)

        Returns:
            Dictionary with:
            - 'outer_fold_results': List of dicts (5 folds)
            - 'mean_metrics': Dict with mean R², RMSE, MAE
            - 'std_metrics': Dict with std R², RMSE, MAE
            - 'best_params_per_fold': List of best params from each fold
        """
        # Get parameter grid
        if param_grid is None:
            param_grid = model.get_param_grid()

        # Initialize outer CV
        outer_cv = GroupKFold(n_splits=self.n_outer)
        outer_results = []

        if self.verbose >= 1:
            print(f'\n{"=" * 60}')
            print(f'Running Nested CV: {model.get_model_name()}')
            print(f'{"=" * 60}')
            print(f'Outer folds: {self.n_outer}, Inner folds: {self.n_inner}')
            print(f'Hyperparameter combinations: {self._count_grid_combinations(param_grid)}')
            print(f'{"=" * 60}\n')

        # Outer loop: Generalization evaluation
        for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y, groups)):
            if self.verbose >= 1:
                print(f'Outer Fold {fold_idx + 1}/{self.n_outer}')
                print(f'{"-" * 60}')

            # Split data
            X_outer_train = X.iloc[outer_train_idx]
            X_outer_test = X.iloc[outer_test_idx]
            y_outer_train = y.iloc[outer_train_idx]
            y_outer_test = y.iloc[outer_test_idx]
            groups_outer_train = groups[outer_train_idx]

            if self.verbose >= 2:
                print(f'  Training samples: {len(X_outer_train)}')
                print(f'  Test samples: {len(X_outer_test)}')
                print(f'  Training groups: {len(np.unique(groups_outer_train))}')
                print(f'  Test groups: {len(np.unique(groups[outer_test_idx]))}')

            # Create preprocessing pipeline
            preprocessor = CVCompatiblePreprocessingPipeline(
                model_type=model.get_model_type(),
                feature_groups=feature_groups,
            )

            # Create sklearn Pipeline
            pipeline = Pipeline(
                [('preprocessor', preprocessor), ('model', model.get_sklearn_model())]
            )

            # Prefix param names with 'model__' for pipeline
            pipeline_param_grid = {f'model__{k}': v for k, v in param_grid.items()}

            # Inner CV for hyperparameter tuning
            inner_cv = GroupKFold(n_splits=self.n_inner)

            # Grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid=pipeline_param_grid,
                cv=inner_cv,
                scoring='r2',
                n_jobs=-1,
                verbose=0 if self.verbose < 2 else 1,
            )

            try:
                # Fit with groups
                grid_search.fit(X_outer_train, y_outer_train, groups=groups_outer_train)

                # Evaluate on outer test set
                y_pred = grid_search.predict(X_outer_test)

                # Calculate metrics
                r2 = r2_score(y_outer_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_outer_test, y_pred))
                mae = mean_absolute_error(y_outer_test, y_pred)

                # Extract best params (remove 'model__' prefix)
                best_params = {
                    k.replace('model__', ''): v for k, v in grid_search.best_params_.items()
                }

                fold_result = {
                    'fold': fold_idx + 1,
                    'best_params': best_params,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'n_train': len(X_outer_train),
                    'n_test': len(X_outer_test),
                    'best_inner_score': grid_search.best_score_,  # Inner CV R²
                }

                if self.verbose >= 1:
                    print('  Results:')
                    print(f'    R²   = {r2:.4f}')
                    print(f'    RMSE = {rmse:.4f}')
                    print(f'    MAE  = {mae:.4f}')
                    if self.verbose >= 2:
                        print(f'    Best inner CV R² = {grid_search.best_score_:.4f}')
                        print(f'    Best params: {best_params}')
                    print()

                outer_results.append(fold_result)

            except Exception as e:
                print(f'\n  ERROR in Outer Fold {fold_idx + 1}: {str(e)}')
                print('  Skipping this fold...\n')
                continue

        # Check if we have any successful folds
        if len(outer_results) == 0:
            raise ValueError('All outer folds failed. Cannot proceed.')

        # Aggregate results
        metrics = ['r2', 'rmse', 'mae']
        mean_metrics = {m: np.mean([r[m] for r in outer_results]) for m in metrics}
        std_metrics = {m: np.std([r[m] for r in outer_results]) for m in metrics}

        if self.verbose >= 1:
            print(f'\n{"=" * 60}')
            print(f'Nested CV Complete: {model.get_model_name()}')
            print(f'{"=" * 60}')
            print(f'Mean R²   = {mean_metrics["r2"]:.4f} ± {std_metrics["r2"]:.4f}')
            print(f'Mean RMSE = {mean_metrics["rmse"]:.4f} ± {std_metrics["rmse"]:.4f}')
            print(f'Mean MAE  = {mean_metrics["mae"]:.4f} ± {std_metrics["mae"]:.4f}')
            print(f'{"=" * 60}\n')

        return {
            'outer_fold_results': outer_results,
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'best_params_per_fold': [r['best_params'] for r in outer_results],
            'n_successful_folds': len(outer_results),
        }

    def _count_grid_combinations(self, param_grid: dict) -> int:
        """
        Count total hyperparameter combinations.

        Args:
            param_grid: Dictionary of parameter lists

        Returns:
            Total number of combinations
        """
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count

    @staticmethod
    def compare_models(cv_results_list: list[dict], model_names: list[str]) -> pd.DataFrame:
        """
        Compare multiple models based on CV results.

        Args:
            cv_results_list: List of CV result dictionaries
            model_names: List of model names

        Returns:
            DataFrame with comparison table
        """
        comparison = []

        for model_name, cv_results in zip(model_names, cv_results_list, strict=False):
            comparison.append(
                {
                    'model': model_name,
                    'mean_r2': cv_results['mean_metrics']['r2'],
                    'std_r2': cv_results['std_metrics']['r2'],
                    'mean_rmse': cv_results['mean_metrics']['rmse'],
                    'std_rmse': cv_results['std_metrics']['rmse'],
                    'mean_mae': cv_results['mean_metrics']['mae'],
                    'std_mae': cv_results['std_metrics']['mae'],
                    'n_folds': cv_results['n_successful_folds'],
                }
            )

        df = pd.DataFrame(comparison)

        # Sort by mean R² (descending)
        df = df.sort_values('mean_r2', ascending=False).reset_index(drop=True)

        # Add best model flag
        df['best_model'] = False
        df.loc[0, 'best_model'] = True

        return df
