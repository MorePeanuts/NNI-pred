"""
Batch Trainer for NNI Prediction

This module orchestrates training of all pollutant-model combinations:
11 pollutants × 3 models = 33 combinations

For each pollutant, it:
1. Trains all 3 candidate models with nested CV
2. Selects the best model (highest mean R²)
3. Saves results and generates reports
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from ..preprocessing import get_feature_groups
from ..validation import SpatialGroupGenerator, NestedSpatialCV
from ..models import ElasticNetModel, RandomForestModel, XGBoostModel


class BatchTrainer:
    """
    Train all 11 pollutants × 3 models = 33 combinations.

    This class manages the entire training pipeline:
    - Loads data
    - Generates spatial groups
    - Runs nested CV for each combination
    - Selects best model per pollutant
    - Saves comprehensive results
    """

    def __init__(
        self,
        n_outer: int = 5,
        n_inner: int = 4,
        random_state: int = 42,
        verbose: int = 1,
        use_pca_for_tree: bool = True,
        inverse_transform_targets: bool = True,
    ):
        """
        Initialize batch trainer.

        Args:
            n_outer: Number of outer CV folds
            n_inner: Number of inner CV folds
            random_state: Random seed
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
            use_pca_for_tree: Whether to apply PCA for tree models (default: True)
            inverse_transform_targets: Whether to apply inverse transformer to targets (default: True)
        """
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.random_state = random_state
        self.verbose = verbose
        self.use_pca_for_tree = use_pca_for_tree
        self.inverse_transform_targets = inverse_transform_targets
        self.feature_groups = get_feature_groups()

    def train_all_combinations(
        self,
        data_path: str,
        output_dir: str,
        pollutants: list[str] | None = None,
        grid_size: str = 'full',
    ) -> pd.DataFrame:
        """
        Main orchestration function to train all combinations.

        Args:
            data_path: Path to processed_data.csv
            output_dir: Directory to save results
            pollutants: List of pollutants to train (if None, train all 11)
            grid_size: 'small', 'medium', or 'full' hyperparameter grid

        Returns:
            DataFrame with results summary
        """
        # Load data
        print(f'\n{"=" * 80}')
        print('BATCH TRAINING: NESTED SPATIAL CROSS-VALIDATION')
        print(f'{"=" * 80}')
        print(f'Loading data from: {data_path}\n')

        df = pd.read_csv(data_path)
        print(f'Data shape: {df.shape}')

        # Prepare data
        X, y_dict, groups = self._prepare_data(df)

        # Determine pollutants to train
        if pollutants is None or pollutants == ['all']:
            pollutants_to_train = self.feature_groups['targets']
        else:
            pollutants_to_train = pollutants

        print(f'\nPollutants to train: {len(pollutants_to_train)}')
        print('Models per pollutant: 3 (ElasticNet, RandomForest, XGBoost)')
        print(f'Total combinations: {len(pollutants_to_train) * 3}')
        print(f'Grid size: {grid_size}')
        print(f'PCA for tree models: {self.use_pca_for_tree}')

        # Initialize models with appropriate grid
        models = self._get_models(grid_size)

        # Initialize nested CV
        nested_cv = NestedSpatialCV(
            n_outer=self.n_outer,
            n_inner=self.n_inner,
            random_state=self.random_state,
            verbose=self.verbose,
            inverse_transform_targets=self.inverse_transform_targets,
        )

        # Results storage
        results = []
        detailed_results = {}

        # Train all combinations
        for pollutant in tqdm(pollutants_to_train, desc='Pollutants', position=0):
            print(f'\n{"=" * 80}')
            print(f'POLLUTANT: {pollutant}')
            print(f'{"=" * 80}')

            y = y_dict[pollutant]
            pollutant_results = {}

            for model_name, model in tqdm(
                models.items(), desc=f'  Models for {pollutant}', position=1, leave=False
            ):
                print(f'\n  Model: {model_name}')
                print(f'  {"-" * 76}')

                try:
                    # Get custom param grid if specified
                    param_grid = self._get_param_grid(model, grid_size)

                    # Run nested CV
                    cv_results = nested_cv.run_nested_cv(
                        X,
                        y,
                        model,
                        self.feature_groups,
                        groups,
                        param_grid,  # type: ignore
                        use_pca_for_tree=self.use_pca_for_tree,
                    )

                    pollutant_results[model_name] = cv_results

                    # Add to results
                    results.append(
                        {
                            'pollutant': pollutant,
                            'model_name': model_name,
                            'mean_r2': cv_results['mean_metrics']['r2'],
                            'std_r2': cv_results['std_metrics']['r2'],
                            'mean_rmse': cv_results['mean_metrics']['rmse'],
                            'std_rmse': cv_results['std_metrics']['rmse'],
                            'mean_mae': cv_results['mean_metrics']['mae'],
                            'std_mae': cv_results['std_metrics']['mae'],
                            'n_folds': cv_results['n_successful_folds'],
                        }
                    )

                except Exception as e:
                    print(f'\n  ERROR training {model_name} for {pollutant}: {str(e)}')
                    print('  Skipping this model...\n')
                    continue

            # Select best model for this pollutant
            if len(pollutant_results) > 0:
                best_model_name = max(
                    pollutant_results, key=lambda m: pollutant_results[m]['mean_metrics']['r2']
                )

                print(f'\n  {"=" * 76}')
                print(f'  BEST MODEL for {pollutant}: {best_model_name}')
                print(f'  R² = {pollutant_results[best_model_name]["mean_metrics"]["r2"]:.4f}')
                print(f'  {"=" * 76}\n')

                # Mark best model
                for r in results:
                    if r['pollutant'] == pollutant:
                        r['best_model'] = r['model_name'] == best_model_name

            # Store detailed results
            detailed_results[pollutant] = pollutant_results

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path / 'nested_cv_results.csv', index=False)

        # Save detailed results as JSON
        detailed_json = self._convert_results_to_json(detailed_results)
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(detailed_json, f, indent=2)

        # Save configuration
        config = {
            'n_outer': self.n_outer,
            'n_inner': self.n_inner,
            'random_state': self.random_state,
            'grid_size': grid_size,
            'use_pca_for_tree': self.use_pca_for_tree,
            'inverse_transform_targets': True,
            'n_samples': len(df),
            'n_features': X.shape[1],
            'n_groups': len(np.unique(groups)),
            'pollutants_trained': pollutants_to_train,
        }
        with open(output_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Generate summary report
        self._generate_report(results_df, output_path)

        print(f'\n{"=" * 80}')
        print('BATCH TRAINING COMPLETE')
        print(f'{"=" * 80}')
        print(f'Results saved to: {output_path}')
        print('  - nested_cv_results.csv: Summary table')
        print('  - detailed_results.json: Full fold-level results')
        print('  - model_selection_report.txt: Human-readable report')
        print('  - config.json: Training configuration')
        print(f'{"=" * 80}\n')

        return results_df

    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features, targets, and spatial groups.

        Args:
            df: Loaded dataframe

        Returns:
            Tuple of (X, y_dict, groups)
        """
        print(f'\n{"=" * 60}')
        print('DATA PREPARATION')
        print(f'{"=" * 60}')

        # Extract targets
        target_cols = self.feature_groups['targets']
        y_dict = {col: df[col] for col in target_cols if col in df.columns}
        print(f'Target pollutants available: {len(y_dict)}')

        # Extract metadata columns
        metadata_cols = ['ID', 'Lon', 'Lat']

        # Features = all columns except targets and metadata
        exclude_cols = list(y_dict.keys()) + metadata_cols
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols]

        print(f'Features: {X.shape[1]} columns')
        print(f'Samples: {X.shape[0]} rows')

        # Generate spatial groups
        groups = SpatialGroupGenerator.generate_groups(df)

        # Validate groups
        validation = SpatialGroupGenerator.validate_groups(df, groups)  # type: ignore
        SpatialGroupGenerator.print_validation_report(validation)

        return X, y_dict, groups

    def _get_models(self, grid_size: str) -> dict:
        """
        Get model instances.

        Args:
            grid_size: 'small', 'medium', or 'full'

        Returns:
            Dictionary of model instances
        """
        return {
            'ElasticNet': ElasticNetModel(),
            'RandomForest': RandomForestModel(),
            'XGBoost': XGBoostModel(),
        }

    def _get_param_grid(self, model, grid_size: str) -> dict | None:
        """
        Get parameter grid based on size specification.

        Args:
            model: Model instance
            grid_size: 'small', 'medium', or 'full'

        Returns:
            Parameter grid dict or None (to use default)
        """
        if grid_size == 'full':
            return None  # Use default full grid

        model_name = model.__class__.__name__

        if grid_size == 'small':
            if model_name == 'ElasticNetModel':
                return {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.3, 0.5, 0.7],
                    'max_iter': [10000],
                }  # 3×3 = 9 combinations
            elif model_name == 'RandomForestModel':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt'],
                }  # 2×2×2 = 8 combinations
            elif model_name == 'XGBoostModel':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'reg_alpha': [0],
                    'reg_lambda': [1],
                }  # 2×2 = 4... let me add one more
                # Actually 2×2×1×1×1×1×1 = 4, but that's okay for small grid

        elif grid_size == 'medium':
            if model_name == 'ElasticNetModel':
                return None  # Use full grid (25 combinations - already reasonable)
            elif model_name == 'RandomForestModel':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt'],
                }  # 2×2×2×2 = 16 combinations
            elif model_name == 'XGBoostModel':
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'reg_alpha': [0],
                    'reg_lambda': [1],
                }  # 2×2×2 = 8 combinations

        return None  # Default to full grid

    def _convert_results_to_json(self, detailed_results: dict):
        """
        Convert numpy types to JSON-serializable types.

        Args:
            detailed_results: Nested dict with CV results

        Returns:
            JSON-serializable dict
        """

        def convert_value(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_value(item) for item in obj]
            else:
                return obj

        return convert_value(detailed_results)

    def _generate_report(self, results_df: pd.DataFrame, output_path: Path):
        """
        Generate human-readable report.

        Args:
            results_df: Results dataframe
            output_path: Directory to save report
        """
        report_path = output_path / 'model_selection_report.txt'

        with open(report_path, 'w') as f:
            f.write('=' * 80 + '\n')
            f.write('NESTED SPATIAL CROSS-VALIDATION RESULTS\n')
            f.write('Model Selection Report\n')
            f.write('=' * 80 + '\n\n')

            f.write('Configuration:\n')
            f.write(f'  Outer CV folds: {self.n_outer}\n')
            f.write(f'  Inner CV folds: {self.n_inner}\n')
            f.write(f'  Random state: {self.random_state}\n\n')

            f.write('=' * 80 + '\n')
            f.write('RESULTS BY POLLUTANT\n')
            f.write('=' * 80 + '\n\n')

            for pollutant in results_df['pollutant'].unique():
                f.write(f'\n{pollutant}\n')
                f.write('-' * 80 + '\n\n')

                pollutant_df = results_df[results_df['pollutant'] == pollutant].sort_values(  # type: ignore
                    'mean_r2', ascending=False
                )

                for _, row in pollutant_df.iterrows():
                    best_marker = ' [BEST MODEL]' if row.get('best_model', False) else ''
                    f.write(f'  {row["model_name"]}{best_marker}\n')
                    f.write(f'    R²   = {row["mean_r2"]:.4f} ± {row["std_r2"]:.4f}\n')
                    f.write(f'    RMSE = {row["mean_rmse"]:.4f} ± {row["std_rmse"]:.4f}\n')
                    f.write(f'    MAE  = {row["mean_mae"]:.4f} ± {row["std_mae"]:.4f}\n')
                    f.write(f'    Folds = {row["n_folds"]}/{self.n_outer}\n')
                    f.write('\n')

            f.write('\n' + '=' * 80 + '\n')
            f.write('SUMMARY: BEST MODELS PER POLLUTANT\n')
            f.write('=' * 80 + '\n\n')

            best_models = results_df[results_df.get('best_model', False)]
            for _, row in best_models.iterrows():
                f.write(
                    f'{row["pollutant"]:<20} {row["model_name"]:<15} R² = {row["mean_r2"]:.4f}\n'
                )

            f.write('\n' + '=' * 80 + '\n')
            f.write('END OF REPORT\n')
            f.write('=' * 80 + '\n')

        print(f'\nReport saved to: {report_path}')
