"""
Final Model Trainer for NNI Prediction

After cross-validation model selection, retrain the best model for each pollutant
on the full dataset. These models will be used for SHAP analysis and final predictions.
"""

import pandas as pd
import joblib
from pathlib import Path

from ..preprocessing import get_feature_groups, CVCompatiblePreprocessingPipeline
from ..models import ElasticNetModel, RandomForestModel, XGBoostModel


class FinalModelTrainer:
    """
    Retrain best models on full dataset for SHAP analysis.

    After nested CV determines the best model for each pollutant,
    this class:
    1. Loads the CV results
    2. Fits preprocessing on full data
    3. Trains final model on full data
    4. Saves model + preprocessor for later use
    """

    def __init__(self, verbose: int = 1):
        """
        Initialize final model trainer.

        Args:
            verbose: Verbosity level
        """
        self.verbose = verbose
        self.feature_groups = get_feature_groups()

    def train_final_models(
        self,
        data_path: str,
        cv_results_path: str,
        output_dir: str,
    ):
        """
        Train final models for all pollutants.

        Args:
            data_path: Path to processed_data.csv
            cv_results_path: Path to nested_cv_results.csv
            output_dir: Directory to save final models
        """
        print(f'\n{"=" * 80}')
        print('FINAL MODEL TRAINING ON FULL DATASET')
        print(f'{"=" * 80}\n')

        # Load data
        print(f'Loading data from: {data_path}')
        df = pd.read_csv(data_path)
        print(f'Data shape: {df.shape}')

        # Load CV results
        print(f'\nLoading CV results from: {cv_results_path}')
        cv_results = pd.read_csv(cv_results_path)

        # Filter to best models only
        best_models = cv_results[cv_results['best_model']]
        print(f'Best models to train: {len(best_models)}')

        # Prepare data
        target_cols = self.feature_groups['targets']
        metadata_cols = ['ID', 'Lon', 'Lat']
        exclude_cols = target_cols + metadata_cols
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y_dict = {col: df[col] for col in target_cols if col in df.columns}

        # Initialize model classes
        model_classes = {
            'ElasticNet': ElasticNetModel,
            'RandomForest': RandomForestModel,
            'XGBoost': XGBoostModel,
        }

        # Output directory
        output_path = Path(output_dir) / 'final_models'
        output_path.mkdir(parents=True, exist_ok=True)

        # Train final models
        print(f'\n{"=" * 80}')
        print('TRAINING FINAL MODELS')
        print(f'{"=" * 80}\n')

        for idx, row in best_models.iterrows():
            pollutant = row['pollutant']
            model_name = row['model_name']

            print(f'Training {model_name} for {pollutant}...')
            print(f'{"-" * 80}')

            if pollutant not in y_dict:
                print(f'  ERROR: Pollutant {pollutant} not found in data. Skipping...')
                continue

            try:
                # Initialize model
                model_class = model_classes[model_name]  # type: ignore
                model = model_class()

                # Fit preprocessing on FULL data
                preprocessor = CVCompatiblePreprocessingPipeline(
                    model_type=model.get_model_type(),
                    feature_groups=self.feature_groups,
                )
                X_processed = preprocessor.fit_transform(X)

                if self.verbose >= 1:
                    print(f'  Preprocessed shape: {X_processed.shape}')
                    if hasattr(preprocessor, 'get_summary'):
                        summary = preprocessor.get_summary()
                        if 'grouped_pca' in summary:
                            pca_summary = summary['grouped_pca']
                            if 'group2_agro' in pca_summary:
                                print(
                                    f'  Group2 PCA components: {pca_summary["group2_agro"]["n_components"]}'
                                )
                            if 'group3_socio' in pca_summary:
                                print(
                                    f'  Group3 PCA components: {pca_summary["group3_socio"]["n_components"]}'
                                )

                # Train model on full data
                model.fit(X_processed, y_dict[pollutant])

                # Prepare save object
                save_obj = {
                    'model': model,
                    'preprocessor': preprocessor,
                    'model_name': model_name,
                    'pollutant': pollutant,
                    'cv_metrics': {
                        'mean_r2': row['mean_r2'],
                        'std_r2': row['std_r2'],
                        'mean_rmse': row['mean_rmse'],
                        'std_rmse': row['std_rmse'],
                        'mean_mae': row['mean_mae'],
                        'std_mae': row['std_mae'],
                    },
                    'training_samples': len(X),
                    'n_features_processed': X_processed.shape[1],
                }

                # Save
                save_path = output_path / f'{pollutant}_{model_name}.pkl'
                joblib.dump(save_obj, save_path)

                print(f'  ✓ Saved to: {save_path}')
                print(f'  CV R² = {row["mean_r2"]:.4f} ± {row["std_r2"]:.4f}')
                print(f'  Training samples: {len(X)}')
                print(f'  Processed features: {X_processed.shape[1]}')
                print()

            except Exception as e:
                print(f'  ERROR training {model_name} for {pollutant}: {str(e)}')
                print('  Skipping this model...\n')
                continue

        print(f'{"=" * 80}')
        print('FINAL MODEL TRAINING COMPLETE')
        print(f'{"=" * 80}')
        print(f'Models saved to: {output_path}')
        print('\nThese models can now be used for:')
        print('  - SHAP analysis and interpretation')
        print('  - Making predictions on new data')
        print('  - Feature importance analysis')
        print(f'{"=" * 80}\n')

    @staticmethod
    def load_final_model(model_path: str) -> dict:
        """
        Load a saved final model.

        Args:
            model_path: Path to .pkl file

        Returns:
            Dictionary with model, preprocessor, and metadata
        """
        return joblib.load(model_path)

    @staticmethod
    def predict_with_final_model(model_dict: dict, X_new: pd.DataFrame) -> pd.Series:
        """
        Make predictions using a loaded final model.

        Args:
            model_dict: Dictionary from load_final_model()
            X_new: New feature data (before preprocessing)

        Returns:
            Predictions
        """
        # Preprocess
        preprocessor = model_dict['preprocessor']
        X_processed = preprocessor.transform(X_new)

        # Predict
        model = model_dict['model']
        predictions = model.predict(X_processed)

        return pd.Series(predictions, index=X_new.index)
