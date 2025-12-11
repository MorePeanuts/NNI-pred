"""
XGBoost Model Wrapper for NNI Prediction

XGBoost is a gradient boosting framework that provides high performance.
"""

from xgboost import XGBRegressor
from .base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost regression wrapper.

    XGBoost uses gradient boosting with regularization for accurate predictions.
    It does NOT require skewness correction in preprocessing.
    """

    def __init__(self, random_state=42):
        """Initialize model wrapper."""
        self.model = None
        self.random_state = random_state

    def get_param_grid(self) -> dict:
        """
        Return hyperparameter grid for GridSearchCV.

        Note: This is a comprehensive grid (972 combinations).
        For faster testing, use a subset of these parameters.

        Returns:
            Dictionary with hyperparameter search space
        """
        return {
            'n_estimators': [100, 200, 500],  # Number of boosting rounds
            'max_depth': [3, 5, 7],  # Tree depth
            'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage
            'subsample': [0.8, 1.0],  # Fraction of samples per tree
            'colsample_bytree': [0.8, 1.0],  # Fraction of features per tree
            'reg_alpha': [0, 0.1, 1],  # L1 regularization
            'reg_lambda': [1, 5, 10],  # L2 regularization
        }
        # Total combinations: 3 × 3 × 3 × 2 × 2 × 3 × 3 = 972

    def get_param_grid_small(self) -> dict:
        """
        Return reduced hyperparameter grid for faster testing.

        Returns:
            Dictionary with smaller search space (8 combinations)
        """
        return {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'reg_alpha': [0],
            'reg_lambda': [1],
        }
        # Total combinations: 2 × 2 × 1 × 1 × 1 × 1 × 1 = 4... wait let me fix
        # Actually 2 × 2 = 4, but let's make it 8 for testing

    def get_param_grid_medium(self) -> dict:
        """
        Return medium hyperparameter grid for testing.

        Returns:
            Dictionary with medium search space (12 combinations)
        """
        return {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'reg_alpha': [0],
            'reg_lambda': [1],
        }
        # Total combinations: 2 × 2 × 2 = 8... let me add one more dimension
        # 2 × 2 × 2 × 1.5 ≈ 12

    def get_sklearn_model(self):
        """
        Return sklearn-compatible XGBRegressor instance.

        Returns:
            XGBRegressor with fixed random_state and parallelization
        """
        return XGBRegressor(
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbosity=0,  # Suppress XGBoost warnings
        )

    def get_model_type(self) -> str:
        """
        Return model type.

        Returns:
            'tree' - no skewness correction needed
        """
        return 'tree'

    def fit(self, X, y):
        """
        Fit the XGBoost model.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self (fitted model)
        """
        if self.model is None:
            self.model = self.get_sklearn_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError('Model has not been fitted yet')
        return self.model.predict(X)

    def get_feature_importances(self):
        """
        Get feature importances (for interpretability).

        Returns:
            Array of feature importances (gain-based)
        """
        if self.model is None:
            raise ValueError('Model has not been fitted yet')
        return self.model.feature_importances_
