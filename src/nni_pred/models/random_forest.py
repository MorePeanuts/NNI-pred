"""
Random Forest Model Wrapper for NNI Prediction

Random Forest is an ensemble of decision trees using bagging.
"""

from sklearn.ensemble import RandomForestRegressor
from .base import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest regression wrapper.

    Random Forest uses bagging of decision trees for robust predictions.
    It does NOT require skewness correction in preprocessing.
    """

    def __init__(self):
        """Initialize model wrapper."""
        self.model = None

    def get_param_grid(self) -> dict:
        """
        Return hyperparameter grid for GridSearchCV.

        Returns:
            Dictionary with hyperparameter search space
        """
        return {
            'n_estimators': [100, 200, 500],  # Number of trees
            'max_depth': [8, 15, 20],  # Tree depth
            'min_samples_split': [5, 10, 15],  # Min samples to split
            'min_samples_leaf': [2, 4, 6],  # Min samples per leaf
            'max_features': ['sqrt', 'log2'],  # Features per split
        }
        # Total combinations: 3 × 3 × 3 × 3 × 2 = 162

    def get_sklearn_model(self):
        """
        Return sklearn RandomForestRegressor instance.

        Returns:
            RandomForestRegressor with fixed random_state and parallelization
        """
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
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
        Fit the Random Forest model.

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
            Array of feature importances (Gini importance)
        """
        if self.model is None:
            raise ValueError('Model has not been fitted yet')
        return self.model.feature_importances_
