"""
Elastic Net Model Wrapper for NNI Prediction

Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization.
"""

from sklearn.linear_model import ElasticNet
from .base import BaseModel


class ElasticNetModel(BaseModel):
    """
    Elastic Net regression wrapper.

    Elastic Net is a linear model with combined L1/L2 regularization.
    It requires skewness correction in preprocessing.
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
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],  # Regularization strength
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],   # L1/L2 mix (1=Lasso, 0=Ridge)
            'max_iter': [10000],                      # Maximum iterations
        }
        # Total combinations: 5 × 5 × 1 = 25

    def get_sklearn_model(self):
        """
        Return sklearn ElasticNet instance.

        Returns:
            ElasticNet regressor with fixed random_state
        """
        return ElasticNet(random_state=42)

    def get_model_type(self) -> str:
        """
        Return model type.

        Returns:
            'linear' - requires skewness correction in preprocessing
        """
        return 'linear'

    def fit(self, X, y):
        """
        Fit the Elastic Net model.

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
            raise ValueError("Model has not been fitted yet")
        return self.model.predict(X)

    def get_coefficients(self):
        """
        Get model coefficients (for interpretability).

        Returns:
            Array of feature coefficients
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        return self.model.coef_
