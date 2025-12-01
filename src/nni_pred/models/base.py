"""
Base Model Interface for NNI Prediction

This module defines the abstract base class that all model wrappers must implement.
This ensures a unified interface for different model types (Elastic Net, RF, XGBoost).
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """
    Abstract base class for all model wrappers.

    All model implementations must inherit from this class and implement
    the required methods. This ensures a consistent interface across different
    model types for use in the training pipeline.
    """

    @abstractmethod
    def get_param_grid(self) -> dict:
        """
        Return hyperparameter grid for GridSearchCV.

        Returns:
            Dictionary mapping parameter names to lists of values to try.
            For use with sklearn.model_selection.GridSearchCV.
        """
        pass

    @abstractmethod
    def get_sklearn_model(self) -> Any:
        """
        Return underlying sklearn model instance.

        Returns:
            Instantiated sklearn-compatible model object (e.g., ElasticNet(),
            RandomForestRegressor(), XGBRegressor()).
        """
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """
        Return model type for preprocessing selection.

        Returns:
            'linear' for linear models (requires skewness correction)
            'tree' for tree-based models (no skewness correction)
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model.

        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target vector

        Returns:
            self (fitted model)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values (numpy array)
        """
        pass

    def get_model_name(self) -> str:
        """
        Get the model's display name.

        Returns:
            String name of the model (e.g., 'ElasticNet', 'RandomForest')
        """
        return self.__class__.__name__.replace('Model', '')
