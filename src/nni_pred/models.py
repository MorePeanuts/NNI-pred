import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass


@dataclass
class RandomForestConfig:
    pass


class NNIPredictorRF:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required for leaf nodes
            max_features: Number of features considered when searching for the best split
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(self, X, y) -> 'NNIPredictorRF':
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
