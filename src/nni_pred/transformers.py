import numpy as np
import pandas as pd
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .data import FeatureGroups, get_feature_groups


class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.offset_ = None

    def fit(self, y, sample_weight=None):
        positive_y = y[y > 0]
        min_pos = np.min(positive_y) if len(positive_y) > 0 else 1e-6
        self.offset_ = min_pos / 2.0
        return self

    def transform(self, y):
        return np.log(y + self.offset_)

    def inverse_transform(self, y):
        return np.exp(y) - self.offset_


class GroupedPCA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_groups: FeatureGroups | None = None,
        variance_threshold: float = 0.95,
        random_state: int = 42,
    ):
        self.feature_groups = feature_groups or get_feature_groups()
        self.variance_threshold = variance_threshold
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(self.feature_groups, FeatureGroups)

        # Group2: Scale + PCA
        group2_cols = self.feature_groups.group2_agro
        self.group2_scaler_ = StandardScaler()
        X_group2 = self.group2_scaler_.fit_transform(X[group2_cols])
        self.group2_pca_ = PCA(n_components=self.variance_threshold, random_state=self.random_state)
        self.group2_pca_.fit(X_group2)

        # Group3: Scale + PCA
        group3_cols = self.feature_groups.group3_socio
        self.group3_scaler_ = StandardScaler()
        X_group3 = self.group3_scaler_.fit_transform(X[group3_cols])
        self.group3_pca_ = PCA(n_components=self.variance_threshold, random_state=self.random_state)
        self.group3_pca_.fit(X_group3)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(self.feature_groups, FeatureGroups)

        # Group2: Scale + PCA
        group2_cols = self.feature_groups.group2_agro
        X_group2 = self.group2_scaler_.transform(X[group2_cols])
        X_group2 = self.group2_pca_.transform(X_group2)
        n_comp2 = X_group2.shape[1]
        X_group2_df = pd.DataFrame(
            X_group2,
            columns=[f'PC_Agro_{i + 1}' for i in range(n_comp2)],  # type: ignore
            index=X.index,
        )

        # Group3: Scale + PCA
        group3_cols = self.feature_groups.group3_socio
        X_group3 = self.group3_scaler_.transform(X[group3_cols])
        X_group3 = self.group3_pca_.transform(X_group3)
        n_comp3 = X_group3.shape[1]
        X_group3_df = pd.DataFrame(
            X_group3,
            columns=[f'PC_Socio_{i + 1}' for i in range(n_comp3)],  # type: ignore
            index=X.index,
        )

        return pd.concat([X_group2_df, X_group3_df], axis=1)

    def get_feature_cols(self):
        assert isinstance(self.feature_groups, FeatureGroups)
        return self.feature_groups.group2_agro + self.feature_groups.group3_socio

    def get_pca_summary(self):
        assert hasattr(self, 'group2_pca_') and hasattr(self, 'group3_pca_')
        return {
            'group2_agro': {
                'n_components': self.group2_pca_.n_components_,
                'explained_variance_ratio': self.group2_pca_.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(self.group2_pca_.explained_variance_ratio_)[-1],
            },
            'group3_socio': {
                'n_components': self.group3_pca_.n_components_,
                'explained_variance_ratio': self.group3_pca_.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(self.group3_pca_.explained_variance_ratio_)[-1],
            },
        }


def get_column_transformer(model_type: Literal['linear', 'rf', 'xgb'], random_state: int = 42):
    match model_type:
        case 'rf' | 'xgb':
            pca = GroupedPCA(random_state=random_state)
            return ColumnTransformer(
                transformers=[('pca', pca, pca.get_feature_cols())], remainder='passthrough'
            )
        case 'linear':
            ...
