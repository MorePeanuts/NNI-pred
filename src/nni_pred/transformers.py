import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .data import FeatureGroups, get_feature_groups


class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset_: float | None = None):
        self.offset_ = offset_

    def fit(self, y, sample_weight=None):
        if self.offset_ is not None:
            return self
        else:
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


class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, feature_groups: FeatureGroups | None = None, skewness_threshold: float = 0.75
    ):
        self.feature_groups = feature_groups or get_feature_groups()
        self.skewness_threshold = skewness_threshold

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(self.feature_groups, FeatureGroups)

        categorical_prefixes = self.feature_groups.categorical
        categorical_cols = [
            col
            for col in X.columns
            if any(col.startswith(prefix) for prefix in categorical_prefixes)
        ]
        self.continuous_cols = [col for col in X.columns if col not in categorical_cols]

        # Calculate skewness on training data
        self.skewness_dict_ = {}
        self.high_skew_features_ = []
        self.min_shifts_ = {}

        for col in self.continuous_cols:
            skew_val = stats.skew(X[col].dropna())
            self.skewness_dict_[col] = skew_val

            # Identify high-skew features
            if abs(skew_val) > self.skewness_threshold:
                self.high_skew_features_.append(col)

                # Store min value for shift (handle negative values)
                min_val = X[col].min()
                self.min_shifts_[col] = min_val if min_val < 0 else 0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()

        for col in self.high_skew_features_:
            if col not in X_transformed.columns:
                continue

            min_shift = self.min_shifts_[col]

            if min_shift < 0:
                # Shift to make all values positive before log
                X_transformed[col] = np.log1p(X[col] - min_shift + 1)
            else:
                # Direct log1p transformation
                X_transformed[col] = np.log1p(X[col])

        return X_transformed


def get_feature_engineering(
    model_type: Literal['linear', 'rf', 'xgb'], random_state: int = 42
) -> Pipeline:
    feature_groups = get_feature_groups()
    match model_type:
        case 'rf' | 'xgb':
            pca = GroupedPCA(random_state=random_state)
            column_transformer = ColumnTransformer(
                transformers=[('pca', pca, pca.get_feature_cols())], remainder='passthrough'
            )
            return Pipeline([('column_transformer', column_transformer)])
        case 'linear':
            skew_shift = SkewnessTransformer()
            group1_scaler = StandardScaler()
            pca = GroupedPCA(random_state=random_state)
            column_transformer = ColumnTransformer(
                transformers=[
                    ('pca', pca, pca.get_feature_cols()),
                    ('group1_scaler', group1_scaler, feature_groups.group1_natural),
                ],
                remainder='passthrough',
            )
            return Pipeline(
                [('skew_shift', skew_shift), ('column_transformer', column_transformer)]
            )
