"""
Pipeline for data feature engineering.
"""

import warnings
import sklearn
import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from .data import VariableGroups


warnings.filterwarnings('ignore')
sklearn.set_config(transform_output='pandas')


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
        variance_threshold: float = 0.95,
        random_state: int = 42,
    ):
        self.variance_threshold = variance_threshold
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        # Group agro: Scale + PCA
        group2_cols = VariableGroups.group_agro
        self.group2_scaler_ = StandardScaler()
        X_group2 = self.group2_scaler_.fit_transform(X[group2_cols])
        self.group2_pca_ = PCA(n_components=self.variance_threshold, random_state=self.random_state)
        self.group2_pca_.fit(X_group2)

        # Group socio: Scale + PCA
        group3_cols = VariableGroups.group_socio
        self.group3_scaler_ = StandardScaler()
        X_group3 = self.group3_scaler_.fit_transform(X[group3_cols])
        self.group3_pca_ = PCA(n_components=self.variance_threshold, random_state=self.random_state)
        self.group3_pca_.fit(X_group3)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Group agro: Scale + PCA
        group2_cols = VariableGroups.group_agro
        X_group2 = self.group2_scaler_.transform(X[group2_cols])
        X_group2 = self.group2_pca_.transform(X_group2)
        # n_comp2 = X_group2.shape[1]
        # X_group2_df = pd.DataFrame(
        #     X_group2,
        #     columns=[f'PC_Agro_{i + 1}' for i in range(n_comp2)],  # type: ignore
        #     index=X.index,
        # )

        # Group socio: Scale + PCA
        group3_cols = VariableGroups.group_socio
        X_group3 = self.group3_scaler_.transform(X[group3_cols])
        X_group3 = self.group3_pca_.transform(X_group3)
        # n_comp3 = X_group3.shape[1]
        # X_group3_df = pd.DataFrame(
        #     X_group3,
        #     columns=[f'PC_Socio_{i + 1}' for i in range(n_comp3)],  # type: ignore
        #     index=X.index,
        # )
        X_combined = np.hstack([X_group2, X_group3])
        cols = [f'PC_Agro_{i + 1}' for i in range(X_group2.shape[1])] + [
            f'PC_Socio_{i + 1}' for i in range(X_group3.shape[1])
        ]

        return pd.DataFrame(X_combined, columns=cols, index=X.index)  # type: ignore

    def get_feature_cols(self):
        return VariableGroups.group_agro + VariableGroups.group_socio

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [f'PC_Agro_{i}' for i in range(self.group2_pca_.n_components_)]
            + [f'PC_Socio_{i}' for i in range(self.group3_pca_.n_components_)]
        )

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
    def __init__(self, skewness_threshold: float = 0.75):
        self.skewness_threshold = skewness_threshold

    def fit(self, X: pd.DataFrame, y=None):
        self.continuous_cols = [
            col
            for col in X.columns
            if col
            in VariableGroups.group_agro
            + VariableGroups.group_natural
            + VariableGroups.group_socio
            + VariableGroups.soil_metabolites
            + VariableGroups.soil_parent
        ]

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


def get_preprocessing_pipeline(
    model_type: Literal['linear', 'rf', 'xgb'],
    # WARNING: Whether to use target to control input features
    target: str | None = None,
    random_state: int = 42,
) -> Pipeline:
    features = (
        VariableGroups.categorical
        + VariableGroups.soil_parent
        + VariableGroups.soil_metabolites
        + VariableGroups.group_natural
        + VariableGroups.group_agro
        + VariableGroups.group_socio
    )
    match model_type:
        case 'rf' | 'xgb':
            pca = GroupedPCA(random_state=random_state)
            keeper = ColumnTransformer(
                transformers=[
                    ('keep', 'passthrough', features),
                ],
                remainder='drop',
                verbose_feature_names_out=False,
            )
            column_transformer = ColumnTransformer(
                transformers=[
                    ('encoder', OrdinalEncoder(), VariableGroups.categorical),
                    ('pca', pca, pca.get_feature_cols()),
                ],
                remainder='passthrough',
            )
            return Pipeline([('keeper', keeper), ('column_transformer', column_transformer)])
        case 'linear':
            keeper = ColumnTransformer(
                transformers=[
                    ('keep', 'passthrough', features),
                ],
                remainder='drop',
                verbose_feature_names_out=False,
            )
            encoder = ColumnTransformer(
                transformers=[
                    (
                        'encoder',
                        OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'),
                        VariableGroups.categorical,
                    )
                ],
                remainder='passthrough',
                verbose_feature_names_out=False,
            )
            skew_shift = SkewnessTransformer()
            group_natural_scaler = StandardScaler()
            pca = GroupedPCA(random_state=random_state)
            column_transformer = ColumnTransformer(
                transformers=[
                    ('pca', pca, pca.get_feature_cols()),
                    (
                        'group_natural_scaler',
                        group_natural_scaler,
                        VariableGroups.group_natural
                        + VariableGroups.soil_parent
                        + VariableGroups.soil_metabolites,
                    ),
                ],
                remainder='passthrough',
                verbose_feature_names_out=False,
            )
            return Pipeline(
                [
                    ('keeper', keeper),
                    ('encoder', encoder),
                    ('skew_shift', skew_shift),
                    ('column_transformer', column_transformer),
                ]
            )
