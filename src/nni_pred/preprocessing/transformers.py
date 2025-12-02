"""
CV-Compatible Preprocessing Transformers for NNI Prediction

This module implements sklearn-compatible transformers that can be used within
cross-validation pipelines without causing data leakage. All transformers follow
the fit/transform pattern where fit() only uses training data.

Key classes:
- CVCompatibleSkewnessTransformer: Log transform high-skew features
- CVCompatibleGroupedPCA: Apply separate PCA to different feature groups
- CVCompatiblePreprocessingPipeline: Unified pipeline that adapts to model type
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

from .feature_groups import get_feature_groups


class CVCompatibleSkewnessTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible skewness correction transformer.

    Applies log(x+1) transformation to features with high skewness.
    The skewness is calculated ONLY on training data during fit().

    Attributes:
        skewness_threshold: float
            Threshold for applying log transformation (default: 0.75)
        skewness_dict_: dict[str, float]
            Skewness values for each feature (fitted)
        high_skew_features_: list[str]
            Features identified as high-skew (fitted)
        min_shifts_: dict[str, float]
            Minimum value shifts for handling negative values (fitted)
    """

    def __init__(self, skewness_threshold: float = 0.75):
        """
        Initialize transformer.

        Args:
            skewness_threshold: Threshold for |skew| to apply transformation
        """
        self.skewness_threshold = skewness_threshold

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit on training data: calculate skewness and identify high-skew features.

        Args:
            X: Training feature dataframe
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        # Identify categorical columns to exclude
        categorical_cols = ['Season', 'Landuse', 'Soil_landuse']

        # Also check for one-hot encoded columns
        onehot_prefixes = ['Season_', 'Landuse_', 'Soil_landuse_']
        categorical_cols_in_data = [
            col
            for col in X.columns
            if col in categorical_cols or any(col.startswith(prefix) for prefix in onehot_prefixes)
        ]

        # Get continuous columns
        continuous_cols = [col for col in X.columns if col not in categorical_cols_in_data]

        # Calculate skewness on training data
        self.skewness_dict_ = {}
        self.high_skew_features_ = []
        self.min_shifts_ = {}

        for col in continuous_cols:
            # Calculate skewness
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
        """
        Apply log transformation to high-skew features.

        Args:
            X: Feature dataframe to transform

        Returns:
            Transformed dataframe
        """
        X_transformed = X.copy()

        for col in self.high_skew_features_:
            if col not in X_transformed.columns:
                continue  # Skip if column not present

            min_shift = self.min_shifts_[col]

            if min_shift < 0:
                # Shift to make all values positive before log
                X_transformed[col] = np.log1p(X[col] - min_shift + 1)
            else:
                # Direct log1p transformation
                X_transformed[col] = np.log1p(X[col])

        return X_transformed

    def get_transformation_summary(self) -> dict:
        """
        Get summary of transformations applied.

        Returns:
            Dictionary with transformation details
        """
        if not hasattr(self, 'skewness_dict_'):
            raise ValueError('Transformer has not been fitted yet')

        summary = {
            'n_features_total': len(self.skewness_dict_),
            'n_features_transformed': len(self.high_skew_features_),
            'threshold': self.skewness_threshold,
            'high_skew_features': self.high_skew_features_,
            'skewness_values': {
                feat: self.skewness_dict_[feat] for feat in self.high_skew_features_
            },
        }

        return summary


class CVCompatibleGroupedPCA(BaseEstimator, TransformerMixin):
    """
    Apply separate PCA to Group2 (Agro) and Group3 (Socio) features.

    Group1 (Natural) features are standardized but not reduced with PCA.
    Categorical features are one-hot encoded.

    Attributes:
        feature_groups: dict
            Feature group definitions
        variance_threshold: float
            Cumulative variance to retain in PCA (default: 0.95)
        group1_scaler_: StandardScaler (fitted)
        group2_scaler_: StandardScaler (fitted)
        group2_pca_: PCA (fitted)
        group3_scaler_: StandardScaler (fitted)
        group3_pca_: PCA (fitted)
        categorical_encoder_: OneHotEncoder (fitted)
    """

    def __init__(
        self,
        feature_groups: dict | None = None,
        variance_threshold: float = 0.95,
        apply_pca: bool = True,
    ):
        """
        Initialize transformer.

        Args:
            feature_groups: Feature group definitions (if None, will use default)
            variance_threshold: Variance to retain in PCA
            apply_pca: Whether to apply PCA to Group2/3 (if False, only scale)
        """
        self.feature_groups = feature_groups or get_feature_groups()
        self.variance_threshold = variance_threshold
        self.apply_pca = apply_pca

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit scalers and PCA on training data only.

        Args:
            X: Training feature dataframe
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        # 1. Encode categorical variables
        cat_cols = self.feature_groups['categorical']

        # Check if categorical columns are already one-hot encoded
        if all(col in X.columns for col in cat_cols):
            # Not yet encoded, fit encoder
            self.categorical_encoder_ = OneHotEncoder(drop='first', sparse_output=False)
            self.categorical_encoder_.fit(X[cat_cols])
            self._categorical_already_encoded = False
        else:
            # Already encoded (e.g., from previous preprocessing)
            self.categorical_encoder_ = None
            self._categorical_already_encoded = True

        # 2. Group1: Scale only (no PCA)
        group1_cols = self.feature_groups['group1_natural']
        group1_available = [col for col in group1_cols if col in X.columns]

        if len(group1_available) > 0:
            self.group1_scaler_ = StandardScaler()
            self.group1_scaler_.fit(X[group1_available])
            self._group1_cols = group1_available
        else:
            self.group1_scaler_ = None
            self._group1_cols = []

        # 3. Group2: Scale + optional PCA
        group2_cols = self.feature_groups['group2_agro']
        group2_available = [col for col in group2_cols if col in X.columns]

        if len(group2_available) > 0:
            self.group2_scaler_ = StandardScaler()
            X_group2_scaled = self.group2_scaler_.fit_transform(X[group2_available])

            if self.apply_pca:
                self.group2_pca_ = PCA(n_components=self.variance_threshold, random_state=42)
                self.group2_pca_.fit(X_group2_scaled)
            else:
                self.group2_pca_ = None
            self._group2_cols = group2_available
        else:
            self.group2_scaler_ = None
            self.group2_pca_ = None
            self._group2_cols = []

        # 4. Group3: Scale + optional PCA
        group3_cols = self.feature_groups['group3_socio']
        group3_available = [col for col in group3_cols if col in X.columns]

        if len(group3_available) > 0:
            self.group3_scaler_ = StandardScaler()
            X_group3_scaled = self.group3_scaler_.fit_transform(X[group3_available])

            if self.apply_pca:
                self.group3_pca_ = PCA(n_components=self.variance_threshold, random_state=42)
                self.group3_pca_.fit(X_group3_scaled)
            else:
                self.group3_pca_ = None
            self._group3_cols = group3_available
        else:
            self.group3_scaler_ = None
            self.group3_pca_ = None
            self._group3_cols = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform with fitted transformers.

        Args:
            X: Feature dataframe to transform

        Returns:
            Transformed dataframe with standardized Group1, PCA-reduced Group2/3,
            and one-hot encoded categorical features
        """
        transformed_parts = []

        # 1. Categorical features
        if not self._categorical_already_encoded:
            cat_cols = self.feature_groups['categorical']
            X_cat = self.categorical_encoder_.transform(X[cat_cols])  # type: ignore
            cat_feature_names = self._get_cat_feature_names()
            X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names, index=X.index)  # type: ignore
            transformed_parts.append(X_cat_df)
        else:
            # Extract one-hot encoded columns
            onehot_cols = [
                col
                for col in X.columns
                if any(col.startswith(f'{cat}_') for cat in self.feature_groups['categorical'])
            ]
            if len(onehot_cols) > 0:
                transformed_parts.append(X[onehot_cols])

        # 2. Group1: Standardize only
        if self.group1_scaler_ is not None and len(self._group1_cols) > 0:
            X_group1_scaled = self.group1_scaler_.transform(X[self._group1_cols])
            X_group1_df = pd.DataFrame(
                X_group1_scaled,
                columns=self._group1_cols,  # type: ignore
                index=X.index,
            )
            transformed_parts.append(X_group1_df)

        # 3. Group2: Scale + optional PCA
        if self.group2_scaler_ is not None and len(self._group2_cols) > 0:
            X_group2_scaled = self.group2_scaler_.transform(X[self._group2_cols])  # type: ignore

            if self.group2_pca_ is not None:
                # Apply PCA
                X_group2_pca = self.group2_pca_.transform(X_group2_scaled)
                n_comp2 = X_group2_pca.shape[1]
                X_group2_df = pd.DataFrame(
                    X_group2_pca,
                    columns=[f'PC_Agro_{i + 1}' for i in range(n_comp2)],  # type: ignore
                    index=X.index,
                )
            else:
                # Just use scaled features (no PCA)
                X_group2_df = pd.DataFrame(
                    X_group2_scaled,
                    columns=self._group2_cols,  # type: ignore
                    index=X.index,
                )
            transformed_parts.append(X_group2_df)

        # 4. Group3: Scale + optional PCA
        if self.group3_scaler_ is not None and len(self._group3_cols) > 0:
            X_group3_scaled = self.group3_scaler_.transform(X[self._group3_cols])  # type: ignore

            if self.group3_pca_ is not None:
                # Apply PCA
                X_group3_pca = self.group3_pca_.transform(X_group3_scaled)
                n_comp3 = X_group3_pca.shape[1]
                X_group3_df = pd.DataFrame(
                    X_group3_pca,
                    columns=[f'PC_Socio_{i + 1}' for i in range(n_comp3)],  # type: ignore
                    index=X.index,
                )
            else:
                # Just use scaled features (no PCA)
                X_group3_df = pd.DataFrame(
                    X_group3_scaled,
                    columns=self._group3_cols,  # type: ignore
                    index=X.index,
                )
            transformed_parts.append(X_group3_df)

        # Concatenate all transformed parts
        if len(transformed_parts) == 0:
            raise ValueError('No features were transformed')

        return pd.concat(transformed_parts, axis=1)

    def _get_cat_feature_names(self) -> list[str]:
        """Generate one-hot encoded feature names."""
        if self.categorical_encoder_ is None:
            return []

        cat_cols = self.feature_groups['categorical']
        feature_names = []

        for i, cat_col in enumerate(cat_cols):
            categories = self.categorical_encoder_.categories_[i][1:]  # drop first
            for cat in categories:
                feature_names.append(f'{cat_col}_{cat}')

        return feature_names

    def get_pca_summary(self) -> dict:
        """
        Get summary of PCA results.

        Returns:
            Dictionary with PCA statistics
        """
        if not hasattr(self, 'group2_pca_'):
            raise ValueError('Transformer has not been fitted yet')

        summary = {}

        if self.group2_pca_ is not None:
            summary['group2_agro'] = {
                'n_components': self.group2_pca_.n_components_,
                'explained_variance_ratio': self.group2_pca_.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(self.group2_pca_.explained_variance_ratio_)[-1],
            }

        if self.group3_pca_ is not None:
            summary['group3_socio'] = {
                'n_components': self.group3_pca_.n_components_,
                'explained_variance_ratio': self.group3_pca_.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(self.group3_pca_.explained_variance_ratio_)[-1],
            }

        return summary


class CVCompatiblePreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Unified preprocessing pipeline that adapts based on model type.

    For Elastic Net (linear):  Skewness correction → Grouped PCA
    For RF/XGBoost (tree):     Grouped PCA (optional) or just scaling

    Attributes:
        model_type: str
            'linear' or 'tree'
        feature_groups: dict
            Feature group definitions
        skewness_threshold: float
            Threshold for skewness correction
        pca_variance: float
            Variance to retain in PCA
        use_pca_for_tree: bool
            Whether to apply PCA for tree models (default: True)
        skewness_transformer_: CVCompatibleSkewnessTransformer (fitted, if linear)
        grouped_pca_: CVCompatibleGroupedPCA (fitted)
    """

    def __init__(
        self,
        model_type: str = 'tree',
        feature_groups: dict | None = None,
        skewness_threshold: float = 0.75,
        pca_variance: float = 0.95,
        use_pca_for_tree: bool = True,
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            model_type: 'linear' or 'tree'
            feature_groups: Feature group definitions
            skewness_threshold: Threshold for skewness correction
            pca_variance: Variance to retain in PCA
            use_pca_for_tree: Whether to apply PCA for tree models (default: True)
        """
        if model_type not in ['linear', 'tree']:
            raise ValueError(f"model_type must be 'linear' or 'tree', got {model_type}")

        self.model_type = model_type
        self.feature_groups = feature_groups or get_feature_groups()
        self.skewness_threshold = skewness_threshold
        self.pca_variance = pca_variance
        self.use_pca_for_tree = use_pca_for_tree

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit preprocessing chain on training data.

        Args:
            X: Training feature dataframe
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        X_current = X.copy()

        # Step 1: Skewness correction (only for linear models)
        if self.model_type == 'linear':
            self.skewness_transformer_ = CVCompatibleSkewnessTransformer(self.skewness_threshold)
            self.skewness_transformer_.fit(X_current)
            X_current = self.skewness_transformer_.transform(X_current)

        # Step 2: Grouped PCA
        # For linear models: always use PCA
        # For tree models: use PCA only if use_pca_for_tree=True
        apply_pca = self.model_type == 'linear' or (self.model_type == 'tree' and self.use_pca_for_tree)

        self.grouped_pca_ = CVCompatibleGroupedPCA(
            self.feature_groups,
            self.pca_variance,
            apply_pca=apply_pca
        )
        self.grouped_pca_.fit(X_current)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing chain.

        Args:
            X: Feature dataframe to transform

        Returns:
            Transformed dataframe
        """
        X_current = X.copy()

        # Step 1: Skewness correction (only for linear models)
        if self.model_type == 'linear':
            if not hasattr(self, 'skewness_transformer_'):
                raise ValueError('Pipeline has not been fitted yet')
            X_current = self.skewness_transformer_.transform(X_current)

        # Step 2: Grouped PCA (all models)
        if not hasattr(self, 'grouped_pca_'):
            raise ValueError('Pipeline has not been fitted yet')
        X_current = self.grouped_pca_.transform(X_current)

        return X_current

    def get_summary(self) -> dict:
        """
        Get summary of preprocessing applied.

        Returns:
            Dictionary with preprocessing details
        """
        if not hasattr(self, 'grouped_pca_'):
            raise ValueError('Pipeline has not been fitted yet')

        summary = {
            'model_type': self.model_type,
            'grouped_pca': self.grouped_pca_.get_pca_summary(),
        }

        if self.model_type == 'linear' and hasattr(self, 'skewness_transformer_'):
            summary['skewness'] = self.skewness_transformer_.get_transformation_summary()

        return summary
