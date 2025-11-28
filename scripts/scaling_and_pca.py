"""
Feature Scaling and Grouped PCA Dimensionality Reduction

This script implements the preprocessing pipeline including:
1. Skewness diagnosis and adaptive transformation (Log(x+1) for |skew| > 0.75)
2. Domain-driven feature grouping (Natural, Agricultural, Socio-Economic)
3. Grouped PCA with 95% variance threshold for Agro & Socio groups
4. Global standardization
5. Comprehensive reporting and visualization

Reference: Research plan in docs/research-plan-zh.md
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class FeaturePreprocessor:
    """
    Feature preprocessing pipeline with grouped PCA for NNIs prediction
    """

    def __init__(self, variance_threshold: float = 0.95, skewness_threshold: float = 0.75):
        """
        Initialize preprocessor with configuration

        Args:
            variance_threshold: Cumulative variance threshold for PCA (default: 0.95)
            skewness_threshold: Threshold for applying log transformation (default: 0.75)
        """
        self.variance_threshold = variance_threshold
        self.skewness_threshold = skewness_threshold

        # Data containers
        self.df = None
        self.X = None
        self.y = None
        self.metadata = None

        # Skewness diagnosis
        self.skewness_original = {}
        self.skewness_transformed = {}
        self.transformed_features = []

        # Feature groups
        self.feature_groups = {}

        # Transformers (to be saved)
        self.skewness_transformer = None
        self.categorical_encoder = None
        self.group1_scaler = None
        self.group2_scaler = None
        self.group2_pca = None
        self.group3_scaler = None
        self.group3_pca = None

        # Processed data
        self.X_final = None

    def load_data(self, input_path: str) -> pd.DataFrame:
        """
        Load merged dataset

        Args:
            input_path: Path to merged_data.csv

        Returns:
            Loaded dataframe
        """
        print(f'Loading data from {input_path}...')
        self.df = pd.read_csv(input_path)
        print(f'Data shape: {self.df.shape}')
        return self.df

    def separate_features_targets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Separate features, targets, and metadata

        Returns:
            X: feature dataframe
            y: target dataframe (11 pollutants)
            metadata: ID, Lon, Lat, Season
        """
        print('\nSeparating features, targets, and metadata...')

        # Target variables (11 pollutants)
        target_cols = [
            'THIA',
            'IMI',
            'CLO',
            'ACE',
            'DIN',
            'parentNNIs',
            'IMI-UREA',
            'DN-IMI',
            'DM-ACE',
            'CLO-UREA',
            'mNNIs',
        ]

        # Metadata columns (for record keeping, not for model input)
        metadata_cols = ['ID', 'Lon', 'Lat', 'Season']

        # Extract
        self.y = self.df[target_cols].copy()  # type: ignore
        self.metadata = self.df[metadata_cols].copy()  # type: ignore

        # Features = all columns except targets and ID/Lon/Lat
        # Note: Season, Landuse, Soil_landuse are features (will be one-hot encoded)
        exclude_cols = target_cols + ['ID', 'Lon', 'Lat']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]  # type: ignore

        self.X = self.df[feature_cols].copy()  # type: ignore

        print(f'Features: {self.X.shape[1]} columns')
        print(f'Targets: {self.y.shape[1]} columns')
        print(f'Metadata: {self.metadata.shape[1]} columns')

        return self.X, self.y, self.metadata  # type: ignore

    def diagnose_skewness(self, X: pd.DataFrame) -> dict[str, float]:
        """
        Calculate skewness for all continuous features

        Args:
            X: Feature dataframe

        Returns:
            Dictionary of {feature: skewness_value}
        """
        print('\nDiagnosing skewness...')

        # Separate categorical features
        categorical_cols = ['Season', 'Landuse', 'Soil_landuse']
        continuous_cols = [col for col in X.columns if col not in categorical_cols]

        skewness_dict = {}
        for col in continuous_cols:
            skew_val = stats.skew(X[col].dropna())
            skewness_dict[col] = skew_val

        self.skewness_original = skewness_dict

        # Count high-skewness features
        high_skew_count = sum(1 for v in skewness_dict.values() if abs(v) > self.skewness_threshold)

        print(f'Total continuous features: {len(continuous_cols)}')
        print(f'High skewness features (|skew| > {self.skewness_threshold}): {high_skew_count}')

        return skewness_dict

    def apply_skewness_correction(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Apply log(x+1) transformation for high-skewness features

        Args:
            X: Feature dataframe

        Returns:
            X_transformed: Transformed dataframe
            transformation_info: Dict of transformation details
        """
        print('\nApplying skewness correction...')

        categorical_cols = ['Season', 'Landuse', 'Soil_landuse']
        X_transformed = X.copy()
        transformation_info = {}

        for col in X.columns:
            if col in categorical_cols:
                continue

            skew_val = self.skewness_original.get(col, 0)

            if abs(skew_val) > self.skewness_threshold:
                # Apply log(x+1) transformation
                # Handle negative values by shifting
                min_val = X[col].min()
                if min_val < 0:
                    X_transformed[col] = np.log1p(X[col] - min_val + 1)
                    transformation_info[col] = {
                        'method': 'log(x-min+1)',
                        'min_shift': min_val,
                        'original_skew': skew_val,
                    }
                else:
                    X_transformed[col] = np.log1p(X[col])
                    transformation_info[col] = {
                        'method': 'log(x+1)',
                        'min_shift': 0,
                        'original_skew': skew_val,
                    }

                # Calculate new skewness
                new_skew = stats.skew(X_transformed[col].dropna())
                transformation_info[col]['new_skew'] = new_skew
                self.transformed_features.append(col)
            else:
                transformation_info[col] = {
                    'method': 'unchanged',
                    'original_skew': skew_val,
                    'new_skew': skew_val,
                }

        # Calculate transformed skewness
        for col in X_transformed.columns:
            if col not in categorical_cols:
                self.skewness_transformed[col] = stats.skew(X_transformed[col].dropna())

        print(f'Transformed features: {len(self.transformed_features)}')

        # Store transformation info only (function can be recreated from info)
        self.skewness_transformer = {
            'transformation_info': transformation_info,
            'skewness_threshold': self.skewness_threshold,
        }

        return X_transformed, transformation_info

    def define_feature_groups(self) -> dict[str, list[str]]:
        """
        Define feature groups based on domain knowledge

        Returns:
            Dictionary of feature groups
        """
        print('\nDefining feature groups...')

        # Group 1: Natural & Spatial Heterogeneity (69 variables)
        group1_natural = []

        # RFP Spatial Features (45)
        rfp_pollutants = [
            'THIA',
            'IMI',
            'CLO',
            'parentNNIs',
            'IMI-UREA',
            'DN-IMI',
            'CLO-UREA',
            'mNNIs',
        ]
        for pollutant in rfp_pollutants:
            for i in range(1, 6):
                group1_natural.append(f'Soil_{pollutant}_{i}')

        for i in range(1, 6):
            group1_natural.append(f'Soil_Dist_{i}')

        # Meteorological (5)
        group1_natural.extend(['PREC', 'T_M', 'T_W', 'Soil_Rain_wMean', 'Soil_Temp_wMean'])

        # Physicochemical (9)
        group1_natural.extend(
            [
                'DO',
                'pH',
                'COND',
                'DOC',
                'Soil_pH_wMean',
                'Soil_TN_wMean',
                'Soil_TOC_wMean',
                'Soil_CC_wMean',
                'Soil_BD_wMean',
            ]
        )

        # Vegetation & Topography (7 - including LST)
        group1_natural.extend(
            [
                'Alt',
                'Soil_Alt_wMean',
                'Soil_NDVI_wMean',
                'Soil_EVI_wMean',
                'Soil_LAI_wMean',
                'Soil_FCOVER_wMean',
                'Soil_LST_wMean',
            ]
        )

        # Categorical (3)
        categorical_vars = ['Season', 'Landuse', 'Soil_landuse']

        # Group 2: Agricultural Intensity (32 variables)
        group2_agro = []
        # Inputs (10)
        group2_agro.extend(
            [
                'FER',
                'Soil_FER_wMean',
                'FERPER',
                'Soil_FERA_wMean',
                'PES',
                'Soil_PES_wMean',
                'PESPER',
                'Soil_PESA_wMean',
                'AMP',
                'Soil_GPAM_wMean',
            ]
        )
        # Crop Area (7)
        group2_agro.extend(
            [
                'TSA',
                'FCA',
                'WA',
                'Soil_WS_wMean',
                'CA',
                'VEGA',
                'ARCA',
            ]
        )
        # Yields (11)
        group2_agro.extend(
            [
                'CROPOUT',
                'WO',
                'Soil_MC_wMean',
                'CO',
                'VO',
                'Soil_VE_wMean',
                'FOP',
                'Soil_FR_wMean',
                'Soil_MCP_wMean',
                'Soil_VEP_wMean',
                'Soil_FRP_wMean',
            ]
        )
        # Water (4)
        group2_agro.extend(['AGR_W', 'Soil_AGR_W_wMean', 'IRR_W', 'Soil_IRR_W_wMean'])
        # Remove duplicate CROPOUT if any and ensure correct count
        group2_agro = list(dict.fromkeys(group2_agro))  # Remove duplicates while preserving order

        # Group 3: Socio-Economic (29 variables)
        group3_socio = []

        # Macro Economy (8)
        group3_socio.extend(
            [
                'GDP',
                'Soil_GDP per capita_wMean',
                'OP_FI',
                'OP_SE',
                'OP_TH',
                'AO',
                'FO',
                'Soil_GAO_wMean',
            ]
        )
        # Urbanization (3)
        group3_socio.extend(
            [
                'Urban',
                'Soil_UR_wMean',
                'POP_TOT',
            ]
        )
        # Structure & Income (10)
        group3_socio.extend(
            [
                'PR',
                'Soil_PR_wMean',
                'SR',
                'Soil_SR_wMean',
                'TR',
                'Soil_TR_wMean',
                'UI',
                'Soil_UI_wMean',
                'RI',
                'Soil_RI_wMean',
            ]
        )
        # Non-Ag Water (8)
        group3_socio.extend(
            [
                'UR_W',
                'Soil_UR_W_wMean',
                'RU_W',
                'Soil_RU_W_wMean',
                'IND_W',
                'Soil_IND_W_wMean',
                'LIF_W',
                'Soil_LIF_W_wMean',
            ]
        )

        self.feature_groups = {
            'group1_natural': group1_natural,
            'group2_agro': group2_agro,
            'group3_socio': group3_socio,
            'categorical': categorical_vars,
        }

        print(f'Group 1 (Natural & Spatial): {len(group1_natural)} variables')
        print(f'Group 2 (Agricultural): {len(group2_agro)} variables')
        print(f'Group 3 (Socio-Economic): {len(group3_socio)} variables')
        print(f'Categorical: {len(categorical_vars)} variables')

        # Verify all features are accounted for
        all_grouped = set(group1_natural + group2_agro + group3_socio + categorical_vars)
        all_features = set(self.X.columns)  # type: ignore

        if all_grouped != all_features:
            missing = all_features - all_grouped
            extra = all_grouped - all_features
            if missing:
                print(f'WARNING: Missing features: {missing}')
            if extra:
                print(f'WARNING: Extra features: {extra}')
        else:
            print('✓ All features accounted for')

        return self.feature_groups

    def encode_categorical(
        self, X: pd.DataFrame, cat_cols: list[str]
    ) -> tuple[pd.DataFrame, OneHotEncoder]:
        """
        Apply one-hot encoding to categorical variables

        Args:
            X: Feature dataframe
            cat_cols: List of categorical column names

        Returns:
            X_encoded: Dataframe with encoded categorical variables
            encoder: Fitted OneHotEncoder
        """
        print(f'\nEncoding categorical variables: {cat_cols}')

        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_array = encoder.fit_transform(X[cat_cols])

        # Create column names
        encoded_cols = []
        for i, cat_col in enumerate(cat_cols):
            categories = encoder.categories_[i][1:]  # Skip first (dropped)
            for cat in categories:
                encoded_cols.append(f'{cat_col}_{cat}')

        # Create dataframe
        X_encoded = pd.DataFrame(
            encoded_array,
            columns=encoded_cols,  # type: ignore
            index=X.index,
        )

        print(f'Encoded features: {X_encoded.shape[1]} columns')

        self.categorical_encoder = encoder

        return X_encoded, encoder

    def process_group1_natural(self, X_group1: pd.DataFrame) -> pd.DataFrame:
        """
        Process Group 1: Standardization only (no PCA)

        Args:
            X_group1: Group 1 features

        Returns:
            Standardized features
        """
        print('\n' + '=' * 60)
        print('Processing Group 1: Natural & Spatial Heterogeneity')
        print('=' * 60)
        print(f'Input shape: {X_group1.shape}')

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_group1)
        X_scaled = pd.DataFrame(X_scaled, columns=X_group1.columns, index=X_group1.index)

        self.group1_scaler = scaler

        print(f'Output shape: {X_scaled.shape}')
        print('Applied: Z-score standardization')

        return X_scaled

    def process_group2_agro_pca(
        self, X_group2: pd.DataFrame
    ) -> tuple[pd.DataFrame, PCA, StandardScaler]:
        """
        Process Group 2: Standardization + PCA (95% variance)

        Args:
            X_group2: Group 2 features

        Returns:
            X_pca: PCA-transformed features
            pca: Fitted PCA model
            scaler: Fitted StandardScaler
        """
        print('\n' + '=' * 60)
        print('Processing Group 2: Agricultural Intensity')
        print('=' * 60)
        print(f'Input shape: {X_group2.shape}')

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_group2)

        # Apply PCA with 95% variance threshold
        pca = PCA(n_components=self.variance_threshold, random_state=42)
        X_pca_array = pca.fit_transform(X_scaled)

        # Create column names
        n_components = X_pca_array.shape[1]
        pca_cols = [f'PC_Agro_{i + 1}' for i in range(n_components)]

        X_pca = pd.DataFrame(
            X_pca_array,
            columns=pca_cols,  # type: ignore
            index=X_group2.index,
        )

        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)

        print(f'Output shape: {X_pca.shape}')
        print(f'Number of components: {n_components}')
        print(f'Explained variance: {explained_var[:5].round(4)} ...')
        print(f'Cumulative variance: {cumsum_var[-1]:.4f}')

        self.group2_scaler = scaler
        self.group2_pca = pca

        return X_pca, pca, scaler

    def process_group3_socio_pca(
        self, X_group3: pd.DataFrame
    ) -> tuple[pd.DataFrame, PCA, StandardScaler]:
        """
        Process Group 3: Standardization + PCA (95% variance)

        Args:
            X_group3: Group 3 features

        Returns:
            X_pca: PCA-transformed features
            pca: Fitted PCA model
            scaler: Fitted StandardScaler
        """
        print('\n' + '=' * 60)
        print('Processing Group 3: Socio-Economic Development')
        print('=' * 60)
        print(f'Input shape: {X_group3.shape}')

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_group3)

        # Apply PCA with 95% variance threshold
        pca = PCA(n_components=self.variance_threshold, random_state=42)
        X_pca_array = pca.fit_transform(X_scaled)

        # Create column names
        n_components = X_pca_array.shape[1]
        pca_cols = [f'PC_Socio_{i + 1}' for i in range(n_components)]

        X_pca = pd.DataFrame(
            X_pca_array,
            columns=pca_cols,  # type: ignore
            index=X_group3.index,
        )

        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)

        print(f'Output shape: {X_pca.shape}')
        print(f'Number of components: {n_components}')
        print(f'Explained variance: {explained_var[:5].round(4)} ...')
        print(f'Cumulative variance: {cumsum_var[-1]:.4f}')

        self.group3_scaler = scaler
        self.group3_pca = pca

        return X_pca, pca, scaler

    def combine_all_features(
        self,
        X_group1: pd.DataFrame,
        X_categorical: pd.DataFrame,
        X_group2_pca: pd.DataFrame,
        X_group3_pca: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine all processed feature groups

        Args:
            X_group1: Processed Group 1 features
            X_categorical: One-hot encoded categorical features
            X_group2_pca: PCA features from Group 2
            X_group3_pca: PCA features from Group 3

        Returns:
            Combined feature dataframe
        """
        print('\n' + '=' * 60)
        print('Combining All Feature Groups')
        print('=' * 60)

        # Concatenate all groups
        X_final = pd.concat([X_group1, X_categorical, X_group2_pca, X_group3_pca], axis=1)

        print(f'Group 1 (Natural): {X_group1.shape[1]} features')
        print(f'Categorical (One-hot): {X_categorical.shape[1]} features')
        print(f'Group 2 (PC_Agro): {X_group2_pca.shape[1]} features')
        print(f'Group 3 (PC_Socio): {X_group3_pca.shape[1]} features')
        print(f'Total final features: {X_final.shape[1]}')

        self.X_final = X_final

        return X_final

    def save_transformers(self, output_dir: str):
        """
        Save all transformer objects

        Args:
            output_dir: Directory to save transformers
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print('\n' + '=' * 60)
        print(f'Saving Transformers to {output_dir}')
        print('=' * 60)

        # Save each transformer
        transformers = {
            'skewness_transformer.pkl': self.skewness_transformer,
            'categorical_encoder.pkl': self.categorical_encoder,
            'group1_scaler.pkl': self.group1_scaler,
            'group2_scaler.pkl': self.group2_scaler,
            'group2_pca.pkl': self.group2_pca,
            'group3_scaler.pkl': self.group3_scaler,
            'group3_pca.pkl': self.group3_pca,
        }

        for filename, transformer in transformers.items():
            filepath = output_path / filename
            with open(filepath, 'wb') as f:
                pickle.dump(transformer, f)
            print(f'✓ Saved: {filename}')

        # Save feature group definitions
        filepath = output_path / 'feature_groups.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(self.feature_groups, f)
        print('✓ Saved: feature_groups.pkl')

    def save_processed_data(self, output_path: str):
        """
        Save processed dataset combining features, targets, and metadata

        Args:
            output_path: Path to save processed_data.csv
        """
        print('\n' + '=' * 60)
        print('Saving Processed Data')
        print('=' * 60)

        # Combine metadata, targets, and features
        processed_df = pd.concat(
            [  # type: ignore
                self.metadata,
                self.y,
                self.X_final,
            ],
            axis=1,
        )

        # Save
        processed_df.to_csv(output_path, index=False)

        print(f'✓ Saved: {output_path}')
        print(f'  Shape: {processed_df.shape}')
        print(f'  Columns: {processed_df.shape[1]}')
        print(f'  Rows: {processed_df.shape[0]}')

    def generate_report(self, output_dir: str):
        """
        Generate comprehensive preprocessing report with statistics and visualizations

        Args:
            output_dir: Directory to save report and plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print('\n' + '=' * 60)
        print('Generating Preprocessing Report')
        print('=' * 60)

        report_file = output_path / 'preprocessing_report.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('Feature Scaling and Grouped PCA Preprocessing Report\n')
            f.write('=' * 80 + '\n\n')

            # 1. Skewness Statistics
            f.write('1. SKEWNESS DIAGNOSIS AND CORRECTION\n')
            f.write('-' * 80 + '\n\n')

            f.write(f'Skewness threshold: {self.skewness_threshold}\n')
            f.write(f'Transformation: Log(x+1) for |skew| > {self.skewness_threshold}\n\n')

            # Count transformations
            transform_info = self.skewness_transformer['transformation_info']  # type: ignore
            transformed_count = sum(
                1 for v in transform_info.values() if v['method'] != 'unchanged'
            )

            f.write(f'Total continuous features: {len(transform_info)}\n')
            f.write(f'Transformed features: {transformed_count}\n')
            f.write(f'Unchanged features: {len(transform_info) - transformed_count}\n\n')

            # Top 10 most skewed features (before transformation)
            sorted_skew = sorted(
                self.skewness_original.items(), key=lambda x: abs(x[1]), reverse=True
            )
            f.write('Top 10 Most Skewed Features (Original):\n')
            f.write(f'{"Feature":<35} {"Original Skew":>15} {"New Skew":>15} {"Method":<20}\n')
            f.write('-' * 85 + '\n')
            for feat, _ in sorted_skew[:10]:
                info = transform_info[feat]
                f.write(
                    f'{feat:<35} {info["original_skew"]:>15.4f} {info["new_skew"]:>15.4f} {info["method"]:<20}\n'
                )
            f.write('\n')

            # 2. Feature Groups
            f.write('2. FEATURE GROUPING\n')
            f.write('-' * 80 + '\n\n')

            f.write(
                f'Group 1 (Natural & Spatial): {len(self.feature_groups["group1_natural"])} features\n'
            )
            f.write('  - Processing: Standardization only (no PCA)\n\n')

            f.write(
                f'Group 2 (Agricultural Intensity): {len(self.feature_groups["group2_agro"])} features\n'
            )
            f.write('  - Processing: Standardization + PCA (95% variance)\n')
            f.write(f'  - Output: {self.group2_pca.n_components_} principal components\n\n')  # type: ignore

            f.write(
                f'Group 3 (Socio-Economic): {len(self.feature_groups["group3_socio"])} features\n'
            )
            f.write('  - Processing: Standardization + PCA (95% variance)\n')
            f.write(f'  - Output: {self.group3_pca.n_components_} principal components\n\n')  # type: ignore

            f.write(f'Categorical: {len(self.feature_groups["categorical"])} features\n')
            f.write('  - Processing: One-hot encoding (drop first)\n')
            f.write(
                f'  - Output: {len([c for c in self.X_final.columns if any(cat in c for cat in self.feature_groups["categorical"])])} features\n\n'  # type: ignore
            )

            # 3. PCA Details - Group 2
            f.write('3. PCA ANALYSIS - GROUP 2 (AGRICULTURAL INTENSITY)\n')
            f.write('-' * 80 + '\n\n')

            explained_var2 = self.group2_pca.explained_variance_ratio_  # type: ignore
            cumsum_var2 = np.cumsum(explained_var2)

            f.write(f'Number of components: {self.group2_pca.n_components_}\n')  # type: ignore
            f.write(f'Total variance explained: {cumsum_var2[-1]:.4f}\n\n')

            f.write(f'{"Component":<15} {"Variance":>15} {"Cumulative":>15}\n')
            f.write('-' * 45 + '\n')
            for i, (var, cumvar) in enumerate(zip(explained_var2, cumsum_var2, strict=False)):
                f.write(f'PC_Agro_{i + 1:<8} {var:>15.4f} {cumvar:>15.4f}\n')

            # Top feature loadings for first 3 PCs
            f.write('\nTop 5 Feature Loadings (Absolute Value):\n')
            f.write('-' * 80 + '\n')

            components2 = self.group2_pca.components_  # type: ignore
            feature_names2 = self.feature_groups['group2_agro']

            for i in range(min(3, components2.shape[0])):
                f.write(f'\nPC_Agro_{i + 1}:\n')
                loadings = pd.Series(components2[i], index=feature_names2)
                top_loadings = loadings.abs().nlargest(5)
                for feat, _ in top_loadings.items():
                    actual_loading = loadings[feat]
                    f.write(f'  {feat:<30} {actual_loading:>10.4f}\n')

            f.write('\n')

            # 4. PCA Details - Group 3
            f.write('4. PCA ANALYSIS - GROUP 3 (SOCIO-ECONOMIC)\n')
            f.write('-' * 80 + '\n\n')

            explained_var3 = self.group3_pca.explained_variance_ratio_  # type: ignore
            cumsum_var3 = np.cumsum(explained_var3)

            f.write(f'Number of components: {self.group3_pca.n_components_}\n')  # type: ignore
            f.write(f'Total variance explained: {cumsum_var3[-1]:.4f}\n\n')

            f.write(f'{"Component":<15} {"Variance":>15} {"Cumulative":>15}\n')
            f.write('-' * 45 + '\n')
            for i, (var, cumvar) in enumerate(zip(explained_var3, cumsum_var3, strict=False)):
                f.write(f'PC_Socio_{i + 1:<8} {var:>15.4f} {cumvar:>15.4f}\n')

            # Top feature loadings for first 3 PCs
            f.write('\nTop 5 Feature Loadings (Absolute Value):\n')
            f.write('-' * 80 + '\n')

            components3 = self.group3_pca.components_  # type: ignore
            feature_names3 = self.feature_groups['group3_socio']

            for i in range(min(3, components3.shape[0])):
                f.write(f'\nPC_Socio_{i + 1}:\n')
                loadings = pd.Series(components3[i], index=feature_names3)
                top_loadings = loadings.abs().nlargest(5)
                for feat, _ in top_loadings.items():
                    actual_loading = loadings[feat]
                    f.write(f'  {feat:<30} {actual_loading:>10.4f}\n')

            f.write('\n')

            # 5. Final Dataset Summary
            f.write('5. FINAL PROCESSED DATASET SUMMARY\n')
            f.write('-' * 80 + '\n\n')

            f.write(f'Total samples: {self.X_final.shape[0]}\n')  # type: ignore
            f.write(f'Total features: {self.X_final.shape[1]}\n')  # type: ignore
            f.write(f'Target variables: {self.y.shape[1]}\n')  # type: ignore
            f.write(f'Metadata columns: {self.metadata.shape[1]}\n\n')  # type: ignore

            f.write('Feature Breakdown:\n')
            group1_cols = [
                c
                for c in self.X_final.columns  # type: ignore
                if c in self.feature_groups['group1_natural']
            ]
            cat_cols = [
                c
                for c in self.X_final.columns  # type: ignore
                if any(cat in c for cat in self.feature_groups['categorical'])
            ]
            agro_cols = [c for c in self.X_final.columns if c.startswith('PC_Agro_')]  # type: ignore
            socio_cols = [c for c in self.X_final.columns if c.startswith('PC_Socio_')]  # type: ignore

            f.write(f'  - Group 1 (Natural & Spatial): {len(group1_cols)} features\n')
            f.write(f'  - Categorical (One-hot): {len(cat_cols)} features\n')
            f.write(f'  - Group 2 (PC_Agro): {len(agro_cols)} features\n')
            f.write(f'  - Group 3 (PC_Socio): {len(socio_cols)} features\n\n')

            f.write('=' * 80 + '\n')
            f.write('END OF REPORT\n')
            f.write('=' * 80 + '\n')

        print('✓ Saved: preprocessing_report.txt')

        # Generate visualizations
        self._generate_visualizations(output_path)

    def _generate_visualizations(self, output_path: Path):
        """
        Generate visualization plots

        Args:
            output_path: Directory to save plots
        """
        print('\nGenerating visualizations...')

        # 1. Skewness comparison
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Original skewness
        skew_original = pd.Series(self.skewness_original).sort_values(key=abs, ascending=False)[:20]
        axes[0].barh(range(len(skew_original)), skew_original.values)  # type: ignore
        axes[0].set_yticks(range(len(skew_original)))
        axes[0].set_yticklabels(skew_original.index, fontsize=8)  # type: ignore
        axes[0].axvline(
            x=self.skewness_threshold,
            color='r',
            linestyle='--',
            label=f'Threshold: {self.skewness_threshold}',
        )
        axes[0].axvline(x=-self.skewness_threshold, color='r', linestyle='--')
        axes[0].set_xlabel('Skewness')
        axes[0].set_title('Top 20 Features - Original Skewness')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Transformed skewness (same features)
        skew_transformed = pd.Series({k: self.skewness_transformed[k] for k in skew_original.index})  # type: ignore
        axes[1].barh(range(len(skew_transformed)), skew_transformed.values, color='green')
        axes[1].set_yticks(range(len(skew_transformed)))
        axes[1].set_yticklabels(skew_transformed.index, fontsize=8)
        axes[1].axvline(
            x=self.skewness_threshold,
            color='r',
            linestyle='--',
            label=f'Threshold: {self.skewness_threshold}',
        )
        axes[1].axvline(x=-self.skewness_threshold, color='r', linestyle='--')
        axes[1].set_xlabel('Skewness')
        axes[1].set_title('Top 20 Features - After Transformation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'skewness_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: skewness_comparison.png')

        # 2. PCA Scree plot - Group 2
        fig, ax = plt.subplots(figsize=(10, 6))

        explained_var2 = self.group2_pca.explained_variance_ratio_  # type: ignore
        cumsum_var2 = np.cumsum(explained_var2)
        x = range(1, len(explained_var2) + 1)

        ax.bar(x, explained_var2, alpha=0.6, label='Individual Variance')
        ax.plot(x, cumsum_var2, 'ro-', label='Cumulative Variance')
        ax.axhline(y=0.95, color='g', linestyle='--', label='95% Threshold')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Scree Plot - Group 2 (Agricultural Intensity)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'pca_scree_plot_agro.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: pca_scree_plot_agro.png')

        # 3. PCA Scree plot - Group 3
        fig, ax = plt.subplots(figsize=(10, 6))

        explained_var3 = self.group3_pca.explained_variance_ratio_  # type: ignore
        cumsum_var3 = np.cumsum(explained_var3)
        x = range(1, len(explained_var3) + 1)

        ax.bar(x, explained_var3, alpha=0.6, color='orange', label='Individual Variance')
        ax.plot(x, cumsum_var3, 'ro-', label='Cumulative Variance')
        ax.axhline(y=0.95, color='g', linestyle='--', label='95% Threshold')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Scree Plot - Group 3 (Socio-Economic)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'pca_scree_plot_socio.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Saved: pca_scree_plot_socio.png')


def main():
    """
    Main execution function
    """
    # Define paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'datasets' / 'merged_data.csv'
    output_data_path = project_root / 'datasets' / 'processed_data.csv'
    transformers_dir = project_root / 'models' / 'transformers'
    reports_dir = project_root / 'reports'

    print('=' * 80)
    print('Feature Scaling and Grouped PCA Preprocessing Pipeline')
    print('=' * 80)

    # Initialize preprocessor
    preprocessor = FeaturePreprocessor(variance_threshold=0.95, skewness_threshold=0.75)

    # Step 1: Load data
    preprocessor.load_data(str(input_path))

    # Step 2: Separate features, targets, metadata
    X, y, metadata = preprocessor.separate_features_targets()

    # Step 3: Diagnose and correct skewness
    preprocessor.diagnose_skewness(X)
    X_transformed, transform_info = preprocessor.apply_skewness_correction(X)

    # Step 4: Define feature groups
    feature_groups = preprocessor.define_feature_groups()

    # Step 5: Extract categorical features and encode
    cat_cols = feature_groups['categorical']
    X_categorical, encoder = preprocessor.encode_categorical(X_transformed, cat_cols)

    # Step 6: Extract numerical features for each group
    X_group1 = X_transformed[feature_groups['group1_natural']]
    X_group2 = X_transformed[feature_groups['group2_agro']]
    X_group3 = X_transformed[feature_groups['group3_socio']]

    # Step 7: Process each group
    X_group1_processed = preprocessor.process_group1_natural(X_group1)  # type: ignore
    X_group2_pca, pca2, scaler2 = preprocessor.process_group2_agro_pca(X_group2)  # type: ignore
    X_group3_pca, pca3, scaler3 = preprocessor.process_group3_socio_pca(X_group3)  # type: ignore

    # Step 8: Combine all features
    X_final = preprocessor.combine_all_features(
        X_group1_processed, X_categorical, X_group2_pca, X_group3_pca
    )

    # Step 9: Save transformers
    preprocessor.save_transformers(str(transformers_dir))

    # Step 10: Save processed data
    preprocessor.save_processed_data(str(output_data_path))

    # Step 11: Generate report
    preprocessor.generate_report(str(reports_dir))

    print('\n' + '=' * 80)
    print('PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!')
    print('=' * 80)
    print('\nOutputs:')
    print(f'  - Processed data: {output_data_path}')
    print(f'  - Transformers: {transformers_dir}/')
    print(f'  - Report: {reports_dir}/preprocessing_report.txt')
    print(f'  - Visualizations: {reports_dir}/')


if __name__ == '__main__':
    main()
