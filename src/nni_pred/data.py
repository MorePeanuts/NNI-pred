"""
This script contains code related to data features and data loading.
"""

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from dataclasses import dataclass


SOIL_POLLUTANTS = [
    'THIA',
    'IMI',
    'CLO',
    'parentNNIs',
    'IMI-UREA',
    'DN-IMI',
    'CLO-UREA',
    'mNNIs',
]
SOIL_SEASONAL_VARS = ['pH', 'TN', 'TOC', 'Temp', 'Rain', 'EVI', 'FCOVER', 'LAI', 'LST', 'NDVI']
SOIL_ANNUAL_VARS = [
    'Alt',
    'CC',
    'BD',
    'GPAM',
    'WS',
    'FER',
    'PES',
    'MC',
    'MCP',
    'VE',
    'VEP',
    'FR',
    'FRP',
    'FERA',
    'PESA',
    'UR',
    'GAO',
    'PR',
    'SR',
    'TR',
    'GDP per capita',
    'UI',
    'RI',
    'UR_W',
    'RU_W',
    'IRR_W',
    'AGR_W',
    'IND_W',
    'LIF_W',
]
SOIL_CATEGORICAL_VARS = ['landuse']


@dataclass
class FeatureGroups:
    group1_natural: list[str]
    group2_agro: list[str]
    group3_socio: list[str]
    categorical: list[str]
    targets: list[str]
    metadata: list[str]


def get_feature_groups() -> FeatureGroups:
    """
    Define feature groups based on domain knowledge.

    Returns:
        Dataclass with fields:
        - 'group1_natural': List of natural/spatial feature names
        - 'group2_agro': List of agricultural feature names
        - 'group3_socio': List of socio-economic feature names
        - 'categorical': List of categorical variable names
        - 'targets': List of target pollutant names
        - 'metadata': List of metadata column names
    """

    # Group 1: Natural & Spatial Heterogeneity Features
    # These are direct drivers of pollutant migration

    # Meteorological factors (5 features)
    meteorological = [
        'PREC',  # Water body precipitation
        'T_M',  # Air temperature
        'T_W',  # Water temperature
        'Soil_Rain_agg',  # Surrounding soil rainfall (IDW aggregated)
        'Soil_Temp_agg',  # Surrounding soil temperature (IDW aggregated)
    ]

    # Physicochemical properties (9 features)
    physicochemical = [
        # Water body properties
        'DO',  # Dissolved oxygen
        'pH',  # pH
        'COND',  # Conductivity
        'DOC',  # Dissolved organic carbon
        # Soil properties (IDW aggregated)
        'Soil_pH_agg',  # Soil pH
        'Soil_TN_agg',  # Total nitrogen
        'Soil_TOC_agg',  # Total organic carbon
        'Soil_CC_agg',  # Clay content
        'Soil_BD_agg',  # Bulk density
    ]

    # Vegetation & Topography (7 features)
    vegetation_topography = [
        'Alt',  # Altitude (water body)
        'Soil_Alt_agg',  # Altitude (surrounding soil, IDW aggregated)
        'Soil_NDVI_agg',  # Normalized Difference Vegetation Index
        'Soil_EVI_agg',  # Enhanced Vegetation Index
        'Soil_LAI_agg',  # Leaf Area Index
        'Soil_FCOVER_agg',  # Fractional vegetation cover
        'Soil_LST_agg',  # Land Surface Temperature
    ]

    # Soil pollutant concentrations (IDW aggregated, 8 pollutants)
    soil_pollutants = [
        'Soil_THIA_agg',
        'Soil_IMI_agg',
        'Soil_CLO_agg',
        'Soil_parentNNIs_agg',
        'Soil_IMI-UREA_agg',
        'Soil_DN-IMI_agg',
        'Soil_CLO-UREA_agg',
        'Soil_mNNIs_agg',
    ]

    group1_natural = meteorological + physicochemical + vegetation_topography + soil_pollutants
    # Total: 5 + 9 + 7 + 8 = 29 features

    # Group 2: Agricultural Intensity Features (32 features)
    # Reflects fertilizer/pesticide input and agricultural production scale

    # Agricultural inputs (10 features)
    agro_inputs = [
        'FER',  # Fertilizer usage (water body location)
        'Soil_FER_agg',  # Fertilizer usage (surrounding soil, IDW)
        'FERPER',  # Fertilizer per unit area
        'Soil_FERA_agg',  # Fertilizer per unit area (soil, IDW)
        'PES',  # Pesticide usage
        'Soil_PES_agg',  # Pesticide usage (soil, IDW)
        'PESPER',  # Pesticide per unit area
        'Soil_PESA_agg',  # Pesticide per unit area (soil, IDW)
        'AMP',  # Agricultural machinery power
        'Soil_GPAM_agg',  # Agricultural machinery power (soil, IDW)
    ]

    # Crop planting area (7 features)
    crop_area = [
        'TSA',  # Total sown area
        'FCA',  # Food crop area
        'WA',  # Wheat area
        'Soil_WS_agg',  # Wheat sowing area (soil, IDW)
        'CA',  # Corn area
        'VEGA',  # Vegetable area
        'ARCA',  # (Additional crop area - check research plan)
    ]

    # Agricultural yields (11 features)
    crop_yields = [
        'CROPOUT',  # Grain crop output
        'WO',  # Wheat output
        'Soil_MC_agg',  # Wheat output (soil, IDW)
        'CO',  # Corn output
        'VO',  # Vegetable output
        'Soil_VE_agg',  # Vegetable output (soil, IDW)
        'FOP',  # Fruit output
        'Soil_FR_agg',  # Fruit output (soil, IDW)
        'Soil_MCP_agg',  # Wheat yield per unit area (soil, IDW)
        'Soil_VEP_agg',  # Vegetable yield per unit area (soil, IDW)
        'Soil_FRP_agg',  # Fruit yield per unit area (soil, IDW)
    ]

    # Agricultural water usage (4 features)
    agro_water = [
        'AGR_W',  # Agricultural water usage
        'Soil_AGR_W_agg',  # Agricultural water usage (soil, IDW)
        'IRR_W',  # Irrigation water usage
        'Soil_IRR_W_agg',  # Irrigation water usage (soil, IDW)
    ]

    group2_agro = agro_inputs + crop_area + crop_yields + agro_water
    # Total: 10 + 7 + 11 + 4 = 32 features

    # Group 3: Socio-Economic Development Features (29 features)
    # Reflects urbanization, industrialization, and population aggregation

    # Macro economy (8 features)
    macro_economy = [
        'GDP',  # Gross Domestic Product
        'Soil_GDP per capita_agg',  # GDP per capita (soil, IDW)
        'OP_FI',  # First industry output
        'OP_SE',  # Second industry output
        'OP_TH',  # Third industry output
        'AO',  # Agricultural output
        'FO',  # Forestry output
        'Soil_GAO_agg',  # Agricultural output (soil, IDW)
    ]

    # Urbanization & population (3 features)
    urbanization = [
        'Urban',  # Urbanization rate
        'Soil_UR_agg',  # Urbanization rate (soil, IDW)
        'POP_TOT',  # Total population
    ]

    # Industrial structure & income (10 features)
    structure_income = [
        'PR',  # Primary industry ratio
        'Soil_PR_agg',  # Primary industry ratio (soil, IDW)
        'SR',  # Secondary industry ratio
        'Soil_SR_agg',  # Secondary industry ratio (soil, IDW)
        'TR',  # Tertiary industry ratio
        'Soil_TR_agg',  # Tertiary industry ratio (soil, IDW)
        'UI',  # Urban resident income
        'Soil_UI_agg',  # Urban resident income (soil, IDW)
        'RI',  # Rural resident income
        'Soil_RI_agg',  # Rural resident income (soil, IDW)
    ]

    # Non-agricultural water usage (8 features)
    non_agro_water = [
        'UR_W',  # Urban residential water usage
        'Soil_UR_W_agg',  # Urban residential water usage (soil, IDW)
        'RU_W',  # Rural residential water usage
        'Soil_RU_W_agg',  # Rural residential water usage (soil, IDW)
        'IND_W',  # Industrial water usage
        'Soil_IND_W_agg',  # Industrial water usage (soil, IDW)
        'LIF_W',  # Total residential water usage
        'Soil_LIF_W_agg',  # Total residential water usage (soil, IDW)
    ]

    group3_socio = macro_economy + urbanization + structure_income + non_agro_water
    # Total: 8 + 3 + 10 + 8 = 29 features

    # Categorical variables (3 variables)
    # These will be one-hot encoded
    categorical = [
        'Season',  # Season (Dry, Normal, Rainy)
        'Landuse',  # Land use type at water body location
        'Soil_landuse',  # Land use type of nearest soil sample
    ]

    # Target variables (11 pollutants)
    # Water body pollutant concentrations (log1p transformed in preprocessing)
    targets = [
        # Parent compounds
        'THIA',  # Thiamethoxam
        'IMI',  # Imidacloprid
        'CLO',  # Clothianidin
        'ACE',  # Acetamiprid
        'DIN',  # Dinotefuran
        'parentNNIs',  # Sum of parent neonicotinoids
        # Metabolites
        'IMI-UREA',  # Imidacloprid-urea
        'DN-IMI',  # Desmethyl-imidacloprid
        'DM-ACE',  # Desmethyl-acetamiprid
        'CLO-UREA',  # Clothianidin-urea
        'mNNIs',  # Sum of metabolites
    ]

    # Metadata (not used as model inputs)
    metadata = [
        'ID',  # Sample ID
        'Lon',  # Longitude
        'Lat',  # Latitude
    ]

    return FeatureGroups(group1_natural, group2_agro, group3_socio, categorical, targets, metadata)


class MergedTabularDataset:
    def __init__(self, data_path: Path | None = None):
        if data_path is None:
            data_path = Path(__file__).parents[2] / 'datasets/merged_data.csv'
        assert data_path.exists(), f'data path {data_path} doesnot exists.'
        self.df = pd.read_csv(data_path, index_col='ID')
        self.groups = self._create_groups()
        self.feature_groups = get_feature_groups()
        self._validate_features()
        self._apply_one_hot_encoding()

    def _create_groups(self, lon_col: str = 'Lon', lat_col: str = 'Lat'):
        """
        Generate group IDs based on spatial location.

        Args:
            lon_col: Name of longitude column (default: 'Lon')
            lat_col: Name of latitude column (default: 'Lat')

        Returns:
            np.ndarray of shape (n_samples,) with group IDs (0 to n_locations-1)
        """
        # Extract unique locations
        locations = self.df[[lon_col, lat_col]].drop_duplicates().reset_index(drop=True)

        # Create mapping: (lon, lat) -> group_id
        location_to_group = {(row[lon_col], row[lat_col]): idx for idx, row in locations.iterrows()}

        # Assign group to each sample
        groups = self.df.apply(
            lambda row: location_to_group[(row[lon_col], row[lat_col])], axis=1
        ).values

        return groups

    def prepare_data(self) -> tuple[pd.DataFrame, dict[str, pd.Series], np.ndarray]:
        """
        Prepare features, targets, and spatial groups.

        Args:
            df: Loaded dataframe

        Returns:
            Tuple of (X, y_dict, groups)
        """
        # Extract targets
        assert isinstance(self.feature_groups, FeatureGroups)
        target_cols = self.feature_groups.targets
        y_dict = {col: self.df[col] for col in target_cols if col in self.df.columns}

        # Extract metadata columns
        metadata_cols = self.feature_groups.metadata

        # Features = all columns except targets and metadata
        exclude_cols = list(y_dict.keys()) + metadata_cols
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        X = self.df[feature_cols]

        return X, y_dict, self.groups

    def _validate_features(self):
        assert isinstance(self.feature_groups, FeatureGroups)
        assert isinstance(self.df, pd.DataFrame)

        for col in self.feature_groups.group1_natural:
            assert col in self.df.columns

        for col in self.feature_groups.group2_agro:
            assert col in self.df.columns

        for col in self.feature_groups.group3_socio:
            assert col in self.df.columns

        for col in self.feature_groups.targets:
            assert col in self.df.columns

        for col in self.feature_groups.metadata:
            assert col in self.df.columns or col == self.df.index.name

        for col in self.feature_groups.categorical:
            assert self.df.columns.str.startswith(col).any()

    def _apply_one_hot_encoding(self):
        """
        Apply one-hot encoding to categorical variables.
        """
        assert isinstance(self.feature_groups, FeatureGroups)
        categorical_vars = self.feature_groups.categorical

        for col in categorical_vars:
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=False, dtype=float)
            self.df = self.df.drop(columns=[col])
            self.df = pd.concat([self.df, dummies], axis=1)
