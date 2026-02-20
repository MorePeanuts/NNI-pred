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


class MergedVariableGroups:
    metadata = [
        'ID',  # Sample ID
        'Lon',  # Longitude
        'Lat',  # Latitude
    ]
    categorical = [
        'Season',  # Season (Dry, Normal, Rainy)
        'Landuse',  # Land use type at water body location
        # 'Soil_landuse',  # Land use type of nearest soil sample
    ]
    targets_parent = [
        'THIA',  # Thiamethoxam
        'IMI',  # Imidacloprid
        'CLO',  # Clothianidin
        'ACE',  # Acetamiprid
        'DIN',  # Dinotefuran
        'parentNNIs',  # Sum of parent neonicotinoids
    ]
    targets_metabolites = [
        'IMI-UREA',  # Imidacloprid-urea
        'DN-IMI',  # Desmethyl-imidacloprid
        'DM-ACE',  # Desmethyl-acetamiprid
        'CLO-UREA',  # Clothianidin-urea
        'mNNIs',  # Sum of metabolites
    ]
    soil_parent = [
        'Soil_THIA_agg',
        'Soil_IMI_agg',
        'Soil_CLO_agg',
        'Soil_parentNNIs_agg',
    ]
    soil_metabolites = [
        'Soil_IMI-UREA_agg',
        'Soil_DN-IMI_agg',
        'Soil_CLO-UREA_agg',
        'Soil_mNNIs_agg',
    ]
    group_natural = [
        'PREC',  # Water body precipitation
        'T_M',  # Air temperature
        'T_W',  # Water temperature
        'DO',  # Dissolved oxygen
        'pH',  # pH
        'COND',  # Conductivity
        'DOC',  # Dissolved organic carbon
        # WARNING: Whether to use altitude as a feature
        # 'Alt',  # Altitude (water body)
    ]
    group_agro = [
        'FER',  # Fertilizer usage (water body location)
        'FERPER',  # Fertilizer per unit area
        'PES',  # Pesticide usage
        'PESPER',  # Pesticide per unit area
        'AMP',  # Agricultural machinery power
        'TSA',  # Total sown area
        'FCA',  # Food crop area
        'WA',  # Wheat area
        'CA',  # Corn area
        'VEGA',  # Vegetable area
        'ARCA',  # Additional crop area
        'CROPOUT',  # Grain crop output
        'WO',  # Wheat output
        'CO',  # Corn output
        'VO',  # Vegetable output
        'FOP',  # Fruit output
        'AGR_W',  # Agricultural water usage
        'IRR_W',  # Irrigation water usage
    ]
    group_socio = [
        'GDP',  # Gross Domestic Product
        'OP_FI',  # First industry output
        'OP_SE',  # Second industry output
        'OP_TH',  # Third industry output
        'AO',  # Agricultural output
        'FO',  # Forestry output
        'Urban',  # Urbanization rate
        'POP_TOT',  # Total population
        'PR',  # Primary industry ratio
        'SR',  # Secondary industry ratio
        'TR',  # Tertiary industry ratio
        'UI',  # Urban resident income
        'RI',  # Rural resident income
        'UR_W',  # Urban residential water usage
        'RU_W',  # Rural residential water usage
        'IND_W',  # Industrial water usage
        'LIF_W',  # Total residential water usage
    ]

    @classmethod
    def get_feature_cols(cls):
        return (
            cls.categorical
            + cls.soil_parent
            + cls.soil_metabolites
            + cls.group_natural
            + cls.group_agro
            + cls.group_socio
        )

    @classmethod
    def get_numerical_feature_cols(cls):
        return (
            cls.soil_parent
            + cls.soil_metabolites
            + cls.group_natural
            + cls.group_agro
            + cls.group_socio
        )

    @classmethod
    def get_natural_feature_cols(cls):
        return cls.group_natural + cls.soil_parent + cls.soil_metabolites


class SoilVariableGroups:
    metadata = [
        'ID',  # Sample ID
        'Lon',  # Longitude
        'Lat',  # Latitude
    ]
    categorical = [
        'Season',  # Season (Dry, Normal, Rainy)
        'landuse',  # Land use type at water body location
    ]
    targets_parent = [
        'THIA',  # Thiamethoxam
        'IMI',  # Imidacloprid
        'CLO',  # Clothianidin
        'parentNNIs',  # Sum of parent neonicotinoids
    ]
    targets_metabolites = [
        'IMI-UREA',  # Imidacloprid-urea
        'DM-CLO',  # Desmethyl-clothianidin
        'DN-IMI',  # Desmethyl-imidacloprid
        'CLO-UREA',  # Clothianidin-urea
        'mNNIs',  # Sum of metabolites
    ]
    group_natural = [
        'pH',
        'TN',
        'TOC',
        'Temp',
        'Rain',
        'EVI',
        'FCOVER',
        'LAI',
        'LST',
        'NDVI',
        'CC',
        'BD',
    ]
    group_agro = [
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
    ]
    group_socio = [
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

    @classmethod
    def get_feature_cols(cls):
        return cls.categorical + cls.group_natural + cls.group_agro + cls.group_socio

    @classmethod
    def get_numerical_feature_cols(cls):
        return cls.group_natural + cls.group_agro + cls.group_socio

    @classmethod
    def get_natural_feature_cols(cls):
        return cls.group_natural


class SoilTabularDataset:
    def __init__(self, data_path: Path | None = None):
        if data_path is None:
            data_path = Path(__file__).parents[2] / 'datasets/soil_data.csv'
        assert data_path.exists(), f'data path {data_path} doesnot exists.'
        self.df = pd.read_csv(data_path, index_col='ID')
        self.groups = self._create_groups()

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
        # Extract targets
        target_cols = SoilVariableGroups.targets_parent + SoilVariableGroups.targets_metabolites
        y_dict = {col: self.df[col] for col in target_cols if col in self.df.columns}

        # Extract metadata columns
        metadata_cols = SoilVariableGroups.metadata

        # Features = all columns except targets and metadata
        exclude_cols = list(y_dict.keys()) + metadata_cols
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        X = self.df[feature_cols]

        return X, y_dict, self.groups

    def _validate_features(self):
        assert isinstance(self.df, pd.DataFrame)

        for col in SoilVariableGroups.categorical:
            assert col in self.df.columns

        for col in SoilVariableGroups.targets_parent:
            assert col in self.df.columns

        for col in SoilVariableGroups.targets_metabolites:
            assert col in self.df.columns

        for col in SoilVariableGroups.group_natural:
            assert col in self.df.columns

        for col in SoilVariableGroups.group_agro:
            assert col in self.df.columns

        for col in SoilVariableGroups.group_socio:
            assert col in self.df.columns


class MergedTabularDataset:
    def __init__(self, data_path: Path | None = None):
        if data_path is None:
            data_path = Path(__file__).parents[2] / 'datasets/merged_data.csv'
        assert data_path.exists(), f'data path {data_path} doesnot exists.'
        self.df = pd.read_csv(data_path, index_col='ID')
        self.groups = self._create_groups()
        self._validate_features()

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
        target_cols = MergedVariableGroups.targets_parent + MergedVariableGroups.targets_metabolites
        y_dict = {col: self.df[col] for col in target_cols if col in self.df.columns}

        # Extract metadata columns
        metadata_cols = MergedVariableGroups.metadata

        # Features = all columns except targets and metadata
        exclude_cols = list(y_dict.keys()) + metadata_cols
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        X = self.df[feature_cols]

        return X, y_dict, self.groups

    def _validate_features(self):
        assert isinstance(self.df, pd.DataFrame)

        for col in MergedVariableGroups.categorical:
            assert col in self.df.columns

        for col in MergedVariableGroups.targets_parent:
            assert col in self.df.columns

        for col in MergedVariableGroups.targets_metabolites:
            assert col in self.df.columns

        for col in MergedVariableGroups.soil_parent:
            assert col in self.df.columns

        for col in MergedVariableGroups.soil_metabolites:
            assert col in self.df.columns

        for col in MergedVariableGroups.group_natural:
            assert col in self.df.columns

        for col in MergedVariableGroups.group_agro:
            assert col in self.df.columns

        for col in MergedVariableGroups.group_socio:
            assert col in self.df.columns
