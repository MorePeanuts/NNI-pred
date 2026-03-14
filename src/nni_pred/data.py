"""
This script contains code related to data features and data loading.
"""

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from dataclasses import dataclass


class MergedVariableGroups:
    metadata = [
        'ID',  # Sample ID
        'Lon',  # Longitude
        'Lat',  # Latitude
        'parentNNIs',  # Sum of parent neonicotinoids
        'mNNIs',  # Sum of metabolites
        'Soil_parentNNIs_agg',
        'Soil_mNNIs_agg',
        'Alt',
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
    ]
    targets_metabolites = [
        'IMI-UREA',  # Imidacloprid-urea
        'DN-IMI',  # Desmethyl-imidacloprid
        'DM-ACE',  # Desmethyl-acetamiprid
        'CLO-UREA',  # Clothianidin-urea
    ]
    soil_parent = [
        'Soil_THIA_agg',
        'Soil_IMI_agg',
        'Soil_CLO_agg',
    ]
    soil_metabolites = [
        'Soil_IMI-UREA_agg',
        'Soil_DN-IMI_agg',
        'Soil_CLO-UREA_agg',
    ]
    group_natural = [
        'PREC',  # Water body precipitation
        'T_M',  # Air temperature
        'T_W',  # Water temperature
        'DO',  # Dissolved oxygen
        'pH',  # pH
        'COND',  # Conductivity
        'DOC',  # Dissolved organic carbon
    ]
    group_agro = [
        'GPAM',
        'WS',
        'FER',  # Fertilizer usage (water body location)
        'PES',  # Pesticide usage
        'MO',  # Corn output
        'CO',
        'VE',
        'FR',
        'FERA',  # Fertilizer per unit area
        'PESA',  # Pesticide per unit area
        'TSA',  # Total sown area
        'WA',  # Wheat area
        'CA',  # Corn area
        'VEGA',  # Vegetable area
        'IRR_W',  # Irrigation water usage
        'AGR_W',  # Agricultural water usage
    ]
    group_socio = [
        'UR',  # Urbanization rate
        'GAO',
        'PR',  # Primary industry ratio
        'SR',  # Secondary industry ratio
        'TR',  # Tertiary industry ratio
        'GDP',  # Gross Domestic Product
        'UI',  # Urban resident income
        'RI',  # Rural resident income
        'UR_W',  # Urban residential water usage
        'RU_W',  # Rural residential water usage
        'IND_W',  # Industrial water usage
        'LIF_W',  # Total residential water usage
        'POP_TOT',  # Total population
    ]

    @classmethod
    def get_feature_cols(cls, target: str):
        all_features = cls.group_natural + cls.group_agro + cls.group_socio
        match target:
            case 'THIA':
                all_features.extend(['Soil_THIA_agg'])
            case 'IMI':
                all_features.extend(['Soil_IMI_agg'])
            case 'CLO':
                all_features.extend(['Soil_CLO_agg', 'Soil_THIA_agg'])
            case 'IMI-UREA':
                all_features.extend(['Soil_IMI-UREA_agg', 'IMI'])
            case 'DN-IMI':
                all_features.extend(['Soil_DN-IMI_agg', 'IMI'])
            case 'DM-ACE':
                all_features.extend(['ACE'])
            case 'CLO-UREA':
                all_features.extend(['Soil_CLO-UREA_agg', 'CLO', 'THIA'])
        return all_features

    @classmethod
    def get_numerical_feature_cols(cls, target: str):
        return cls.get_feature_cols(target)

    @classmethod
    def get_natural_feature_cols(cls):
        return cls.group_natural


class SoilVariableGroups:
    metadata = [
        'ID',  # Sample ID
        'Lon',  # Longitude
        'Lat',  # Latitude
        'parentNNIs',  # Sum of parent neonicotinoids
        'mNNIs',  # Sum of metabolites
    ]
    categorical = [
        'Season',  # Season (Dry, Normal, Rainy)
        'landuse',  # Land use type at water body location
    ]
    targets_parent = [
        'THIA',  # Thiamethoxam
        'IMI',  # Imidacloprid
        'CLO',  # Clothianidin
    ]
    targets_metabolites = [
        'IMI-UREA',  # Imidacloprid-urea
        'DM-CLO',  # Desmethyl-clothianidin
        'DN-IMI',  # Desmethyl-imidacloprid
        'CLO-UREA',  # Clothianidin-urea
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
        'FER',  # Fertilizer usage (water body location)
        'PES',  # Pesticide usage
        'MO',  # Corn output
        'CO',
        'VE',
        'FR',
        'FERA',  # Fertilizer per unit area
        'PESA',  # Pesticide per unit area
        'TSA',  # Total sown area
        'WA',  # Wheat area
        'CA',  # Corn area
        'VEGA',  # Vegetable area
        'IRR_W',  # Irrigation water usage
        'AGR_W',  # Agricultural water usage
    ]
    group_socio = [
        'UR',  # Urbanization rate
        'GAO',
        'PR',  # Primary industry ratio
        'SR',  # Secondary industry ratio
        'TR',  # Tertiary industry ratio
        'GDP',  # Gross Domestic Product
        'UI',  # Urban resident income
        'RI',  # Rural resident income
        'UR_W',  # Urban residential water usage
        'RU_W',  # Rural residential water usage
        'IND_W',  # Industrial water usage
        'LIF_W',  # Total residential water usage
        'POP_TOT',  # Total population
    ]

    @classmethod
    def get_feature_cols(cls, target):
        all_features = cls.group_natural + cls.group_agro + cls.group_socio
        match target:
            case 'IMI-UREA':
                all_features.extend(['IMI'])
            case 'DN-IMI':
                all_features.extend(['IMI'])
            case 'CLO-UREA':
                all_features.extend(['CLO', 'THIA'])
            case 'DM-CLO':
                all_features.extend(['CLO', 'THIA'])
        return all_features

    @classmethod
    def get_numerical_feature_cols(cls, target):
        return cls.get_feature_cols(target)

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

    def prepare_data(self, target: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        # Extract targets
        target_cols = SoilVariableGroups.targets_parent + SoilVariableGroups.targets_metabolites
        assert target in target_cols
        y = self.df[target]
        # y_dict = {col: self.df[col] for col in target_cols if col in self.df.columns}
        #
        # # Extract metadata columns
        # metadata_cols = SoilVariableGroups.metadata
        #
        # # Features = all columns except targets and metadata
        # exclude_cols = list(y_dict.keys()) + metadata_cols
        # feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        feature_cols = SoilVariableGroups.get_feature_cols(target)
        assert all([col in self.df for col in feature_cols])
        X = self.df[feature_cols]

        return X, y, self.groups

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

    def prepare_data(self, target: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Prepare features, targets, and spatial groups.

        Args:
            df: Loaded dataframe

        Returns:
            Tuple of (X, y_dict, groups)
        """
        # Extract targets
        target_cols = MergedVariableGroups.targets_parent + MergedVariableGroups.targets_metabolites
        assert target in target_cols
        y = self.df[target]
        # y_dict = {col: self.df[col] for col in target_cols if col in self.df.columns}

        # Extract metadata columns
        # metadata_cols = MergedVariableGroups.metadata

        # Features = all columns except targets and metadata
        # exclude_cols = list(y_dict.keys()) + metadata_cols
        # feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        feature_cols = MergedVariableGroups.get_feature_cols(target)
        assert all([col in self.df for col in feature_cols])
        X = self.df[feature_cols]

        return X, y, self.groups

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
