"""
Spatial Feature Engineering (SFE) - Merge soil and water datasets

This script implements the RFP (Random Forest Proximity) method to link soil and water samples
by spatial proximity, creating a merged dataset for NNIs prediction.

Processing methods:
1. RFP: For pollutant concentrations (8 variables) - keep raw values from k=5 nearest neighbors
2. Nearest neighbor: For landuse - use the value from the nearest soil sample
3. IDW2: For other variables - inverse distance squared weighted average

Reference: Hengl et al. (2018, PeerJ) - https://peerj.com/articles/5518/
"""

import pandas as pd
import numpy as np
from pathlib import Path


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points on earth (in kilometers)

    Args:
        lon1, lat1: Coordinates of point 1
        lon2, lat2: Coordinates of point 2

    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r


def find_nearest_neighbors(
    water_lon: float, water_lat: float, soil_df: pd.DataFrame, k: int = 5
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Find k nearest soil samples to a water sample point

    Args:
        water_lon: Water sample longitude
        water_lat: Water sample latitude
        soil_df: Soil dataframe (already filtered by season)
        k: Number of nearest neighbors

    Returns:
        Tuple of (nearest k soil samples, distances array)
    """
    # Calculate distances to all soil samples
    distances = np.array(
        [
            haversine_distance(water_lon, water_lat, row['Lon'], row['Lat'])  # type: ignore
            for _, row in soil_df.iterrows()
        ]
    )

    # Get indices of k nearest neighbors
    nearest_indices = np.argsort(distances)[:k]

    # Return nearest neighbors and their distances
    return soil_df.iloc[nearest_indices].copy(), distances[nearest_indices]


def extract_rfp_features(
    nearest_soils: pd.DataFrame, distances: np.ndarray, pollutant_vars: list[str]
) -> dict:
    """
    Extract RFP features: pollutant concentrations from k nearest neighbors

    Args:
        nearest_soils: k nearest soil samples
        distances: Distances to k nearest neighbors
        pollutant_vars: List of pollutant variable names

    Returns:
        Dictionary of RFP features
    """
    features = {}

    # Extract pollutant concentrations from each neighbor
    for i, (_, soil_row) in enumerate(nearest_soils.iterrows(), 1):
        for var in pollutant_vars:
            features[f'Soil_{var}_{i}'] = soil_row[var]

    # Add distances
    for i, dist in enumerate(distances, 1):
        features[f'Soil_Dist_{i}'] = dist

    return features


def extract_idw2_features(
    nearest_soils: pd.DataFrame, distances: np.ndarray, idw_vars: list[str]
) -> dict:
    """
    Extract IDW2 weighted features for environmental variables

    Args:
        nearest_soils: k nearest soil samples
        distances: Distances to k nearest neighbors
        idw_vars: List of variables to apply IDW2

    Returns:
        Dictionary of IDW2 weighted features
    """
    features = {}

    # Avoid division by zero - add small epsilon for very close points
    epsilon = 1e-10
    distances_safe = distances + epsilon

    # Calculate weights (inverse distance squared)
    weights = 1 / (distances_safe**2)
    weights_normalized = weights / weights.sum()

    # Calculate weighted average for each variable
    for var in idw_vars:
        values = nearest_soils[var].values
        weighted_value = np.sum(values * weights_normalized)
        features[f'Soil_{var}_wMean'] = weighted_value

    return features


def merge_datasets(soil_path: str, water_path: str, output_path: str, k: int = 5) -> pd.DataFrame:
    """
    Main function to merge soil and water datasets using spatial feature engineering

    Args:
        soil_path: Path to soil dataset
        water_path: Path to water dataset
        output_path: Path to save merged dataset
        k: Number of nearest neighbors (default: 5)

    Returns:
        Merged dataframe
    """
    print('Loading datasets...')
    soil_df = pd.read_csv(soil_path)
    water_df = pd.read_csv(water_path)

    print(f'Soil data shape: {soil_df.shape}')
    print(f'Water data shape: {water_df.shape}')

    # Define variable groups
    # Pollutant variables (8 total) - to be processed with RFP
    pollutant_vars = ['THIA', 'IMI', 'CLO', 'parentNNIs', 'IMI-UREA', 'DN-IMI', 'CLO-UREA', 'mNNIs']

    # Variables to exclude from IDW2 processing
    exclude_vars = ['ID', 'Lon', 'Lat', 'Season', 'landuse'] + pollutant_vars

    # Get all soil columns for IDW2 processing
    idw_vars = [col for col in soil_df.columns if col not in exclude_vars]

    print('\nProcessing strategy:')
    print(
        f'  - RFP (k={k}): {len(pollutant_vars)} pollutant variables → {len(pollutant_vars) * k + k} features'
    )
    print('  - Nearest neighbor: landuse → 1 feature')
    print(f'  - IDW2: {len(idw_vars)} environmental variables → {len(idw_vars)} features')

    # Process each water sample
    merged_rows = []

    for idx, water_row in water_df.iterrows():
        if (idx + 1) % 20 == 0:  # type: ignore
            print(f'Processing water sample {idx + 1}/{len(water_df)}...')  # type: ignore

        # Filter soil samples by same season
        season = water_row['Season']
        soil_season = soil_df[soil_df['Season'] == season]

        if len(soil_season) < k:
            print(f'Warning: Only {len(soil_season)} soil samples available for season {season}')
            continue

        # Find k nearest neighbors
        nearest_soils, distances = find_nearest_neighbors(
            water_row['Lon'],  # type: ignore
            water_row['Lat'],  # type: ignore
            soil_season,  # type: ignore
            k,
        )

        # Start with water sample data
        merged_row = water_row.to_dict()

        # Extract RFP features (pollutants + distances)
        rfp_features = extract_rfp_features(nearest_soils, distances, pollutant_vars)
        merged_row.update(rfp_features)

        # Extract nearest neighbor landuse
        merged_row['Soil_landuse'] = nearest_soils.iloc[0]['landuse']

        # Extract IDW2 weighted features
        idw_features = extract_idw2_features(nearest_soils, distances, idw_vars)
        merged_row.update(idw_features)

        merged_rows.append(merged_row)

    # Create merged dataframe
    merged_df = pd.DataFrame(merged_rows)

    print(f'\nMerged data shape: {merged_df.shape}')
    print(f'Total features added from soil: {len(pollutant_vars) * k + k + 1 + len(idw_vars)}')

    # Save to CSV
    print(f'\nSaving merged dataset to {output_path}...')
    merged_df.to_csv(output_path, index=False)

    # Print feature summary
    print('\n' + '=' * 60)
    print('FEATURE SUMMARY')
    print('=' * 60)

    # Original water features (excluding pollutants which are target variables)
    water_features = [
        col
        for col in water_df.columns
        if col
        not in ['ID', 'Lon', 'Lat']
        + [
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
    ]
    print(f'\nOriginal water features: {len(water_features)}')

    # Soil-derived features
    rfp_cols = [
        col
        for col in merged_df.columns
        if col.startswith('Soil_')
        and (
            '_1' in col
            or '_2' in col
            or '_3' in col
            or '_4' in col
            or '_5' in col
            or col.startswith('Soil_Dist_')
        )
    ]
    print(f'RFP features (pollutants + distances): {len(rfp_cols)}')

    idw_cols = [col for col in merged_df.columns if col.endswith('_wMean')]
    print(f'IDW2 weighted features: {len(idw_cols)}')

    print('Landuse feature: 1 (Soil_landuse)')

    # Target variables
    target_vars = [
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
    available_targets = [t for t in target_vars if t in merged_df.columns]
    print(f'\nTarget variables (water pollutants): {len(available_targets)}')
    print(f'  {", ".join(available_targets)}')

    return merged_df


def main():
    """Main execution function"""
    # Define paths
    project_root = Path(__file__).parent.parent
    soil_path = project_root / 'datasets' / 'soil_data.csv'
    water_path = project_root / 'datasets' / 'water_data.csv'
    output_path = project_root / 'datasets' / 'merged_data.csv'

    # Check if input files exist
    if not soil_path.exists():
        raise FileNotFoundError(f'Soil data not found: {soil_path}')
    if not water_path.exists():
        raise FileNotFoundError(f'Water data not found: {water_path}')

    # Merge datasets
    merged_df = merge_datasets(str(soil_path), str(water_path), str(output_path), k=5)

    print('\n' + '=' * 60)
    print('MERGE COMPLETE!')
    print('=' * 60)
    print(f'\nOutput saved to: {output_path}')
    print(f'Final dataset shape: {merged_df.shape}')

    # Display first few columns
    print('\nFirst 5 rows preview:')
    print(merged_df.head())


if __name__ == '__main__':
    main()
