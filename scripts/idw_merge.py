"""
IDW-based Water-Soil Sample Matching and General Data Preprocessing

This script implements the data preprocessing pipeline described in the research plan:
1. Water-Soil Sample Matching:
   - For each water sample, find all soil samples within 30km radius with the same season
   - Aggregate soil numerical features using IDW2 (Inverse Distance Weighted, p=2)
   - Take the nearest soil sample for categorical features (landuse)
   - Aggregated variables are named as Soil_XXX_agg

2. General Data Preprocessing:
   - One-hot encode categorical variables (Season, Landuse, Soil_landuse)
   - Log1p transform target variables (11 pollutant concentrations)
   - Save the processed dataset to datasets/processed_data.csv
"""

import argparse
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


def find_neighbors_within_radius(
    water_lon: float,
    water_lat: float,
    water_season: str,
    soil_df: pd.DataFrame,
    radius_km: float = 30.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Find all soil samples within specified radius and same season as water sample

    Args:
        water_lon: Water sample longitude
        water_lat: Water sample latitude
        water_season: Water sample season
        soil_df: Complete soil dataframe
        radius_km: Search radius in kilometers (default: 30km)

    Returns:
        Tuple of (filtered soil samples, distances array)
    """
    # Filter soil samples by same season first
    soil_season = soil_df[soil_df['Season'] == water_season].copy()

    if len(soil_season) == 0:
        raise ValueError(f'No soil samples found for season {water_season}')

    # Calculate distances to all soil samples in the same season
    distances = np.array(
        [
            haversine_distance(water_lon, water_lat, row['Lon'], row['Lat'])  # type: ignore
            for _, row in soil_season.iterrows()
        ]
    )

    # Filter samples within radius
    within_radius_mask = distances <= radius_km
    neighbors = soil_season[within_radius_mask].copy()
    neighbor_distances = distances[within_radius_mask]

    if len(neighbors) == 0:
        raise ValueError(
            f'No soil samples found within {radius_km}km radius for season {water_season}'
        )

    return neighbors, neighbor_distances  # type: ignore


def aggregate_soil_features_idw(
    neighbors: pd.DataFrame, distances: np.ndarray, numerical_vars: list[str]
) -> dict:
    """
    Aggregate soil numerical features using IDW2 (Inverse Distance Weighted, p=2)

    Formula: V_soil_agg = Σ(w_ij * V_soil_j) / Σ(w_ij)
    where w_ij = 1 / (d_ij ^ 2)

    Args:
        neighbors: Soil samples within radius
        distances: Distances to these samples
        numerical_vars: List of numerical variable names to aggregate

    Returns:
        Dictionary of aggregated features with naming pattern Soil_XXX_agg
    """
    features = {}

    # Avoid division by zero - add small epsilon for very close points
    epsilon = 1e-10
    distances_safe = distances + epsilon

    # Calculate weights (inverse distance squared, p=2)
    weights = 1 / (distances_safe**2)
    weights_sum = weights.sum()

    # Calculate weighted average for each numerical variable
    for var in numerical_vars:
        if var in neighbors.columns:
            values = neighbors[var].values
            # Handle potential NaN values
            if np.all(np.isnan(values)):
                weighted_value = np.nan
            else:
                # Only use non-NaN values for aggregation
                valid_mask = ~np.isnan(values)
                if np.any(valid_mask):
                    valid_values = values[valid_mask]
                    valid_weights = weights[valid_mask]
                    weighted_value = np.sum(valid_values * valid_weights) / valid_weights.sum()
                else:
                    weighted_value = np.nan

            features[f'Soil_{var}_agg'] = weighted_value

    return features


def get_nearest_categorical(neighbors: pd.DataFrame, distances: np.ndarray) -> str:
    """
    Get the landuse category from the nearest soil sample

    Args:
        neighbors: Soil samples within radius
        distances: Distances to these samples

    Returns:
        Landuse value from the nearest sample
    """
    nearest_idx = np.argmin(distances)
    return neighbors.iloc[nearest_idx]['landuse']  # type: ignore


def merge_water_soil_samples(
    soil_df: pd.DataFrame, water_df: pd.DataFrame, radius_km: float = 30.0
) -> pd.DataFrame:
    """
    Merge water and soil datasets by spatial matching using IDW aggregation

    Args:
        soil_df: Soil dataset
        water_df: Water dataset
        radius_km: Search radius in kilometers (default: 30km)

    Returns:
        Merged dataframe with soil features aggregated as Soil_XXX_agg
    """
    print(f'Starting water-soil sample matching with {radius_km}km radius...')
    print(f'Soil data shape: {soil_df.shape}')
    print(f'Water data shape: {water_df.shape}')

    # Define soil numerical variables to aggregate (excluding ID, coordinates, Season, and categorical)
    soil_pollutants = [
        'THIA',
        'IMI',
        'CLO',
        'parentNNIs',
        'IMI-UREA',
        'DN-IMI',
        'CLO-UREA',
        'mNNIs',
    ]

    # Seasonal variables
    soil_seasonal_vars = ['pH', 'TN', 'TOC', 'Temp', 'Rain', 'EVI', 'FCOVER', 'LAI', 'LST', 'NDVI']

    # Annual average variables
    soil_annual_vars = [
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

    # All numerical variables to aggregate
    soil_numerical_vars = soil_pollutants + soil_seasonal_vars + soil_annual_vars

    print(f'\nSoil numerical variables to aggregate: {len(soil_numerical_vars)}')
    print(f'  - Pollutants: {len(soil_pollutants)}')
    print(f'  - Seasonal variables: {len(soil_seasonal_vars)}')
    print(f'  - Annual variables: {len(soil_annual_vars)}')

    # Process each water sample
    merged_rows = []
    skipped_samples = []

    for idx, water_row in water_df.iterrows():
        if (idx + 1) % 20 == 0:  # type: ignore
            print(f'Processing water sample {idx + 1}/{len(water_df)}...')  # type: ignore

        try:
            # Find neighboring soil samples within radius and same season
            neighbors, distances = find_neighbors_within_radius(
                water_lon=water_row['Lon'],  # type: ignore
                water_lat=water_row['Lat'],  # type: ignore
                water_season=water_row['Season'],  # type: ignore
                soil_df=soil_df,
                radius_km=radius_km,
            )

            # Start with original water sample data
            merged_row = water_row.to_dict()

            # Aggregate soil numerical features using IDW2
            idw_features = aggregate_soil_features_idw(neighbors, distances, soil_numerical_vars)
            merged_row.update(idw_features)

            # Get categorical feature from nearest neighbor
            merged_row['Soil_landuse'] = get_nearest_categorical(neighbors, distances)

            merged_rows.append(merged_row)

        except ValueError as e:
            print(f'  Warning: Skipping water sample {idx}: {e}')  # type: ignore
            skipped_samples.append(idx)
            continue

    # Create merged dataframe
    merged_df = pd.DataFrame(merged_rows)

    print(f'\n{"=" * 60}')
    print('WATER-SOIL MATCHING COMPLETE')
    print(f'{"=" * 60}')
    print(f'Successfully merged: {len(merged_rows)} water samples')
    if skipped_samples:
        print(f'Skipped samples: {len(skipped_samples)} (IDs: {skipped_samples})')
    print(f'Total soil features added: {len(soil_numerical_vars) + 1}')
    print(f'  - Aggregated numerical (Soil_XXX_agg): {len(soil_numerical_vars)}')
    print('  - Categorical (Soil_landuse): 1')

    return merged_df


def apply_general_preprocessing(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply general data preprocessing:
    1. One-hot encode categorical variables (Season, Landuse, Soil_landuse)
    2. Log1p transform target variables (11 pollutant concentrations)

    Args:
        merged_df: Merged water-soil dataset

    Returns:
        Preprocessed dataframe
    """
    print(f'\n{"=" * 60}')
    print('APPLYING GENERAL PREPROCESSING')
    print(f'{"=" * 60}')

    df = merged_df.copy()

    # 1. One-hot encode categorical variables
    print('\n1. One-hot encoding categorical variables...')
    categorical_vars = ['Season', 'Landuse', 'Soil_landuse']

    for var in categorical_vars:
        if var in df.columns:
            print(f'   - {var}: {df[var].nunique()} unique values')
            # Create dummy variables
            dummies = pd.get_dummies(df[var], prefix=var, drop_first=False)
            # Drop original column and add dummy columns
            df = df.drop(columns=[var])
            df = pd.concat([df, dummies], axis=1)

    # 2. Log1p transform target variables (pollutant concentrations)
    print('\n2. Log1p transforming target variables...')

    # Water pollutants (target variables)
    water_pollutants = [
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

    transformed_count = 0
    for pollutant in water_pollutants:
        if pollutant in df.columns:
            df[pollutant] = np.log1p(df[pollutant])
            transformed_count += 1
            print(f'   - {pollutant}: log1p transformed')

    print(f'\nTotal target variables transformed: {transformed_count}')

    return df


def main():
    """Main execution function"""
    # Define paths
    project_root = Path(__file__).parent.parent
    soil_path = project_root / 'datasets' / 'soil_data.csv'
    water_path = project_root / 'datasets' / 'water_data.csv'
    output_path = project_root / 'datasets' / 'processed_data.csv'

    # Check if input files exist
    if not soil_path.exists():
        raise FileNotFoundError(f'Soil data not found: {soil_path}')
    if not water_path.exists():
        raise FileNotFoundError(f'Water data not found: {water_path}')

    print(f'{"=" * 60}')
    print('IDW-BASED DATA PREPROCESSING PIPELINE')
    print(f'{"=" * 60}')

    # Load datasets
    print('\nLoading datasets...')
    soil_df = pd.read_csv(soil_path)
    water_df = pd.read_csv(water_path)

    # Step 1: Water-Soil Sample Matching with IDW aggregation
    merged_df = merge_water_soil_samples(soil_df, water_df, radius_km=30.0)

    if args.preprocessing:
        # Step 2: General Preprocessing (one-hot encoding and log transformation)
        processed_df = apply_general_preprocessing(merged_df)
    else:
        processed_df = merged_df

    # Save processed dataset
    print(f'\n{"=" * 60}')
    print('SAVING PROCESSED DATASET')
    print(f'{"=" * 60}')
    print(f'Output path: {output_path}')
    print(f'Final shape: {processed_df.shape}')

    processed_df.to_csv(output_path, index=False)

    # Print summary
    print(f'\n{"=" * 60}')
    print('PROCESSING COMPLETE!')
    print(f'{"=" * 60}')
    print(f'\nDataset saved to: {output_path}')
    print(f'Total samples: {len(processed_df)}')
    print(f'Total features: {processed_df.shape[1]}')

    # Feature breakdown
    soil_agg_features = [
        col for col in processed_df.columns if col.startswith('Soil_') and col.endswith('_agg')
    ]
    onehot_features = [
        col
        for col in processed_df.columns
        if col.startswith('Season_')
        or col.startswith('Landuse_')
        or col.startswith('Soil_landuse_')
    ]

    print('\nFeature breakdown:')
    print(f'  - Soil aggregated features: {len(soil_agg_features)}')
    print(f'  - One-hot encoded features: {len(onehot_features)}')
    print(
        f'  - Original water features: {processed_df.shape[1] - len(soil_agg_features) - len(onehot_features) - 1}'
    )  # -1 for Soil_landuse before encoding

    print('\nFirst 3 rows preview:')
    print(processed_df.head(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply KNN-IDW algorithm to connect soil samples and water samples.'
    )
    parser.add_argument(
        '--preprocessing',
        action='store_true',
        help='Apply one-hot encoding and log transformation if true',
    )
    args = parser.parse_args()
    main()
