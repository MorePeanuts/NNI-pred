"""
This script is used to merge the data from soil_data.csv and water_data.csv using the Inverse Distance Weighting (IDW) algorithm to obtain the merged_data.csv file.

For each water body sample point, find all soil sample points within a distance of 30 km. Use IDW2 (p=2) weighted averaging for some features,
 and for another set of features, take the characteristics of the nearest soil sample point.

The aggregated features are named Soil_{feature}_{agg} or Soil_{feature}.
"""

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from nni_pred.data import (
    SOIL_ANNUAL_VARS,
    SOIL_POLLUTANTS,
    SOIL_SEASONAL_VARS,
    SOIL_CATEGORICAL_VARS,
)


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
            haversine_distance(water_lon, water_lat, row['Lon'], row['Lat'])
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

    return neighbors, neighbor_distances


def aggregate_soil_features_idw(
    neighbors: pd.DataFrame, distances: np.ndarray, numerical_vars: list[str], p: int = 2
) -> dict:
    """
    Aggregate soil numerical features using IDW (Inverse Distance Weighted, p=2)

    Formula: V_soil_agg = Σ(w_ij * V_soil_j) / Σ(w_ij)
    where w_ij = 1 / (d_ij ^ p)

    Args:
        neighbors: Soil samples within radius
        distances: Distances to these samples
        numerical_vars: List of numerical variable names to aggregate
        p: Exponential of distance

    Returns:
        Dictionary of aggregated features with naming pattern Soil_XXX_agg
    """
    features = {}

    # Avoid division by zero - add small epsilon for very close points
    epsilon = 1e-10
    distances_safe = distances + epsilon

    # Calculate weights (inverse distance squared, p=2)
    weights = 1 / (distances_safe**p)

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


def get_nearest_categorical(neighbors: pd.DataFrame, distances: np.ndarray) -> dict:
    """
    Get the landuse category from the nearest soil sample

    Args:
        neighbors: Soil samples within radius
        distances: Distances to these samples

    Returns:
        A dict of categorical values from the nearest sample
    """
    nearest_idx = np.argmin(distances)
    res = {}
    for field in SOIL_CATEGORICAL_VARS:
        res['Soil_' + field] = neighbors.iloc[nearest_idx][field]
    return res


def merge_water_soil_samples(
    soil_df: pd.DataFrame, water_df: pd.DataFrame, radius_km: float = 30.0
):
    logger.info(f'Starting water-soil sample matching with {radius_km}km radius...')
    logger.info(f'Soil data shape: {soil_df.shape}')
    logger.info(f'Water data shape: {water_df.shape}')

    soil_numerical_vars = SOIL_POLLUTANTS + SOIL_SEASONAL_VARS + SOIL_ANNUAL_VARS

    logger.info(f'\nSoil numerical variables to aggregate: {len(soil_numerical_vars)}')
    logger.info(f'  - Pollutants: {len(SOIL_POLLUTANTS)}')
    logger.info(f'  - Seasonal variables: {len(SOIL_SEASONAL_VARS)}')
    logger.info(f'  - Annual variables: {len(SOIL_ANNUAL_VARS)}')

    # Process each water sample
    merged_rows = []

    for _, water_row in water_df.iterrows():
        neighbors, distances = find_neighbors_within_radius(
            water_lon=water_row['Lon'],
            water_lat=water_row['Lat'],
            water_season=water_row['Season'],
            soil_df=soil_df,
            radius_km=radius_km,
        )

        # Start with original water sample data
        merged_row = water_row.to_dict()

        # Aggregate soil numerical features using IDW2
        idw_features = aggregate_soil_features_idw(neighbors, distances, soil_numerical_vars)
        merged_row.update(idw_features)

        # Get categorical feature from nearest neighbor
        categorical_vars = get_nearest_categorical(neighbors, distances)
        for k, v in categorical_vars.items():
            merged_row[k] = v

        merged_rows.append(merged_row)

    # Create merged dataframe
    merged_df = pd.DataFrame(merged_rows)

    logger.info(f'Successfully merged: {len(merged_rows)} water samples')
    logger.info(f'Total soil features added: {len(soil_numerical_vars) + 1}')
    logger.info(f'  - Aggregated numerical (Soil_XXX_agg): {len(soil_numerical_vars)}')
    logger.info('  - Categorical (Soil_landuse): 1')

    return merged_df


def main():
    root_path = Path(__file__).parents[1]
    soil_path = root_path / 'datasets' / 'soil_data.csv'
    water_path = root_path / 'datasets' / 'water_data.csv'
    output_path = root_path / 'datasets' / 'merged_data.csv'

    # Check if input files exist
    if not soil_path.exists():
        raise FileNotFoundError(f'Soil data not found: {soil_path}')
    if not water_path.exists():
        raise FileNotFoundError(f'Water data not found: {water_path}')

    logger.info('IDW-based Data Preprocessing Pipeline running...')
    logger.info(f'Loadding datasets from {soil_path} and {water_path}')
    soil_df = pd.read_csv(soil_path)
    water_df = pd.read_csv(water_path)

    merged_df = merge_water_soil_samples(soil_df, water_df, radius_km=30.0)
    merged_df.to_csv(output_path, index=False)
    logger.info(f'\nDataset saved to: {output_path}')
    logger.info(f'Total samples: {len(merged_df)}')
    logger.info(f'Total features: {merged_df.shape[1]}')


if __name__ == '__main__':
    main()
