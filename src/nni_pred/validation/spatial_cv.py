"""
Spatial Group Generator for GroupKFold Cross-Validation

This module generates spatial group IDs for use with sklearn's GroupKFold.
Each unique spatial location (defined by Lon/Lat) gets a unique group ID.
All seasonal samples from the same location share the same group ID.
"""

import pandas as pd
import numpy as np


class SpatialGroupGenerator:
    """
    Generate spatial groups for GroupKFold cross-validation.

    Since each of 53 locations has 3 seasonal samples (Dry, Normal, Rainy),
    we assign the same group ID to all samples from the same location.
    This ensures that when we split by groups, all seasons from a location
    stay together (preventing data leakage across space).
    """

    @staticmethod
    def generate_groups(df: pd.DataFrame, lon_col: str = 'Lon', lat_col: str = 'Lat') -> np.ndarray:
        """
        Generate group IDs based on spatial location.

        Args:
            df: DataFrame with location columns
            lon_col: Name of longitude column (default: 'Lon')
            lat_col: Name of latitude column (default: 'Lat')

        Returns:
            np.ndarray of shape (n_samples,) with group IDs (0 to n_locations-1)
        """
        # Extract unique locations
        locations = df[[lon_col, lat_col]].drop_duplicates().reset_index(drop=True)

        # Create mapping: (lon, lat) -> group_id
        location_to_group = {
            (row[lon_col], row[lat_col]): idx
            for idx, row in locations.iterrows()
        }

        # Assign group to each sample
        groups = df.apply(
            lambda row: location_to_group[(row[lon_col], row[lat_col])],
            axis=1
        ).values

        # Print summary
        n_groups = len(np.unique(groups))
        n_samples = len(df)
        samples_per_group = n_samples / n_groups

        print(f"\n{'='*60}")
        print("Spatial Group Generation Summary")
        print(f"{'='*60}")
        print(f"Total samples: {n_samples}")
        print(f"Unique spatial locations: {n_groups}")
        print(f"Samples per location: {samples_per_group:.1f} (expect ~3.0 for 3 seasons)")
        print(f"Group IDs range: 0 to {n_groups - 1}")
        print(f"{'='*60}\n")

        return groups

    @staticmethod
    def validate_groups(df: pd.DataFrame, groups: np.ndarray) -> dict:
        """
        Validate spatial groups to ensure correct structure.

        Args:
            df: DataFrame with 'Season' column
            groups: Generated group IDs

        Returns:
            Dictionary with validation results
        """
        n_groups = len(np.unique(groups))

        # Check samples per group
        unique_groups, counts = np.unique(groups, return_counts=True)
        samples_per_group = dict(zip(unique_groups, counts))

        # Expected: 3 samples per group (3 seasons)
        groups_with_3_samples = sum(1 for count in counts if count == 3)
        groups_with_other = sum(1 for count in counts if count != 3)

        # Check if each group has all 3 seasons
        if 'Season' in df.columns:
            seasons_per_group = {}
            for group_id in unique_groups:
                group_seasons = df[groups == group_id]['Season'].unique()
                seasons_per_group[group_id] = sorted(group_seasons)

            groups_with_all_seasons = sum(
                1 for seasons in seasons_per_group.values()
                if len(seasons) == 3
            )
        else:
            seasons_per_group = None
            groups_with_all_seasons = None

        validation = {
            'n_groups': n_groups,
            'samples_per_group': samples_per_group,
            'groups_with_3_samples': groups_with_3_samples,
            'groups_with_other_counts': groups_with_other,
            'all_groups_have_3_seasons': groups_with_all_seasons == n_groups if seasons_per_group else None,
            'valid': groups_with_3_samples == n_groups,
        }

        return validation

    @staticmethod
    def print_validation_report(validation: dict):
        """
        Print human-readable validation report.

        Args:
            validation: Dictionary from validate_groups()
        """
        print(f"\n{'='*60}")
        print("Spatial Group Validation Report")
        print(f"{'='*60}")
        print(f"Total spatial groups: {validation['n_groups']}")
        print(f"Groups with 3 samples: {validation['groups_with_3_samples']}")
        print(f"Groups with other counts: {validation['groups_with_other_counts']}")

        if validation['all_groups_have_3_seasons'] is not None:
            status = "✓ PASS" if validation['all_groups_have_3_seasons'] else "✗ FAIL"
            print(f"All groups have 3 seasons: {status}")

        overall_status = "✓ VALID" if validation['valid'] else "✗ INVALID"
        print(f"\nOverall validation: {overall_status}")
        print(f"{'='*60}\n")

        if not validation['valid']:
            print("WARNING: Some groups do not have exactly 3 samples!")
            print("This may indicate missing data or duplicate locations.")
            print("Samples per group:")
            for group_id, count in validation['samples_per_group'].items():
                if count != 3:
                    print(f"  Group {group_id}: {count} samples")
