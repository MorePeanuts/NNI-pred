#!/usr/bin/env python
"""
Calculate Detection Rates of Pollutants in Water Bodies

This script analyzes the detection rates of all pollutants in the water body dataset.
Detection rate is defined as the percentage of samples with concentration > 0.
Since the data is log1p transformed, we need to inverse transform first.

Usage:
    uv run scripts/calculate_detection_rate.py
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nni_pred.preprocessing import get_feature_groups


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate detection rates of pollutants in water bodies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to processed_data.csv (default: datasets/processed_data.csv)',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output files (default: output/)',
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Detection threshold in original concentration units (default: 0.0)',
    )

    return parser.parse_args()


def calculate_detection_rates(
    df: pd.DataFrame, pollutant_cols: list[str], threshold: float = 0.0
) -> pd.DataFrame:
    """
    Calculate detection rates for specified pollutants.

    Args:
        df: DataFrame with log1p transformed pollutant concentrations
        pollutant_cols: List of pollutant column names
        threshold: Detection threshold in original concentration units

    Returns:
        DataFrame with detection rates for each pollutant
    """
    results = []

    for pollutant in pollutant_cols:
        if pollutant not in df.columns:
            print(f'WARNING: {pollutant} not found in data')
            continue

        # Get log1p transformed values
        log_values = df[pollutant].values

        # Inverse transform to original scale
        original_values = np.expm1(log_values)

        # Calculate detection rate (values > threshold)
        detected = (original_values > threshold).sum()
        total = len(original_values)
        detection_rate = (detected / total) * 100

        # Calculate statistics for detected values only
        if detected > 0:
            detected_values = original_values[original_values > threshold]
            mean_concentration = detected_values.mean()
            median_concentration = np.median(detected_values)
            std_concentration = detected_values.std()
            min_concentration = detected_values.min()
            max_concentration = detected_values.max()
        else:
            mean_concentration = median_concentration = std_concentration = min_concentration = (
                max_concentration
            ) = 0.0

        # Categorize pollutant
        category = (
            'Parent Compound'
            if pollutant in ['THIA', 'IMI', 'CLO', 'ACE', 'DIN']
            else 'Total'
            if pollutant in ['parentNNIs', 'mNNIs']
            else 'Metabolite'
        )

        results.append(
            {
                'Pollutant': pollutant,
                'Category': category,
                'Detection Rate (%)': round(detection_rate, 2),
                'Detected Samples': detected,
                'Total Samples': total,
                'Mean Concentration': round(mean_concentration, 3),
                'Median Concentration': round(median_concentration, 3),
                'Std Concentration': round(std_concentration, 3),
                'Min Concentration': round(min_concentration, 3),
                'Max Concentration': round(max_concentration, 3),
            }
        )

    return pd.DataFrame(results)


def print_summary_table(results_df: pd.DataFrame):
    """Print a formatted summary table."""
    print('\n' + '=' * 100)
    print('DETECTION RATE SUMMARY TABLE')
    print('=' * 100)

    # Format table
    print(
        f'{"Pollutant":<15} {"Category":<15} {"Detection Rate":<15} {"Detected/Total":<20} {"Mean Conc.":<12} {"Median Conc.":<14}'
    )
    print('-' * 100)

    for _, row in results_df.iterrows():
        print(
            f'{row["Pollutant"]:<15} {row["Category"]:<15} {row["Detection Rate (%)"]:<14.2f}% '
            f'{row["Detected Samples"]}/{row["Total Samples"]:<14} {row["Mean Concentration"]:<12.3f} '
            f'{row["Median Concentration"]:<14.3f}'
        )

    print('=' * 100)


def calculate_seasonal_detection_rates(
    df: pd.DataFrame, pollutant_cols: list[str], threshold: float = 0.0
) -> pd.DataFrame:
    """
    Calculate detection rates by season.

    Args:
        df: DataFrame with data and season columns
        pollutant_cols: List of pollutant column names
        threshold: Detection threshold in original concentration units

    Returns:
        DataFrame with seasonal detection rates
    """
    # Determine season columns
    season_cols = ['Season_Dry', 'Season_Normal', 'Season_Rainy']
    season_names = ['Dry', 'Normal', 'Rainy']

    results = []

    for season, season_col in zip(season_names, season_cols):
        if season_col not in df.columns:
            print(f'WARNING: {season_col} not found in data')
            continue

        # Get data for this season
        season_mask = df[season_col] == True
        season_df = df[season_mask]

        print(f'\n{season} Season (n={len(season_df)} samples):')

        season_results = calculate_detection_rates(season_df, pollutant_cols, threshold)  # type: ignore
        season_results['Season'] = season

        results.append(season_results)

        # Print seasonal summary
        for _, row in season_results.iterrows():
            if row['Detection Rate (%)'] > 0:
                print(
                    f'  {row["Pollutant"]:<15}: {row["Detection Rate (%)"]:.1f}% '
                    f'({row["Detected Samples"]}/{row["Total Samples"]})'
                )

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def save_results(results_df: pd.DataFrame, seasonal_df: pd.DataFrame, output_dir: Path):
    """Save results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save overall results
    overall_path = output_dir / 'detection_rates_overall.csv'
    results_df.to_csv(overall_path, index=False)
    print(f'\nOverall results saved to: {overall_path}')

    # Save seasonal results if available
    if not seasonal_df.empty:
        seasonal_path = output_dir / 'detection_rates_seasonal.csv'
        seasonal_df.to_csv(seasonal_path, index=False)
        print(f'Seasonal results saved to: {seasonal_path}')

    # Save detailed statistics
    detailed_path = output_dir / 'detection_rates_detailed.csv'
    results_df.to_csv(detailed_path, index=False)
    print(f'Detailed statistics saved to: {detailed_path}')


def main():
    """Main execution function."""
    args = parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent

    if args.data_path is None:
        data_path = project_root / 'datasets' / 'processed_data.csv'
    else:
        data_path = Path(args.data_path)

    if args.output_dir is None:
        output_dir = project_root / 'output'
    else:
        output_dir = Path(args.output_dir)

    # Check if data file exists
    if not data_path.exists():
        print(f'ERROR: Data file not found: {data_path}')
        print('Please run the preprocessing script first:')
        print('  uv run scripts/idw_merge.py')
        sys.exit(1)

    # Load data
    print(f'\nLoading data from: {data_path}')
    df = pd.read_csv(data_path)
    print(f'Data shape: {df.shape}')

    # Get pollutant columns
    feature_groups = get_feature_groups()
    pollutant_cols = feature_groups['targets']

    print(f'\nAnalyzing {len(pollutant_cols)} pollutants:')
    print(f'  Parent compounds: THIA, IMI, CLO, ACE, DIN')
    print(f'  Metabolites: IMI-UREA, DN-IMI, DM-ACE, CLO-UREA')
    print(f'  Totals: parentNNIs, mNNIs')

    # Calculate overall detection rates
    print(f'\n{"=" * 80}')
    print('CALCULATING OVERALL DETECTION RATES')
    print(f'{"=" * 80}')
    print(f'Detection threshold: {args.threshold} (original concentration units)')

    results_df = calculate_detection_rates(df, pollutant_cols, args.threshold)

    # Sort by detection rate (descending)
    results_df = results_df.sort_values('Detection Rate (%)', ascending=False)

    # Print summary table
    print_summary_table(results_df)

    # Calculate seasonal detection rates
    print(f'\n{"=" * 80}')
    print('CALCULATING SEASONAL DETECTION RATES')
    print(f'{"=" * 80}')

    seasonal_df = calculate_seasonal_detection_rates(df, pollutant_cols, args.threshold)

    # Print summary by category
    print(f'\n{"=" * 80}')
    print('DETECTION RATES BY CATEGORY')
    print(f'{"=" * 80}')

    for category in ['Parent Compound', 'Metabolite', 'Total']:
        category_df = results_df[results_df['Category'] == category]
        if not category_df.empty:
            print(f'\n{category} (n={len(category_df)}):')
            print(f'  Average detection rate: {category_df["Detection Rate (%)"].mean():.1f}%')
            print(
                f'  Range: {category_df["Detection Rate (%)"].min():.1f}% - {category_df["Detection Rate (%)"].max():.1f}%'
            )

    # Save results
    save_results(results_df, seasonal_df, Path(output_dir))

    print(f'\n{"=" * 80}')
    print('ANALYSIS COMPLETE')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()

