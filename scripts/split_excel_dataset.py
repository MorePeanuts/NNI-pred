"""
This script is used to split the raw data from raw_data.xlsx into soil_data.csv and water_data.csv.
"""

import pandas as pd
from pathlib import Path


def main() -> None:
    """
    Split the raw_data.xlsx into:
    - 2 CSV files (water body data and soil data)
    """
    root_dir = Path(__file__).parent
    datasets_dir = root_dir.parent / 'datasets'
    raw_data_file_path = datasets_dir / 'raw_data.xlsx'
    assert raw_data_file_path.exists(), f'{raw_data_file_path} not exist.'

    try:
        # 1. Processing water body data
        print('Processing water body data...')
        water_data = pd.read_excel(raw_data_file_path, sheet_name='water')
        # Remove spaces on both sides of the column names
        water_data.columns = water_data.columns.str.strip()
        water_csv_path = datasets_dir / 'water_data.csv'
        water_data.to_csv(water_csv_path, index=False, encoding='utf-8')
        print(f'Water body data has been saved to: {water_csv_path}')

        # 2. Processing soil data
        print('Processing soil data...')
        soil_data = pd.read_excel(raw_data_file_path, sheet_name='soil')
        # Remove whitespace on both sides of column names
        soil_data.columns = soil_data.columns.str.strip()
        soil_csv_path = datasets_dir / 'soil_data.csv'
        soil_data.to_csv(soil_csv_path, index=False, encoding='utf-8')
        print(f'Soil data has been saved to: {soil_csv_path}')

        print('All files have been split and completed!')
        print('Generated file:')
        print(f'- CSV files: {water_csv_path}, {soil_csv_path}')

    except FileNotFoundError:
        print(f'Error: File not found {raw_data_file_path}')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
