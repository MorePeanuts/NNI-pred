import pandas as pd
from pathlib import Path


def main() -> None:
    """
    将raw_data.xlsx中拆分为：
    - 2个CSV文件（水体数据和土壤数据）
    """
    root_dir = Path(__file__).parent
    datasets_dir = root_dir.parent / 'datasets'
    raw_data_file_path = datasets_dir / 'raw_data.xlsx'
    assert raw_data_file_path.exists(), f'{raw_data_file_path} not exist.'

    try:
        # 1. 处理水体数据
        print('处理水体数据...')
        water_data = pd.read_excel(raw_data_file_path, sheet_name='water')
        # 去除列名两边的空白
        water_data.columns = water_data.columns.str.strip()
        water_csv_path = datasets_dir / 'water_data.csv'
        water_data.to_csv(water_csv_path, index=False, encoding='utf-8')
        print(f'水体数据已保存到: {water_csv_path}')

        # 2. 处理土壤数据
        print('处理土壤数据...')
        soil_data = pd.read_excel(raw_data_file_path, sheet_name='soil')
        # 去除列名两边的空白
        soil_data.columns = soil_data.columns.str.strip()
        soil_csv_path = datasets_dir / 'soil_data.csv'
        soil_data.to_csv(soil_csv_path, index=False, encoding='utf-8')
        print(f'土壤数据已保存到: {soil_csv_path}')

        print('\n所有文件拆分完成！')
        print('生成的文件:')
        print(f'- CSV文件: {water_csv_path}, {soil_csv_path}')

    except FileNotFoundError:
        print(f'错误: 找不到文件 {raw_data_file_path}')
    except Exception as e:
        print(f'错误: {e}')


if __name__ == '__main__':
    main()
