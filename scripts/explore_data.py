# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "tabulate",
#   "scipy"
# ]
# ///

"""
Exploratory Data Analysis for Soil and Water Pollutant Dataset
土壤和水体污染物数据集的探索性数据分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import itertools

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def dataframe_to_markdown(df, title="", precision=2):
    """Convert pandas DataFrame to markdown table string"""
    if df.empty:
        return f"\n{title}\n数据为空\n"

    # Format numeric values
    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=['number']).columns:
        if precision >= 0:
            df_formatted[col] = df_formatted[col].round(precision)

    # Convert to markdown
    markdown = f"\n{title}\n"
    markdown += df_formatted.to_markdown(floatfmt=f".{precision}f")
    markdown += "\n"

    return markdown

def load_data():
    """Load soil and water datasets"""
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'datasets'

    # Load datasets
    soil_data = pd.read_csv(data_path / 'soil_data.csv', index_col=0)
    water_data = pd.read_csv(data_path / 'water_data.csv', index_col=0)

    print(f"土壤数据集形状: {soil_data.shape}")
    print(f"水体数据集形状: {water_data.shape}")

    return soil_data, water_data

def basic_info_analysis(df, dataset_name):
    """Basic information analysis for dataset"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 数据集基本信息")
    print(f"{'='*50}")

    print(f"数据维度: {df.shape}")
    print(f"特征数量: {df.shape[1]}")
    print(f"样本数量: {df.shape[0]}")

    # Data types
    print(f"\n数据类型统计:")
    print(df.dtypes.value_counts())

    # Missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    print(f"\n缺失值统计:")
    missing_summary = pd.DataFrame({
        '缺失值数量': missing_values,
        '缺失比例(%)': missing_percentage
    })
    print(missing_summary[missing_summary['缺失值数量'] > 0])

    return missing_summary

def pollutant_analysis(df, dataset_name):
    """Analyze pollutant concentrations"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 污染物浓度分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        parent_pollutants = ['THIA', 'IMI', 'CLO', 'parentNNIs']
        metabolite_pollutants = ['IMI-UREA', 'DN-IMI', 'CLO-UREA', 'mNNIs']
    else:  # 水体
        parent_pollutants = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs']
        metabolite_pollutants = ['IMI-UREA', 'DN-IMI', 'DM-ACE', 'CLO-UREA', 'mNNIs']

    # Parent pollutant analysis
    parent_stats = df[parent_pollutants].describe()
    print(dataframe_to_markdown(parent_stats, "母体污染物统计", 2))

    metabolite_stats = df[metabolite_pollutants].describe()
    print(dataframe_to_markdown(metabolite_stats, "代谢物污染物统计", 2))

    # Detection rates (assuming 0 means not detected)
    print("\n检出率 (假设0表示未检出):")
    detection_rates = {}
    for pollutant in parent_pollutants + metabolite_pollutants:
        if pollutant in df.columns:
            detection_rate = (df[pollutant] > 0).mean() * 100
            detection_rates[pollutant] = detection_rate
            print(f"{pollutant}: {detection_rate:.1f}%")

    return parent_stats, metabolite_stats, detection_rates

def seasonal_analysis(df, dataset_name):
    """Analyze seasonal patterns"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 季节分布分析")
    print(f"{'='*50}")

    if 'Season' in df.columns:
        season_counts = df['Season'].value_counts()
        print("季节分布:")
        print(season_counts)

        # Seasonal pollutant analysis
        if dataset_name == "土壤":
            pollutants = ['parentNNIs', 'mNNIs']
        else:
            pollutants = ['parentNNIs', 'mNNIs']

        print("\n季节性污染物浓度分布:")
        for pollutant in pollutants:
            if pollutant in df.columns:
                seasonal_stats = df.groupby('Season')[pollutant].describe()
                print(dataframe_to_markdown(seasonal_stats, f"{pollutant} 按季节统计", 2))

        return season_counts
    else:
        print("未找到季节信息")
        return None

def vegetation_indices_analysis(df, dataset_name):
    """Analyze vegetation indices"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 植被指数分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        vegetation_indices = ['EVI', 'FCOVER', 'LAI', 'LST', 'NDVI']
        available_indices = [idx for idx in vegetation_indices if idx in df.columns]

        if available_indices:
            veg_stats = df[available_indices].describe()
            print(dataframe_to_markdown(veg_stats, "植被指数统计", 3))
            return veg_stats
        else:
            print("未找到植被指数数据")
            return None
    else:
        print("水体数据不包含植被指数")
        return None

def agricultural_variables_analysis(df, dataset_name):
    """Analyze agricultural variables"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 农业变量分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        agri_vars = ['CC', 'BD', 'GPAM', 'WS', 'FER', 'PES', 'MC', 'MCP',
                     'VE', 'VEP', 'FR', 'FRP', 'FERA', 'PESA', 'GAO']
        available_vars = [var for var in agri_vars if var in df.columns]
    else:  # 水体
        agri_vars = ['AMP', 'FER', 'PES', 'FERPER', 'PESPER', 'TSA', 'FCA',
                     'WA', 'CA', 'VEGA', 'CROPOUT', 'WO', 'CO', 'VO', 'FOP', 'IRR_W', 'AO', 'FO']
        available_vars = [var for var in agri_vars if var in df.columns]

    if available_vars:
        agri_stats = df[available_vars].describe()
        print(dataframe_to_markdown(agri_stats, "农业变量统计", 2))
        return agri_stats
    else:
        print("未找到农业变量数据")
        return None

def economic_variables_analysis(df, dataset_name):
    """Analyze economic variables"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 经济变量分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        econ_vars = ['UR', 'GAO', 'PR', 'SR', 'TR', 'GDP per capita', 'UI', 'RI']
        available_vars = [var for var in econ_vars if var in df.columns]
    else:  # 水体
        econ_vars = ['Urban', 'GDP', 'PR', 'SR', 'TR', 'UI', 'RI', 'POP_TOT',
                     'OP_FI', 'OP_SE', 'OP_TH']
        available_vars = [var for var in econ_vars if var in df.columns]

    if available_vars:
        econ_stats = df[available_vars].describe()
        print(dataframe_to_markdown(econ_stats, "经济变量统计", 2))
        return econ_stats
    else:
        print("未找到经济变量数据")
        return None

def water_usage_analysis(df, dataset_name):
    """Analyze water usage variables"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 用水变量分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        water_vars = ['UR_W', 'RU_W', 'IRR_W', 'AGR_W', 'IND_W', 'LIF_W']
        available_vars = [var for var in water_vars if var in df.columns]
    else:  # 水体
        water_vars = ['UR_W', 'RU_W', 'AGR_W', 'IND_W', 'LIF_W', 'IRR_W']
        available_vars = [var for var in water_vars if var in df.columns]

    if available_vars:
        water_stats = df[available_vars].describe()
        print(dataframe_to_markdown(water_stats, "用水变量统计", 2))
        return water_stats
    else:
        print("未找到用水变量数据")
        return None

def landuse_analysis(df, dataset_name):
    """Analyze land use patterns"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 土地利用分析")
    print(f"{'='*50}")

    if 'landuse' in df.columns or 'Landuse' in df.columns:
        landuse_col = 'landuse' if 'landuse' in df.columns else 'Landuse'
        landuse_counts = df[landuse_col].value_counts()
        print("土地利用分布:")
        print(landuse_counts)

        # Calculate percentages
        landuse_pct = df[landuse_col].value_counts(normalize=True) * 100
        print("\n土地利用比例 (%):")
        print(landuse_pct.round(1))

        return landuse_counts
    else:
        print("未找到土地利用数据")
        return None

def environmental_factors_analysis(df, dataset_name):
    """Analyze environmental factors"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 基础环境因子分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        env_factors = ['pH', 'TN', 'TOC', 'Temp', 'Rain', 'Alt', 'CC', 'BD']
    else:  # 水体
        env_factors = ['pH', 'DO', 'COND', 'DOC', 'T_W', 'T_M', 'PREC', 'Alt']

    available_factors = [factor for factor in env_factors if factor in df.columns]

    env_stats = df[available_factors].describe()
    print(dataframe_to_markdown(env_stats, "基础环境因子统计", 3))

    return env_stats

def geographic_analysis(df, dataset_name):
    """Analyze geographic distribution"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 地理分布分析")
    print(f"{'='*50}")

    if all(col in df.columns for col in ['Lon', 'Lat']):
        geo_stats = {
            '经度范围': (df['Lon'].min(), df['Lon'].max()),
            '纬度范围': (df['Lat'].min(), df['Lat'].max()),
            '海拔范围': (df['Alt'].min(), df['Alt'].max()) if 'Alt' in df.columns else None
        }

        for key, value in geo_stats.items():
            if value:
                print(f"{key}: {value[0]:.3f} ~ {value[1]:.3f}")

        return geo_stats
    else:
        print("未找到完整的地理坐标信息")
        return None

def create_enhanced_visualizations(soil_data, water_data):
    """Create enhanced visualization plots with additional analysis"""
    print("\n生成增强可视化图表...")

    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'outputs' / 'eda_plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Geographic distribution heatmaps
    if all(col in soil_data.columns for col in ['Lon', 'Lat']):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('污染物地理分布热力图', fontsize=16, fontweight='bold')

        # Soil THIA heatmap
        if 'THIA' in soil_data.columns:
            scatter = axes[0, 0].scatter(soil_data['Lon'], soil_data['Lat'],
                                       c=soil_data['THIA'], cmap='YlOrRd',
                                       alpha=0.7, s=30, edgecolors='black', linewidth=0.3)
            axes[0, 0].set_title('土壤 THIA 浓度分布')
            axes[0, 0].set_xlabel('经度')
            axes[0, 0].set_ylabel('纬度')
            plt.colorbar(scatter, ax=axes[0, 0], label='THIA (ng/g)')

        # Soil parentNNIs heatmap
        if 'parentNNIs' in soil_data.columns:
            scatter = axes[0, 1].scatter(soil_data['Lon'], soil_data['Lat'],
                                       c=soil_data['parentNNIs'], cmap='YlOrRd',
                                       alpha=0.7, s=30, edgecolors='black', linewidth=0.3)
            axes[0, 1].set_title('土壤 母体NNIs 总浓度分布')
            axes[0, 1].set_xlabel('经度')
            axes[0, 1].set_ylabel('纬度')
            plt.colorbar(scatter, ax=axes[0, 1], label='parentNNIs (ng/g)')

        # Water parentNNIs heatmap
        if 'parentNNIs' in water_data.columns:
            scatter = axes[1, 0].scatter(water_data['Lon'], water_data['Lat'],
                                       c=water_data['parentNNIs'], cmap='Blues',
                                       alpha=0.7, s=30, edgecolors='black', linewidth=0.3)
            axes[1, 0].set_title('水体 母体NNIs 总浓度分布')
            axes[1, 0].set_xlabel('经度')
            axes[1, 0].set_ylabel('纬度')
            plt.colorbar(scatter, ax=axes[1, 0], label='parentNNIs (ng/L)')

        # Water mNNIs heatmap
        if 'mNNIs' in water_data.columns:
            scatter = axes[1, 1].scatter(water_data['Lon'], water_data['Lat'],
                                       c=water_data['mNNIs'], cmap='Blues',
                                       alpha=0.7, s=30, edgecolors='black', linewidth=0.3)
            axes[1, 1].set_title('水体 代谢NNIs 浓度分布')
            axes[1, 1].set_xlabel('经度')
            axes[1, 1].set_ylabel('纬度')
            plt.colorbar(scatter, ax=axes[1, 1], label='mNNIs (ng/L)')

        plt.tight_layout()
        plt.savefig(output_dir / 'geographic_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Pollutant composition pie charts
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('污染物组成分析', fontsize=16, fontweight='bold')

    # Soil pollutant composition
    soil_parents = ['THIA', 'IMI', 'CLO']
    available_soil_parents = [p for p in soil_parents if p in soil_data.columns]
    if available_soil_parents:
        soil_composition = soil_data[available_soil_parents].sum()
        # Only show pollutants with concentration > 0
        soil_composition = soil_composition[soil_composition > 0]
        if len(soil_composition) > 0:
            axes[0].pie(soil_composition.values, labels=soil_composition.index,
                       autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
            axes[0].set_title('土壤母体污染物组成')

    # Water pollutant composition
    water_parents = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN']
    available_water_parents = [p for p in water_parents if p in water_data.columns]
    if available_water_parents:
        water_composition = water_data[available_water_parents].sum()
        water_composition = water_composition[water_composition > 0]
        if len(water_composition) > 0:
            axes[1].pie(water_composition.values, labels=water_composition.index,
                       autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'])
            axes[1].set_title('水体母体污染物组成')

    plt.tight_layout()
    plt.savefig(output_dir / 'pollutant_composition.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Environmental factors scatter plot matrix
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('环境因子与污染物关系散点图', fontsize=16, fontweight='bold')

    # Soil environmental correlations
    soil_pollutants = ['parentNNIs']
    soil_env_factors = ['pH', 'TOC', 'Temp', 'Rain', 'EVI', 'Alt']

    for i, factor in enumerate(soil_env_factors[:6]):
        if factor in soil_data.columns and 'parentNNIs' in soil_data.columns:
            row, col = i // 3, i % 3
            axes[row, col].scatter(soil_data[factor], soil_data['parentNNIs'],
                                  alpha=0.5, s=20, color='brown')
            axes[row, col].set_xlabel(factor)
            axes[row, col].set_ylabel('parentNNIs (ng/g)')
            axes[row, col].set_title(f'土壤: {factor} vs parentNNIs')

            # Add correlation coefficient
            corr = soil_data[[factor, 'parentNNIs']].corr().iloc[0, 1]
            axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}',
                              transform=axes[row, col].transAxes,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'environmental_scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Land use vs pollutant concentration boxplots
    if 'landuse' in soil_data.columns or 'Landuse' in water_data.columns:
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('土地利用类型与污染物浓度关系', fontsize=16, fontweight='bold')

        # Soil land use analysis
        if 'landuse' in soil_data.columns and 'parentNNIs' in soil_data.columns:
            landuse_order = soil_data.groupby('landuse')['parentNNIs'].median().sort_values(ascending=False).index
            soil_data.boxplot(column='parentNNIs', by='landuse', ax=axes[0], grid=False)
            axes[0].set_title('土壤不同土地利用类型的parentNNIs浓度')
            axes[0].set_xlabel('土地利用类型')
            axes[0].set_ylabel('parentNNIs (ng/g)')
            axes[0].tick_params(axis='x', rotation=45)

        # Water land use analysis
        if 'Landuse' in water_data.columns and 'parentNNIs' in water_data.columns:
            water_data.boxplot(column='parentNNIs', by='Landuse', ax=axes[1], grid=False)
            axes[1].set_title('水体不同土地利用类型的parentNNIs浓度')
            axes[1].set_xlabel('土地利用类型')
            axes[1].set_ylabel('parentNNIs (ng/L)')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'landuse_pollutant_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Agricultural intensity vs pollutants
    if 'FER' in soil_data.columns and 'PES' in soil_data.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('农业活动强度与污染物关系', fontsize=16, fontweight='bold')

        # Create agricultural intensity index
        fer_norm = (soil_data['FER'] - soil_data['FER'].min()) / (soil_data['FER'].max() - soil_data['FER'].min())
        pes_norm = (soil_data['PES'] - soil_data['PES'].min()) / (soil_data['PES'].max() - soil_data['PES'].min())
        agri_intensity = (fer_norm + pes_norm) / 2

        soil_temp = soil_data.copy()
        soil_temp['agri_intensity'] = agri_intensity
        soil_temp['intensity_group'] = pd.qcut(agri_intensity, 4, labels=['低', '中低', '中高', '高'])

        pollutants = ['THIA', 'IMI', 'CLO', 'parentNNIs']
        for i, poll in enumerate(pollutants):
            if poll in soil_data.columns:
                row, col = i // 2, i % 2
                soil_temp.boxplot(column=poll, by='intensity_group', ax=axes[row, col], grid=False)
                axes[row, col].set_title(f'{poll} 农业强度分组')
                axes[row, col].set_xlabel('农业强度')
                axes[row, col].set_ylabel(f'{poll} (ng/g)')

        plt.tight_layout()
        plt.savefig(output_dir / 'agricultural_intensity_pollutants.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Seasonal pollutant profiles
    if 'Season' in soil_data.columns and 'Season' in water_data.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('污染物季节性变化轮廓图', fontsize=16, fontweight='bold')

        # Soil seasonal profiles
        if 'parentNNIs' in soil_data.columns:
            season_order = ['Dry', 'Normal', 'Rainy']
            soil_seasonal_data = [soil_data[soil_data['Season'] == season]['parentNNIs'].dropna()
                                 for season in season_order]

            bp = axes[0, 0].boxplot(soil_seasonal_data, labels=season_order, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            axes[0, 0].set_title('土壤 parentNNIs 季节变化')
            axes[0, 0].set_ylabel('浓度 (ng/g)')

        if 'mNNIs' in soil_data.columns:
            soil_seasonal_meta = [soil_data[soil_data['Season'] == season]['mNNIs'].dropna()
                                 for season in season_order]

            bp = axes[0, 1].boxplot(soil_seasonal_meta, labels=season_order, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')
            axes[0, 1].set_title('土壤 mNNIs 季节变化')
            axes[0, 1].set_ylabel('浓度 (ng/g)')

        # Water seasonal profiles
        if 'parentNNIs' in water_data.columns:
            water_seasonal_data = [water_data[water_data['Season'] == season]['parentNNIs'].dropna()
                                  for season in season_order]

            bp = axes[1, 0].boxplot(water_seasonal_data, labels=season_order, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
            axes[1, 0].set_title('水体 parentNNIs 季节变化')
            axes[1, 0].set_ylabel('浓度 (ng/L)')

        if 'mNNIs' in water_data.columns:
            water_seasonal_meta = [water_data[water_data['Season'] == season]['mNNIs'].dropna()
                                  for season in season_order]

            bp = axes[1, 1].boxplot(water_seasonal_meta, labels=season_order, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightyellow')
            axes[1, 1].set_title('水体 mNNIs 季节变化')
            axes[1, 1].set_ylabel('浓度 (ng/L)')

        plt.tight_layout()
        plt.savefig(output_dir / 'seasonal_pollutant_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"增强可视化图表已保存到 {output_dir}")

def create_visualizations(soil_data, water_data):
    """Create visualization plots"""
    print("\n生成基础可视化图表...")

    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'outputs' / 'eda_plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Soil pollutant concentration distributions
    soil_parents = ['THIA', 'IMI', 'CLO']
    available_soil_parents = [p for p in soil_parents if p in soil_data.columns]

    if available_soil_parents:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('土壤新烟碱类污染物浓度分布', fontsize=16, fontweight='bold')

        for i, pollutant in enumerate(available_soil_parents):
            ax = axes[i]
            data = soil_data[pollutant]
            positive_data = data[data > 0]
            ax.hist(positive_data, bins=30, alpha=0.7, edgecolor='black', color='brown')
            ax.set_title(f'土壤 {pollutant} 浓度分布\n(ng/g, n={len(positive_data)})')
            ax.set_xlabel('浓度 (ng/g)')
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'均值: {positive_data.mean():.2f}\n中位数: {positive_data.median():.2f}\n检出率: {(len(positive_data)/len(data)*100):.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_dir / 'soil_pollutant_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Water pollutant concentration distributions
    water_parents = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN']
    available_water_parents = [p for p in water_parents if p in water_data.columns]

    if available_water_parents:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('水体新烟碱类污染物浓度分布', fontsize=16, fontweight='bold')

        for i, pollutant in enumerate(available_water_parents):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            data = water_data[pollutant]
            positive_data = data[data > 0]
            ax.hist(positive_data, bins=30, alpha=0.7, edgecolor='black', color='blue')
            ax.set_title(f'水体 {pollutant} 浓度分布\n(ng/L, n={len(positive_data)})')
            ax.set_xlabel('浓度 (ng/L)')
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'均值: {positive_data.mean():.2f}\n中位数: {positive_data.median():.2f}\n检出率: {(len(positive_data)/len(data)*100):.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Hide empty subplot
        if len(available_water_parents) < 6:
            axes[1, 2].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / 'water_pollutant_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Seasonal comparison of total pollutants
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('污染物季节性变化', fontsize=16, fontweight='bold')

    # Soil seasonal comparison
    if 'Season' in soil_data.columns:
        soil_parent_total = 'parentNNIs'
        soil_metabolite_total = 'mNNIs'

        # Parent pollutants by season
        season_order = ['Dry', 'Normal', 'Rainy']
        soil_season_data = [soil_data[soil_data['Season'] == season][soil_parent_total] for season in season_order]

        axes[0, 0].boxplot(soil_season_data, labels=season_order)
        axes[0, 0].set_title('土壤母体污染物季节变化\n(parentNNIs, ng/g)')
        axes[0, 0].set_ylabel('浓度 (ng/g)')
        axes[0, 0].grid(True, alpha=0.3)

        # Metabolites by season
        soil_season_meta = [soil_data[soil_data['Season'] == season][soil_metabolite_total] for season in season_order]
        axes[0, 1].boxplot(soil_season_meta, labels=season_order)
        axes[0, 1].set_title('土壤代谢物季节变化\n(mNNIs, ng/g)')
        axes[0, 1].set_ylabel('浓度 (ng/g)')
        axes[0, 1].grid(True, alpha=0.3)

    # Water seasonal comparison
    if 'Season' in water_data.columns:
        water_parent_total = 'parentNNIs'
        water_metabolite_total = 'mNNIs'

        # Parent pollutants by season
        water_season_data = [water_data[water_data['Season'] == season][water_parent_total] for season in season_order]
        axes[1, 0].boxplot(water_season_data, labels=season_order)
        axes[1, 0].set_title('水体母体污染物季节变化\n(parentNNIs, ng/L)')
        axes[1, 0].set_ylabel('浓度 (ng/L)')
        axes[1, 0].grid(True, alpha=0.3)

        # Metabolites by season
        water_season_meta = [water_data[water_data['Season'] == season][water_metabolite_total] for season in season_order]
        axes[1, 1].boxplot(water_season_meta, labels=season_order)
        axes[1, 1].set_title('水体代谢物季节变化\n(mNNIs, ng/L)')
        axes[1, 1].set_ylabel('浓度 (ng/L)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'seasonal_pollutant_changes.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation heatmap for main pollutants
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Soil pollutants correlation
    soil_pollutants = ['THIA', 'IMI', 'CLO', 'parentNNIs', 'IMI-UREA', 'DN-IMI', 'CLO-UREA', 'mNNIs']
    available_soil_pollutants = [p for p in soil_pollutants if p in soil_data.columns]
    if len(available_soil_pollutants) > 1:
        soil_corr = soil_data[available_soil_pollutants].corr()
        sns.heatmap(soil_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0], fmt='.2f')
        axes[0].set_title('土壤污染物相关性矩阵')

    # Water pollutants correlation
    water_pollutants = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs', 'IMI-UREA', 'DN-IMI', 'DM-ACE', 'CLO-UREA', 'mNNIs']
    available_water_pollutants = [p for p in water_pollutants if p in water_data.columns]
    if len(available_water_pollutants) > 1:
        water_corr = water_data[available_water_pollutants].corr()
        sns.heatmap(water_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1], fmt='.2f')
        axes[1].set_title('水体污染物相关性矩阵')

    plt.tight_layout()
    plt.savefig(output_dir / 'pollutant_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Geographic distribution
    if all(col in soil_data.columns for col in ['Lon', 'Lat']):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Soil sampling locations
        scatter = axes[0].scatter(soil_data['Lon'], soil_data['Lat'],
                                c=soil_data['parentNNIs'] if 'parentNNIs' in soil_data.columns else soil_data['THIA'],
                                cmap='viridis', alpha=0.6, s=50)
        axes[0].set_xlabel('经度')
        axes[0].set_ylabel('纬度')
        axes[0].set_title('土壤采样点分布')
        plt.colorbar(scatter, ax=axes[0], label='parentNNIs 浓度')

        # Water sampling locations
        scatter = axes[1].scatter(water_data['Lon'], water_data['Lat'],
                                c=water_data['parentNNIs'] if 'parentNNIs' in water_data.columns else water_data['THIA'],
                                cmap='viridis', alpha=0.6, s=50)
        axes[1].set_xlabel('经度')
        axes[1].set_ylabel('纬度')
        axes[1].set_title('水体采样点分布')
        plt.colorbar(scatter, ax=axes[1], label='parentNNIs 浓度')

        plt.tight_layout()
        plt.savefig(output_dir / 'geographic_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"图表已保存到 {output_dir}")

def summary_statistics(soil_data, water_data):
    """Generate comprehensive summary statistics"""
    print(f"\n{'='*60}")
    print("数据集综合统计摘要")
    print(f"{'='*60}")

    print(f"土壤数据集: {soil_data.shape[0]} 个样本, {soil_data.shape[1]} 个特征")
    print(f"水体数据集: {water_data.shape[0]} 个样本, {water_data.shape[1]} 个特征")

    # Pollutant comparison
    print(f"\n污染物浓度对比 (ng/g 或 ng/L):")
    print("-" * 50)

    if 'parentNNIs' in soil_data.columns and 'parentNNIs' in water_data.columns:
        soil_total = soil_data['parentNNIs'].describe()
        water_total = water_data['parentNNIs'].describe()

        print(f"{'指标':<15} {'土壤':<15} {'水体':<15}")
        print(f"{'-'*45}")
        print(f"{'样本数':<15} {soil_total['count']:<15.0f} {water_total['count']:<15.0f}")
        print(f"{'平均值':<15} {soil_total['mean']:<15.2f} {water_total['mean']:<15.2f}")
        print(f"{'标准差':<15} {soil_total['std']:<15.2f} {water_total['std']:<15.2f}")
        print(f"{'最小值':<15} {soil_total['min']:<15.2f} {water_total['min']:<15.2f}")
        print(f"{'25%分位数':<15} {soil_total['25%']:<15.2f} {water_total['25%']:<15.2f}")
        print(f"{'中位数':<15} {soil_total['50%']:<15.2f} {water_total['50%']:<15.2f}")
        print(f"{'75%分位数':<15} {soil_total['75%']:<15.2f} {water_total['75%']:<15.2f}")
        print(f"{'最大值':<15} {soil_total['max']:<15.2f} {water_total['max']:<15.2f}")

def save_statistics_tables(soil_data, water_data, soil_parent_stats, soil_metabolite_stats,
                           water_parent_stats, water_metabolite_stats, soil_detection, water_detection,
                           soil_veg=None, soil_agri=None, soil_econ=None, soil_water=None, soil_landuse=None,
                           water_agri=None, water_econ=None, water_water=None, water_landuse=None):
    """Save all statistical tables to files"""
    script_dir = Path(__file__).parent
    tables_dir = script_dir.parent / 'outputs' / 'eda_tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n保存统计表格到 {tables_dir}...")

    # 1. Soil parent pollutants statistics
    soil_parent_stats.to_csv(tables_dir / 'soil_parent_pollutants_stats.csv', float_format='%.2f')
    print(f"✓ 土壤母体污染物统计: {tables_dir / 'soil_parent_pollutants_stats.csv'}")

    # 2. Soil metabolites statistics
    soil_metabolite_stats.to_csv(tables_dir / 'soil_metabolites_stats.csv', float_format='%.2f')
    print(f"✓ 土壤代谢物统计: {tables_dir / 'soil_metabolites_stats.csv'}")

    # 3. Water parent pollutants statistics
    water_parent_stats.to_csv(tables_dir / 'water_parent_pollutants_stats.csv', float_format='%.2f')
    print(f"✓ 水体母体污染物统计: {tables_dir / 'water_parent_pollutants_stats.csv'}")

    # 4. Water metabolites statistics
    water_metabolite_stats.to_csv(tables_dir / 'water_metabolites_stats.csv', float_format='%.2f')
    print(f"✓ 水体代谢物统计: {tables_dir / 'water_metabolites_stats.csv'}")

    # 5. Detection rates
    detection_df = pd.DataFrame({
        '污染物': list(soil_detection.keys()) + list(water_detection.keys()),
        '检出率(%)': list(soil_detection.values()) + list(water_detection.values()),
        '介质': ['土壤'] * len(soil_detection) + ['水体'] * len(water_detection)
    })
    detection_df.to_csv(tables_dir / 'detection_rates.csv', index=False, float_format='%.1f')
    print(f"✓ 检出率统计: {tables_dir / 'detection_rates.csv'}")

    # 6. Seasonal statistics
    soil_seasonal_parent = soil_data.groupby('Season')['parentNNIs'].describe()
    soil_seasonal_parent.to_csv(tables_dir / 'soil_seasonal_parent_pollutants.csv', float_format='%.2f')
    print(f"✓ 土壤季节性母体污染物统计: {tables_dir / 'soil_seasonal_parent_pollutants.csv'}")

    soil_seasonal_metabolites = soil_data.groupby('Season')['mNNIs'].describe()
    soil_seasonal_metabolites.to_csv(tables_dir / 'soil_seasonal_metabolites.csv', float_format='%.2f')
    print(f"✓ 土壤季节性代谢物统计: {tables_dir / 'soil_seasonal_metabolites.csv'}")

    water_seasonal_parent = water_data.groupby('Season')['parentNNIs'].describe()
    water_seasonal_parent.to_csv(tables_dir / 'water_seasonal_parent_pollutants.csv', float_format='%.2f')
    print(f"✓ 水体季节性母体污染物统计: {tables_dir / 'water_seasonal_parent_pollutants.csv'}")

    water_seasonal_metabolites = water_data.groupby('Season')['mNNIs'].describe()
    water_seasonal_metabolites.to_csv(tables_dir / 'water_seasonal_metabolites.csv', float_format='%.2f')
    print(f"✓ 水体季节性代谢物统计: {tables_dir / 'water_seasonal_metabolites.csv'}")

    # 7. Environmental factors statistics
    soil_env_factors = ['pH', 'TN', 'TOC', 'Temp', 'Rain', 'EVI', 'NDVI', 'Alt', 'CC', 'BD']
    available_soil_env = [f for f in soil_env_factors if f in soil_data.columns]
    if available_soil_env:
        soil_data[available_soil_env].describe().to_csv(tables_dir / 'soil_environmental_factors.csv', float_format='%.3f')
        print(f"✓ 土壤环境因子统计: {tables_dir / 'soil_environmental_factors.csv'}")

    water_env_factors = ['pH', 'DO', 'COND', 'DOC', 'T_W', 'T_M', 'PREC', 'Alt']
    available_water_env = [f for f in water_env_factors if f in water_data.columns]
    if available_water_env:
        water_data[available_water_env].describe().to_csv(tables_dir / 'water_environmental_factors.csv', float_format='%.3f')
        print(f"✓ 水体环境因子统计: {tables_dir / 'water_environmental_factors.csv'}")

    # 8. Dataset summary
    summary_data = {
        '指标': ['样本数', '特征数', '缺失值数'],
        '土壤': [soil_data.shape[0], soil_data.shape[1], soil_data.isnull().sum().sum()],
        '水体': [water_data.shape[0], water_data.shape[1], water_data.isnull().sum().sum()]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(tables_dir / 'dataset_summary.csv', index=False)
    print(f"✓ 数据集概要: {tables_dir / 'dataset_summary.csv'}")

    # 9. Geographic statistics
    geo_data = {
        '指标': ['经度最小值', '经度最大值', '纬度最小值', '纬度最大值', '海拔最小值', '海拔最大值'],
        '土壤': [soil_data['Lon'].min(), soil_data['Lon'].max(),
                soil_data['Lat'].min(), soil_data['Lat'].max(),
                soil_data['Alt'].min(), soil_data['Alt'].max()],
        '水体': [water_data['Lon'].min(), water_data['Lon'].max(),
                water_data['Lat'].min(), water_data['Lat'].max(),
                water_data['Alt'].min(), water_data['Alt'].max()]
    }
    geo_df = pd.DataFrame(geo_data)
    geo_df.to_csv(tables_dir / 'geographic_summary.csv', index=False, float_format='%.3f')
    print(f"✓ 地理分布统计: {tables_dir / 'geographic_summary.csv'}")

    # 10. Vegetation indices (soil only)
    if soil_veg is not None:
        soil_veg.to_csv(tables_dir / 'soil_vegetation_indices.csv', float_format='%.3f')
        print(f"✓ 土壤植被指数统计: {tables_dir / 'soil_vegetation_indices.csv'}")

    # 11. Agricultural variables
    if soil_agri is not None:
        soil_agri.to_csv(tables_dir / 'soil_agricultural_variables.csv', float_format='%.2f')
        print(f"✓ 土壤农业变量统计: {tables_dir / 'soil_agricultural_variables.csv'}")

    if water_agri is not None:
        water_agri.to_csv(tables_dir / 'water_agricultural_variables.csv', float_format='%.2f')
        print(f"✓ 水体农业变量统计: {tables_dir / 'water_agricultural_variables.csv'}")

    # 12. Economic variables
    if soil_econ is not None:
        soil_econ.to_csv(tables_dir / 'soil_economic_variables.csv', float_format='%.2f')
        print(f"✓ 土壤经济变量统计: {tables_dir / 'soil_economic_variables.csv'}")

    if water_econ is not None:
        water_econ.to_csv(tables_dir / 'water_economic_variables.csv', float_format='%.2f')
        print(f"✓ 水体经济变量统计: {tables_dir / 'water_economic_variables.csv'}")

    # 13. Water usage variables
    if soil_water is not None:
        soil_water.to_csv(tables_dir / 'soil_water_usage.csv', float_format='%.2f')
        print(f"✓ 土壤用水变量统计: {tables_dir / 'soil_water_usage.csv'}")

    if water_water is not None:
        water_water.to_csv(tables_dir / 'water_water_usage.csv', float_format='%.2f')
        print(f"✓ 水体用水变量统计: {tables_dir / 'water_water_usage.csv'}")

    # 14. Land use analysis
    if soil_landuse is not None:
        soil_landuse.to_csv(tables_dir / 'soil_landuse_distribution.csv')
        soil_landuse_pct = (soil_landuse / len(soil_data)) * 100
        soil_landuse_pct.to_csv(tables_dir / 'soil_landuse_percentages.csv', float_format='%.1f')
        print(f"✓ 土壤土地利用分布: {tables_dir / 'soil_landuse_distribution.csv'}")

    if water_landuse is not None:
        water_landuse.to_csv(tables_dir / 'water_landuse_distribution.csv')
        water_landuse_pct = (water_landuse / len(water_data)) * 100
        water_landuse_pct.to_csv(tables_dir / 'water_landuse_percentages.csv', float_format='%.1f')
        print(f"✓ 水体土地利用分布: {tables_dir / 'water_landuse_distribution.csv'}")

    return tables_dir

def enhanced_pollutant_analysis(df, dataset_name):
    """Enhanced pollutant analysis with metabolite/parent ratios and composition"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 增强污染物分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        parent_pollutants = ['THIA', 'IMI', 'CLO']
        metabolite_pollutants = ['IMI-UREA', 'DN-IMI', 'CLO-UREA']
        total_parent = 'parentNNIs'
        total_metabolite = 'mNNIs'
    else:  # 水体
        parent_pollutants = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN']
        metabolite_pollutants = ['IMI-UREA', 'DN-IMI', 'DM-ACE', 'CLO-UREA']
        total_parent = 'parentNNIs'
        total_metabolite = 'mNNIs'

    # 1. Pollutant composition analysis
    print("\n污染物组成分析:")
    available_parents = [p for p in parent_pollutants if p in df.columns]
    if available_parents:
        composition = df[available_parents].sum()
        composition_pct = (composition / composition.sum() * 100).round(2)
        composition_df = pd.DataFrame({
            '浓度': composition,
            '占比(%)': composition_pct
        }).sort_values('浓度', ascending=False)
        print(dataframe_to_markdown(composition_df, "母体污染物组成", 2))

    # 2. Metabolite/Parent ratios
    print("\n代谢物/母体比值分析:")
    ratios = {}
    for parent, metabolite in [('THIA', None), ('IMI', 'IMI-UREA'), ('CLO', 'CLO-UREA')]:
        if parent in df.columns:
            parent_conc = df[parent]
            if metabolite and metabolite in df.columns:
                metabolite_conc = df[metabolite]
                ratio = metabolite_conc[parent_conc > 0] / parent_conc[parent_conc > 0]
                ratios[f'{metabolite}/{parent}'] = {
                    '平均值': ratio.mean(),
                    '中位数': ratio.median(),
                    '标准差': ratio.std(),
                    '有效样本数': len(ratio)
                }

    if ratios:
        ratio_df = pd.DataFrame(ratios).T
        print(dataframe_to_markdown(ratio_df, "代谢物/母体比值统计", 3))

    # 3. Total pollutant ratios
    if total_parent in df.columns and total_metabolite in df.columns:
        total_ratio = df[total_metabolite] / (df[total_parent] + df[total_metabolite])
        total_ratio = total_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"\n代谢物占总污染物比例:")
        print(f"平均值: {total_ratio.mean():.3f}")
        print(f"中位数: {total_ratio.median():.3f}")
        print(f"标准差: {total_ratio.std():.3f}")
        print(f"范围: {total_ratio.min():.3f} ~ {total_ratio.max():.3f}")

    return composition_pct if available_parents else None, ratios

def environmental_correlation_analysis(df, dataset_name):
    """Analyze correlations between pollutants and environmental factors"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 环境因子关联分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        pollutants = ['THIA', 'IMI', 'CLO', 'parentNNIs', 'mNNIs']
        env_factors = ['pH', 'TN', 'TOC', 'Temp', 'Rain', 'EVI', 'NDVI', 'Alt']
        agri_factors = ['FER', 'PES', 'FERA', 'PESA']
    else:  # 水体
        pollutants = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs', 'mNNIs']
        env_factors = ['pH', 'DO', 'COND', 'DOC', 'T_W', 'T_M', 'PREC', 'Alt']
        agri_factors = ['FER', 'PES', 'FERPER', 'PESPER']

    # Find available columns
    available_pollutants = [p for p in pollutants if p in df.columns]
    available_env = [e for e in env_factors if e in df.columns]
    available_agri = [a for a in agri_factors if a in df.columns]

    analysis_vars = available_pollutants + available_env + available_agri

    if len(analysis_vars) > 1:
        # 1. Correlation matrix
        corr_matrix = df[analysis_vars].corr()

        # 2. Focus on pollutant correlations
        pollutant_corr = corr_matrix.loc[available_pollutants, available_env + available_agri]

        print("\n污染物与环境因子相关性矩阵:")
        print(dataframe_to_markdown(pollutant_corr, "污染物-环境因子相关性", 3))

        # 3. Strong correlations (|r| > 0.3)
        strong_corr = []
        for poll in available_pollutants:
            for factor in available_env + available_agri:
                corr_val = corr_matrix.loc[poll, factor]
                if abs(corr_val) > 0.3:
                    strong_corr.append({
                        '污染物': poll,
                        '环境因子': factor,
                        '相关系数': round(corr_val, 3),
                        '相关强度': '强正相关' if corr_val > 0.3 else '强负相关'
                    })

        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr)
            print(f"\n强相关性汇总 (|r| > 0.3):")
            print(dataframe_to_markdown(strong_corr_df, "强相关性", 3))

        return corr_matrix, strong_corr_df if strong_corr else None

    return None, None

def agricultural_impact_analysis(df, dataset_name):
    """Analyze the impact of agricultural practices on pollutant levels"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 农业活动影响分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        pollutants = ['THIA', 'IMI', 'CLO', 'parentNNIs', 'mNNIs']
        agri_vars = ['FER', 'PES', 'FERA', 'PESA', 'CC', 'BD']
        crop_vars = ['MC', 'VE', 'FR', 'MCP', 'VEP', 'FRP']
    else:  # 水体
        pollutants = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs', 'mNNIs']
        agri_vars = ['FER', 'PES', 'FERPER', 'PESPER', 'AMP']
        crop_vars = ['WO', 'CO', 'VO', 'FOP']

    available_pollutants = [p for p in pollutants if p in df.columns]
    available_agri = [a for a in agri_vars if a in df.columns]
    available_crops = [c for c in crop_vars if c in df.columns]

    # 1. High vs low fertilizer/pesticide usage analysis
    if available_agri:
        print(f"\n化肥农药使用量分组分析:")
        for var in ['FER', 'PES']:
            if var in df.columns:
                median_val = df[var].median()
                high_group = df[df[var] > median_val]
                low_group = df[df[var] <= median_val]

                print(f"\n{var} 高使用组 vs 低使用组 (分割值: {median_val:.2f}):")
                for poll in available_pollutants:
                    high_mean = high_group[poll].mean()
                    low_mean = low_group[poll].mean()
                    if high_mean > 0 or low_mean > 0:
                        ratio = high_mean / low_mean if low_mean > 0 else np.inf
                        print(f"  {poll}: 高组 {high_mean:.3f} vs 低组 {low_mean:.3f} (比值: {ratio:.2f})")

    # 2. Crop yield correlation with pollutants
    if available_crops:
        print(f"\n作物产量与污染物相关性:")
        crop_pollutant_corr = {}
        for crop in available_crops:
            for poll in available_pollutants:
                corr = df[[crop, poll]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    crop_pollutant_corr[f'{crop}-{poll}'] = corr

        if crop_pollutant_corr:
            corr_df = pd.DataFrame(list(crop_pollutant_corr.items()),
                                 columns=['作物-污染物', '相关系数'])
            corr_df = corr_df.sort_values('相关系数', key=abs, ascending=False)
            print(dataframe_to_markdown(corr_df.head(10), "作物产量-污染物相关性TOP10", 3))

    # 3. Agricultural intensity zones
    if 'FER' in df.columns and 'PES' in df.columns:
        # Create agricultural intensity index
        fer_norm = (df['FER'] - df['FER'].min()) / (df['FER'].max() - df['FER'].min())
        pes_norm = (df['PES'] - df['PES'].min()) / (df['PES'].max() - df['PES'].min())
        agri_intensity = (fer_norm + pes_norm) / 2

        df_temp = df.copy()
        df_temp['agri_intensity'] = agri_intensity

        # Divide into quartiles
        df_temp['intensity_group'] = pd.qcut(agri_intensity, 4, labels=['低', '中低', '中高', '高'])

        print(f"\n农业强度分区污染物分析:")
        intensity_stats = df_temp.groupby('intensity_group')[available_pollutants].mean()
        print(dataframe_to_markdown(intensity_stats, "不同农业强度区污染物平均浓度", 3))

    return None

def economic_impact_analysis(df, dataset_name):
    """Analyze the relationship between economic factors and pollutant distribution"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 经济因素影响分析")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        pollutants = ['THIA', 'IMI', 'CLO', 'parentNNIs', 'mNNIs']
        econ_vars = ['UR', 'GAO', 'GDP per capita', 'UI', 'RI', 'PR', 'SR', 'TR']
    else:  # 水体
        pollutants = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs', 'mNNIs']
        econ_vars = ['Urban', 'GDP', 'UI', 'RI', 'PR', 'SR', 'TR', 'POP_TOT']

    available_pollutants = [p for p in pollutants if p in df.columns]
    available_econ = [e for e in econ_vars if e in df.columns]

    # 1. Economic development level vs pollutant concentration
    if 'GDP per capita' in df.columns or 'GDP' in df.columns:
        gdp_col = 'GDP per capita' if 'GDP per capita' in df.columns else 'GDP'
        gdp_median = df[gdp_col].median()

        high_gdp = df[df[gdp_col] > gdp_median]
        low_gdp = df[df[gdp_col] <= gdp_median]

        print(f"\n经济发展水平分组分析 (基于{gdp_col}):")
        print(f"高GDP组样本数: {len(high_gdp)}, 低GDP组样本数: {len(low_gdp)}")

        for poll in available_pollutants:
            high_mean = high_gdp[poll].mean()
            low_mean = low_gdp[poll].mean()
            if high_mean > 0 or low_mean > 0:
                print(f"  {poll}: 高GDP {high_mean:.3f} vs 低GDP {low_mean:.3f}")

    # 2. Urbanization rate analysis
    if 'UR' in df.columns or 'Urban' in df.columns:
        urban_col = 'UR' if 'UR' in df.columns else 'Urban'
        urban_quartiles = pd.qcut(df[urban_col], 4, labels=['低城镇化', '中低城镇化', '中高城镇化', '高城镇化'])

        df_temp = df.copy()
        df_temp['urban_level'] = urban_quartiles

        print(f"\n城镇化水平分区污染物分析:")
        urban_stats = df_temp.groupby('urban_level')[available_pollutants].mean()
        print(dataframe_to_markdown(urban_stats, "不同城镇化水平污染物平均浓度", 3))

    # 3. Income disparity analysis
    if 'UI' in df.columns and 'RI' in df.columns:
        df_temp = df.copy()
        df_temp['income_gap'] = df_temp['UI'] - df_temp['RI']
        df_temp['income_gap_level'] = pd.qcut(df_temp['income_gap'], 3,
                                             labels=['城乡差距小', '城乡差距中', '城乡差距大'])

        print(f"\n城乡收入差距与污染物分析:")
        income_gap_stats = df_temp.groupby('income_gap_level')[available_pollutants].mean()
        print(dataframe_to_markdown(income_gap_stats, "不同收入差距区污染物平均浓度", 3))

    return None

def professional_analysis_indicators(df, dataset_name):
    """Professional analysis indicators including risk assessment and trend analysis"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 专业分析指标")
    print(f"{'='*50}")

    if dataset_name == "土壤":
        pollutants = ['THIA', 'IMI', 'CLO', 'parentNNIs', 'mNNIs']
        env_factors = ['pH', 'TN', 'TOC', 'Temp', 'Rain']
    else:  # 水体
        pollutants = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs', 'mNNIs']
        env_factors = ['pH', 'DO', 'COND', 'DOC', 'T_W', 'T_M']

    available_pollutants = [p for p in pollutants if p in df.columns]

    # 1. Detection rate confidence intervals
    print("\n1. 检出率置信区间 (95%):")
    detection_ci = {}
    for poll in available_pollutants:
        detected = (df[poll] > 0).sum()
        total = len(df)
        detection_rate = detected / total

        # Wilson confidence interval for proportion
        z = 1.96  # 95% confidence
        p = detection_rate
        n = total
        denominator = 1 + (z**2)/n
        centre_adjusted = p + (z**2)/(2*n)
        margin = z * np.sqrt((p*(1-p) + (z**2)/(4*n))/n)

        ci_lower = (centre_adjusted - margin) / denominator
        ci_upper = (centre_adjusted + margin) / denominator

        detection_ci[poll] = {
            '检出率(%)': detection_rate * 100,
            'CI下限(%)': max(0, ci_lower * 100),
            'CI上限(%)': min(100, ci_upper * 100),
            '检出数': detected,
            '总样本数': total
        }

    ci_df = pd.DataFrame(detection_ci).T
    print(dataframe_to_markdown(ci_df, "检出率置信区间", 2))

    # 2. Environmental risk assessment
    print("\n2. 环境风险评估:")

    # Risk thresholds (example values based on environmental guidelines)
    risk_thresholds = {
        'soil': {'low': 10, 'medium': 50, 'high': 200},      # ng/g for soil
        'water': {'low': 1, 'medium': 10, 'high': 100}       # ng/L for water
    }

    medium_type = 'soil' if dataset_name == "土壤" else 'water'
    thresholds = risk_thresholds[medium_type]

    risk_assessment = {}
    for poll in available_pollutants:
        concentrations = df[poll][df[poll] > 0]  # Only detected samples

        if len(concentrations) > 0:
            low_risk = (concentrations <= thresholds['low']).sum()
            medium_risk = ((concentrations > thresholds['low']) &
                          (concentrations <= thresholds['medium'])).sum()
            high_risk = ((concentrations > thresholds['medium']) &
                        (concentrations <= thresholds['high'])).sum()
            very_high_risk = (concentrations > thresholds['high']).sum()

            total_detected = len(concentrations)

            risk_assessment[poll] = {
                '低风险(%)': (low_risk / total_detected) * 100 if total_detected > 0 else 0,
                '中风险(%)': (medium_risk / total_detected) * 100 if total_detected > 0 else 0,
                '高风险(%)': (high_risk / total_detected) * 100 if total_detected > 0 else 0,
                '极高风险(%)': (very_high_risk / total_detected) * 100 if total_detected > 0 else 0,
                '检出样本数': total_detected
            }

    if risk_assessment:
        risk_df = pd.DataFrame(risk_assessment).T
        print(dataframe_to_markdown(risk_df, "环境风险等级分布", 2))

    # 3. Seasonal trend analysis
    if 'Season' in df.columns:
        print("\n3. 季节性趋势分析:")
        seasonal_trends = {}
        season_order = {'Dry': 0, 'Normal': 1, 'Rainy': 2}

        for poll in available_pollutants:
            seasonal_means = []
            seasonal_seasons = []

            for season in ['Dry', 'Normal', 'Rainy']:
                season_data = df[df['Season'] == season][poll].dropna()
                if len(season_data) > 0:
                    seasonal_means.append(season_data.mean())
                    seasonal_seasons.append(season_order[season])

            if len(seasonal_means) >= 2:
                # Calculate trend (linear regression on seasonal means)
                slope, intercept, r_value, p_value, std_err = stats.linregress(seasonal_seasons, seasonal_means)

                # Calculate seasonal change rate
                if len(seasonal_means) == 3:
                    dry_to_rainy = ((seasonal_means[2] - seasonal_means[0]) / seasonal_means[0] * 100) if seasonal_means[0] > 0 else 0
                else:
                    dry_to_rainy = 0

                seasonal_trends[poll] = {
                    '趋势斜率': slope,
                    '相关系数': r_value,
                    'P值': p_value,
                    '显著性': '显著' if p_value < 0.05 else '不显著',
                    '旱季到雨季变化(%)': dry_to_rainy
                }

        if seasonal_trends:
            trend_df = pd.DataFrame(seasonal_trends).T
            print(dataframe_to_markdown(trend_df, "季节性趋势分析", 4))

    # 4. Statistical significance testing
    print("\n4. 统计显著性检验:")

    # Test differences between land use types if available
    if 'landuse' in df.columns or 'Landuse' in df.columns:
        landuse_col = 'landuse' if 'landuse' in df.columns else 'Landuse'
        landuse_types = df[landuse_col].unique()

        if len(landuse_types) >= 2:
            significance_results = []

            for poll in available_pollutants:
                for landuse_pair in itertools.combinations(landuse_types, 2):
                    group1 = df[df[landuse_col] == landuse_pair[0]][poll].dropna()
                    group2 = df[df[landuse_col] == landuse_pair[1]][poll].dropna()

                    # Only test positive values (detected concentrations)
                    group1 = group1[group1 > 0]
                    group2 = group2[group2 > 0]

                    if len(group1) > 2 and len(group2) > 2:
                        # Mann-Whitney U test (non-parametric)
                        stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')

                        # Effect size (Cliff's delta)
                        n1, n2 = len(group1), len(group2)
                        all_data = np.concatenate([group1, group2])
                        ranks = stats.rankdata(all_data)
                        rank_sum1 = ranks[:n1].sum()

                        u1 = rank_sum1 - n1*(n1+1)/2
                        u2 = n1*n2 - u1
                        cliffs_delta = (u1 - u2) / (n1*n2)

                        significance_results.append({
                            '污染物': poll,
                            '土地利用对比': f'{landuse_pair[0]} vs {landuse_pair[1]}',
                            '样本数1': len(group1),
                            '样本数2': len(group2),
                            'P值': p_val,
                            '显著性': '显著' if p_val < 0.05 else '不显著',
                            '效应量': cliffs_delta
                        })

            if significance_results:
                sig_df = pd.DataFrame(significance_results)
                sig_df = sig_df[sig_df['显著性'] == '显著'].sort_values('P值')
                if len(sig_df) > 0:
                    print(dataframe_to_markdown(sig_df.head(10), "显著差异检验结果 (P<0.05)", 4))

    return detection_ci, risk_assessment, seasonal_trends if 'Season' in df.columns else None

def spatial_autocorrelation_analysis(df, dataset_name):
    """Spatial autocorrelation analysis using Moran's I and spatial patterns"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 空间自相关分析")
    print(f"{'='*50}")

    if not all(col in df.columns for col in ['Lon', 'Lat']):
        print("缺少地理坐标信息，无法进行空间自相关分析")
        return None

    pollutants = ['parentNNIs'] if dataset_name == "土壤" else ['parentNNIs']
    available_pollutants = [p for p in pollutants if p in df.columns]

    if len(available_pollutants) == 0:
        print("没有找到合适的污染物数据进行空间分析")
        return None

    spatial_results = {}

    for poll in available_pollutants:
        # Filter positive values only
        df_positive = df[df[poll] > 0].copy()

        if len(df_positive) < 10:
            print(f"{poll}: 检出样本数不足 (n={len(df_positive)})，跳过空间分析")
            continue

        print(f"\n{poll} 空间自相关分析:")
        print(f"分析样本数: {len(df_positive)}")

        # 1. Calculate spatial weights (distance-based)
        coords = df_positive[['Lon', 'Lat']].values

        # Calculate distance matrix
        distances = pdist(coords, metric='euclidean')
        dist_matrix = squareform(distances)

        # Set diagonal to infinity to avoid self-comparison
        np.fill_diagonal(dist_matrix, np.inf)

        # Find nearest neighbors (using k=5 for robustness)
        k_neighbors = min(5, len(df_positive) - 1)
        nearest_neighbors = np.argsort(dist_matrix, axis=1)[:, :k_neighbors]

        # Create spatial weights matrix (row-standardized)
        n = len(df_positive)
        W = np.zeros((n, n))

        for i in range(n):
            W[i, nearest_neighbors[i]] = 1
            # Row standardization
            if W[i].sum() > 0:
                W[i] = W[i] / W[i].sum()

        # 2. Calculate Moran's I
        values = df_positive[poll].values
        values_std = (values - values.mean()) / values.std()

        # Moran's I formula
        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * values_std[i] * values_std[j]
            denominator += values_std[i] ** 2

        morans_I = numerator / denominator

        # Expected value under null hypothesis
        expected_I = -1 / (n - 1)

        # Variance calculation (simplified)
        S1 = 0
        S2 = 0
        for i in range(n):
            for j in range(n):
                S1 += (W[i, j] + W[j, i]) ** 2
                S2 += (W[i, :] + W[:, j]).sum() ** 2

        var_I = (n * S1 - S2) / ((n - 1) * (n - 2) * (n - 3) * denominator ** 2)

        # Z-score and p-value
        if var_I > 0:
            z_score = (morans_I - expected_I) / np.sqrt(var_I)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1

        # 3. Spatial clustering analysis
        spatial_clusters = []

        # High-High and Low-Low clusters
        values_mean = values.mean()

        for i in range(n):
            neighbors_idx = nearest_neighbors[i]
            neighbors_values = values[neighbors_idx]

            # Local spatial autocorrelation (simplified)
            local_sum = sum(W[i, j] * values_std[j] for j in range(n))

            if values_std[i] > 0 and local_sum > 0:
                cluster_type = "High-High"
            elif values_std[i] < 0 and local_sum < 0:
                cluster_type = "Low-Low"
            elif values_std[i] > 0 and local_sum < 0:
                cluster_type = "High-Low"
            elif values_std[i] < 0 and local_sum > 0:
                cluster_type = "Low-High"
            else:
                cluster_type = "Not Significant"

            spatial_clusters.append({
                'Lon': df_positive.iloc[i]['Lon'],
                'Lat': df_positive.iloc[i]['Lat'],
                'Value': values[i],
                'Cluster_Type': cluster_type,
                'Local_I': local_sum
            })

        cluster_df = pd.DataFrame(spatial_clusters)
        cluster_counts = cluster_df['Cluster_Type'].value_counts()

        # 4. Spatial statistics summary
        print(f"Moran\'s I: {morans_I:.4f}")
        print(f"期望值: {expected_I:.4f}")
        print(f"Z分数: {z_score:.4f}")
        print(f"P值: {p_value:.4f}")
        print(f"显著性: {'显著' if p_value < 0.05 else '不显著'}")
        print(f"空间聚集性: {'聚集' if morans_I > expected_I else '离散' if morans_I < expected_I else '随机'}")

        print(f"\n空间聚类分布:")
        print(cluster_counts)

        # 5. Spatial range analysis
        # Calculate average distance between high concentration points
        high_values = df_positive[df_positive[poll] > df_positive[poll].quantile(0.75)]
        if len(high_values) > 1:
            high_coords = high_values[['Lon', 'Lat']].values
            high_distances = pdist(high_coords, metric='euclidean')
            avg_distance = high_distances.mean()
            print(f"\n高浓度点间平均距离: {avg_distance:.4f} 度")

        spatial_results[poll] = {
            'Morans_I': morans_I,
            'Expected_I': expected_I,
            'Z_Score': z_score,
            'P_Value': p_value,
            'Significant': p_value < 0.05,
            'Cluster_Counts': cluster_counts.to_dict(),
            'Sample_Size': len(df_positive)
        }

    return spatial_results

def data_quality_assessment(df, dataset_name):
    """Comprehensive data quality assessment including outliers, completeness, and consistency"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} 数据质量评估")
    print(f"{'='*50}")

    quality_results = {}

    # 1. Outlier detection
    print("\n1. 异常值检测:")
    outliers = {}

    if dataset_name == "土壤":
        key_variables = ['THIA', 'IMI', 'CLO', 'parentNNIs', 'mNNIs', 'pH', 'TN', 'TOC', 'Temp', 'Rain']
    else:  # 水体
        key_variables = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs', 'mNNIs', 'pH', 'DO', 'COND']

    available_vars = [var for var in key_variables if var in df.columns]

    for var in available_vars:
        # Only analyze detected values for pollutants
        if var in ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs', 'parentNNIs', 'mNNIs', 'IMI-UREA', 'DN-IMI', 'CLO-UREA', 'DM-ACE']:
            data = df[df[var] > 0][var]
        else:
            data = df[var].dropna()

        if len(data) > 3:
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]

            # Z-score method
            z_scores = np.abs((data - data.mean()) / data.std())
            z_outliers = data[z_scores > 3]

            outliers[var] = {
                '总样本数': len(df),
                '有效样本数': len(data),
                'IQR异常值数': len(iqr_outliers),
                'IQR异常值比例(%)': (len(iqr_outliers) / len(data)) * 100,
                'Z-score异常值数': len(z_outliers),
                'Z-score异常值比例(%)': (len(z_outliers) / len(data)) * 100,
                '最小值': data.min(),
                '最大值': data.max(),
                'IQR下界': lower_bound,
                'IQR上界': upper_bound
            }

    if outliers:
        outlier_df = pd.DataFrame(outliers).T
        print(dataframe_to_markdown(outlier_df, "异常值检测结果", 2))

        # Variables with high outlier rates (>5%)
        high_outlier_vars = outlier_df[outlier_df['IQR异常值比例(%)'] > 5]
        if len(high_outlier_vars) > 0:
            print(f"\n高异常值率变量 (>5%):")
            print(dataframe_to_markdown(high_outlier_vars, "需要关注的异常值", 2))

    quality_results['outliers'] = outliers

    # 2. Data completeness analysis
    print("\n2. 数据完整性分析:")

    # Missing value patterns
    missing_analysis = {}
    total_samples = len(df)

    # Overall missing rate
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    overall_missing_rate = (total_missing / total_cells) * 100

    print(f"总体缺失率: {overall_missing_rate:.2f}%")
    print(f"总缺失值数: {total_missing} / {total_cells}")

    # Missing by column
    missing_by_col = df.isnull().sum()
    missing_by_col_pct = (missing_by_col / total_samples) * 100

    # Categorize missing rates
    no_missing = missing_by_col_pct[missing_by_col_pct == 0]
    low_missing = missing_by_col_pct[(missing_by_col_pct > 0) & (missing_by_col_pct <= 5)]
    moderate_missing = missing_by_col_pct[(missing_by_col_pct > 5) & (missing_by_col_pct <= 20)]
    high_missing = missing_by_col_pct[missing_by_col_pct > 20]

    print(f"\n缺失值分布:")
    print(f"无缺失值变量: {len(no_missing)} 个")
    print(f"低缺失率变量 (≤5%): {len(low_missing)} 个")
    print(f"中等缺失率变量 (5-20%): {len(moderate_missing)} 个")
    print(f"高缺失率变量 (>20%): {len(high_missing)} 个")

    if len(high_missing) > 0:
        print(f"\n高缺失率变量:")
        high_missing_df = pd.DataFrame({
            '变量名': high_missing.index,
            '缺失率(%)': high_missing.values,
            '缺失数量': missing_by_col[high_missing.index].values
        })
        print(dataframe_to_markdown(high_missing_df, "高缺失率变量", 2))

    # Missing value pattern analysis
    print(f"\n缺失值模式分析:")
    missing_patterns = df.isnull().value_counts()
    print(f"发现 {len(missing_patterns)} 种缺失值模式")
    print("主要缺失模式:")
    for pattern, count in missing_patterns.head(5).items():
        missing_vars = [df.columns[i] for i, is_missing in enumerate(pattern) if is_missing]
        if missing_vars:
            print(f"  模式 {list(missing_vars)}: {count} 个样本 ({count/total_samples*100:.1f}%)")

    quality_results['missing'] = {
        'overall_rate': overall_missing_rate,
        'by_column': missing_by_col_pct.to_dict(),
        'patterns': missing_patterns.to_dict()
    }

    # 3. Data consistency checks
    print("\n3. 数据一致性检查:")

    consistency_issues = []

    # Check pollutant total consistency
    if dataset_name == "土壤":
        if all(col in df.columns for col in ['THIA', 'IMI', 'CLO', 'parentNNIs']):
            # parentNNIs should be approximately equal to sum of individual NNIs
            calculated_sum = df[['THIA', 'IMI', 'CLO']].sum(axis=1)
            reported_total = df['parentNNIs']

            # Check for major discrepancies (>20% difference)
            mask = (calculated_sum > 0) & (reported_total > 0)
            if mask.sum() > 0:
                ratio = reported_total[mask] / calculated_sum[mask]
                discrepancies = ratio[(ratio < 0.8) | (ratio > 1.2)]

                if len(discrepancies) > 0:
                    consistency_issues.append({
                        '检查类型': '母体污染物总量一致性',
                        '问题样本数': len(discrepancies),
                        '问题比例(%)': (len(discrepancies) / mask.sum()) * 100,
                        '描述': f'parentNNIs与单项加总差异>20%'
                    })

        # Check metabolite consistency
        if all(col in df.columns for col in ['IMI-UREA', 'DN-IMI', 'CLO-UREA', 'mNNIs']):
            metabolite_sum = df[['IMI-UREA', 'DN-IMI', 'CLO-UREA']].sum(axis=1)
            metabolite_total = df['mNNIs']

            mask = (metabolite_sum > 0) & (metabolite_total > 0)
            if mask.sum() > 0:
                ratio = metabolite_total[mask] / metabolite_sum[mask]
                discrepancies = ratio[(ratio < 0.8) | (ratio > 1.2)]

                if len(discrepancies) > 0:
                    consistency_issues.append({
                        '检查类型': '代谢物总量一致性',
                        '问题样本数': len(discrepancies),
                        '问题比例(%)': (len(discrepancies) / mask.sum()) * 100,
                        '描述': f'mNNIs与单项代谢物加总差异>20%'
                    })

    else:  # 水体
        if all(col in df.columns for col in ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs']):
            calculated_sum = df[['THIA', 'IMI', 'CLO', 'ACE', 'DIN']].sum(axis=1)
            reported_total = df['parentNNIs']

            mask = (calculated_sum > 0) & (reported_total > 0)
            if mask.sum() > 0:
                ratio = reported_total[mask] / calculated_sum[mask]
                discrepancies = ratio[(ratio < 0.8) | (ratio > 1.2)]

                if len(discrepancies) > 0:
                    consistency_issues.append({
                        '检查类型': '母体污染物总量一致性',
                        '问题样本数': len(discrepancies),
                        '问题比例(%)': (len(discrepancies) / mask.sum()) * 100,
                        '描述': f'parentNNIs与单项加总差异>20%'
                    })

    # Check for negative values in pollutant concentrations
    pollutant_cols = [col for col in df.columns if col in ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs', 'parentNNIs', 'mNNIs', 'IMI-UREA', 'DN-IMI', 'CLO-UREA', 'DM-ACE']]
    for col in pollutant_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                consistency_issues.append({
                    '检查类型': '负值检测',
                    '变量': col,
                    '问题样本数': negative_count,
                    '问题比例(%)': (negative_count / total_samples) * 100,
                    '描述': f'{col}存在负值'
                })

    # Check for unrealistic values
    if dataset_name == "土壤":
        # pH should typically be between 3-10 for soil
        if 'pH' in df.columns:
            unrealistic_ph = df[(df['pH'] < 3) | (df['pH'] > 10)].dropna(subset=['pH'])
            if len(unrealistic_ph) > 0:
                consistency_issues.append({
                    '检查类型': '异常pH值',
                    '问题样本数': len(unrealistic_ph),
                    '问题比例(%)': (len(unrealistic_ph) / df['pH'].dropna().shape[0]) * 100,
                    '描述': '土壤pH值超出合理范围(3-10)'
                })

        # Temperature in Celsius should be reasonable
        if 'Temp' in df.columns:
            unrealistic_temp = df[(df['Temp'] < -20) | (df['Temp'] > 60)].dropna(subset=['Temp'])
            if len(unrealistic_temp) > 0:
                consistency_issues.append({
                    '检查类型': '异常温度值',
                    '问题样本数': len(unrealistic_temp),
                    '问题比例(%)': (len(unrealistic_temp) / df['Temp'].dropna().shape[0]) * 100,
                    '描述': '温度值超出合理范围(-20°C to 60°C)'
                })

    if consistency_issues:
        consistency_df = pd.DataFrame(consistency_issues)
        print(dataframe_to_markdown(consistency_df, "数据一致性问题", 2))
    else:
        print("未发现明显的数据一致性问题")

    quality_results['consistency'] = consistency_issues

    # 4. Data quality score
    print("\n4. 数据质量评分:")

    # Calculate quality metrics
    completeness_score = max(0, 100 - overall_missing_rate)
    outlier_score = max(0, 100 - outlier_df['IQR异常值比例(%)'].mean() if len(outlier_df) > 0 else 100)
    consistency_score = max(0, 100 - len(consistency_issues) * 10)  # Deduct 10 points per issue

    overall_quality_score = (completeness_score + outlier_score + consistency_score) / 3

    print(f"完整性评分: {completeness_score:.1f}/100")
    print(f"异常值评分: {outlier_score:.1f}/100")
    print(f"一致性评分: {consistency_score:.1f}/100")
    print(f"总体质量评分: {overall_quality_score:.1f}/100")

    # Quality level
    if overall_quality_score >= 90:
        quality_level = "优秀"
    elif overall_quality_score >= 80:
        quality_level = "良好"
    elif overall_quality_score >= 70:
        quality_level = "一般"
    elif overall_quality_score >= 60:
        quality_level = "较差"
    else:
        quality_level = "很差"

    print(f"数据质量等级: {quality_level}")

    quality_results['quality_score'] = {
        'completeness': completeness_score,
        'outlier': outlier_score,
        'consistency': consistency_score,
        'overall': overall_quality_score,
        'level': quality_level
    }

    return quality_results

def comparative_analysis(soil_data, water_data):
    """Comprehensive comparative analysis between soil and water datasets"""
    print(f"\n{'='*60}")
    print("土壤-水体污染物对比分析")
    print(f"{'='*60}")

    comparison_results = {}

    # 1. Detection rate comparison
    print("\n1. 检出率对比:")

    # Common pollutants between soil and water
    common_pollutants = ['THIA', 'IMI', 'CLO']
    available_common = [p for p in common_pollutants if p in soil_data.columns and p in water_data.columns]

    detection_comparison = {}
    for poll in available_common:
        soil_detection = (soil_data[poll] > 0).mean() * 100
        water_detection = (water_data[poll] > 0).mean() * 100

        detection_comparison[poll] = {
            '土壤检出率(%)': soil_detection,
            '水体检出率(%)': water_detection,
            '差异': soil_detection - water_detection,
            '土壤样本数': len(soil_data),
            '水体样本数': len(water_data)
        }

    if detection_comparison:
        detection_df = pd.DataFrame(detection_comparison).T
        print(dataframe_to_markdown(detection_df, "土壤-水体污染物检出率对比", 2))

    comparison_results['detection_rates'] = detection_comparison

    # 2. Concentration level comparison
    print("\n2. 浓度水平对比:")

    concentration_comparison = {}
    for poll in available_common:
        soil_positive = soil_data[soil_data[poll] > 0][poll]
        water_positive = water_data[water_data[poll] > 0][poll]

        if len(soil_positive) > 0 and len(water_positive) > 0:
            # Statistical test for difference in concentrations
            stat, p_value = stats.mannwhitneyu(soil_positive, water_positive, alternative='two-sided')

            concentration_comparison[poll] = {
                '土壤平均浓度(ng/g)': soil_positive.mean(),
                '土壤中位数(ng/g)': soil_positive.median(),
                '水体平均浓度(ng/L)': water_positive.mean(),
                '水体中位数(ng/L)': water_positive.median(),
                '土壤检出数': len(soil_positive),
                '水体检出数': len(water_positive),
                '统计P值': p_value,
                '显著性差异': '是' if p_value < 0.05 else '否'
            }

    if concentration_comparison:
        conc_df = pd.DataFrame(concentration_comparison).T
        print(dataframe_to_markdown(conc_df, "土壤-水体污染物浓度对比", 3))

    comparison_results['concentrations'] = concentration_comparison

    # 3. Pollutant composition comparison
    print("\n3. 污染物组成对比:")

    soil_composition = {}
    water_composition = {}

    for poll in available_common:
        soil_total = soil_data[poll].sum()
        water_total = water_data[poll].sum()

        if soil_total > 0:
            soil_composition[poll] = soil_total
        if water_total > 0:
            water_composition[poll] = water_total

    if soil_composition and water_composition:
        # Normalize to percentages
        soil_pct = {k: (v / sum(soil_composition.values()) * 100) for k, v in soil_composition.items()}
        water_pct = {k: (v / sum(water_composition.values()) * 100) for k, v in water_composition.items()}

        composition_comparison = {}
        for poll in available_common:
            composition_comparison[poll] = {
                '土壤占比(%)': soil_pct.get(poll, 0),
                '水体占比(%)': water_pct.get(poll, 0),
                '占比差异': soil_pct.get(poll, 0) - water_pct.get(poll, 0)
            }

        comp_df = pd.DataFrame(composition_comparison).T
        print(dataframe_to_markdown(comp_df, "土壤-水体污染物组成对比", 2))

    comparison_results['composition'] = composition_comparison if soil_composition and water_composition else None

    # 4. Environmental factor correlation comparison
    print("\n4. 环境因子相关性对比:")

    # Compare correlations with environmental factors
    env_correlation_comparison = {}

    soil_env_factors = ['pH', 'Temp', 'Rain'] if 'Temp' in soil_data.columns else ['pH', 'Rain']
    water_env_factors = ['pH', 'T_W', 'PREC'] if 'T_W' in water_data.columns else ['pH', 'PREC']

    soil_pollutant = 'parentNNIs' if 'parentNNIs' in soil_data.columns else 'THIA'
    water_pollutant = 'parentNNIs' if 'parentNNIs' in water_data.columns else 'THIA'

    for factor in soil_env_factors:
        if factor in soil_data.columns and soil_pollutant in soil_data.columns:
            soil_corr = soil_data[[factor, soil_pollutant]].corr().iloc[0, 1]
            env_correlation_comparison[factor] = {
                '土壤相关系数': soil_corr,
                '水体相关系数': np.nan,
                '相关系数差异': np.nan
            }

    for factor in water_env_factors:
        if factor in water_data.columns and water_pollutant in water_data.columns:
            water_corr = water_data[[factor, water_pollutant]].corr().iloc[0, 1]
            if factor in env_correlation_comparison:
                env_correlation_comparison[factor]['水体相关系数'] = water_corr
                env_correlation_comparison[factor]['相关系数差异'] = (
                    env_correlation_comparison[factor]['土壤相关系数'] - water_corr
                )
            else:
                env_correlation_comparison[factor] = {
                    '土壤相关系数': np.nan,
                    '水体相关系数': water_corr,
                    '相关系数差异': np.nan
                }

    if env_correlation_comparison:
        env_df = pd.DataFrame(env_correlation_comparison).T
        print(dataframe_to_markdown(env_df, "土壤-水体环境因子相关性对比", 3))

    comparison_results['environmental_correlations'] = env_correlation_comparison

    # 5. Seasonal pattern comparison
    if 'Season' in soil_data.columns and 'Season' in water_data.columns:
        print("\n5. 季节性模式对比:")

        seasonal_comparison = {}
        season_order = ['Dry', 'Normal', 'Rainy']

        for poll in available_common:
            soil_seasonal = []
            water_seasonal = []

            for season in season_order:
                soil_season_data = soil_data[soil_data['Season'] == season][poll].dropna()
                water_season_data = water_data[water_data['Season'] == season][poll].dropna()

                soil_seasonal.append({
                    'season': season,
                    'mean': soil_season_data.mean(),
                    'median': soil_season_data.median(),
                    'detection_rate': (soil_season_data > 0).mean() * 100 if len(soil_season_data) > 0 else 0
                })

                water_seasonal.append({
                    'season': season,
                    'mean': water_season_data.mean(),
                    'median': water_season_data.median(),
                    'detection_rate': (water_season_data > 0).mean() * 100 if len(water_season_data) > 0 else 0
                })

            seasonal_comparison[poll] = {
                'soil_seasonal': soil_seasonal,
                'water_seasonal': water_seasonal
            }

        # Display seasonal patterns
        for poll in available_common:
            if poll in seasonal_comparison:
                print(f"\n{poll} 季节性变化:")
                soil_data_season = seasonal_comparison[poll]['soil_seasonal']
                water_data_season = seasonal_comparison[poll]['water_seasonal']

                print(f"{'季节':<8} {'土壤均值':<12} {'土壤检出率(%)':<15} {'水体均值':<12} {'水体检出率(%)':<15}")
                print("-" * 65)
                for i, season in enumerate(season_order):
                    print(f"{season:<8} {soil_data_season[i]['mean']:<12.3f} {soil_data_season[i]['detection_rate']:<15.1f} "
                          f"{water_data_season[i]['mean']:<12.3f} {water_data_season[i]['detection_rate']:<15.1f}")

        comparison_results['seasonal_patterns'] = seasonal_comparison

    # 6. Geographic distribution comparison
    if all(col in soil_data.columns for col in ['Lon', 'Lat']) and all(col in water_data.columns for col in ['Lon', 'Lat']):
        print("\n6. 地理分布对比:")

        # Calculate geographic ranges
        soil_geo_range = {
            '经度范围': (soil_data['Lon'].min(), soil_data['Lon'].max()),
            '纬度范围': (soil_data['Lat'].min(), soil_data['Lat'].max()),
            '样本数': len(soil_data)
        }

        water_geo_range = {
            '经度范围': (water_data['Lon'].min(), water_data['Lon'].max()),
            '纬度范围': (water_data['Lat'].min(), water_data['Lat'].max()),
            '样本数': len(water_data)
        }

        geo_comparison = {
            '土壤地理范围': soil_geo_range,
            '水体地理范围': water_geo_range
        }

        print(f"土壤采样范围: 经度 {soil_geo_range['经度范围'][0]:.3f}~{soil_geo_range['经度范围'][1]:.3f}, "
              f"纬度 {soil_geo_range['纬度范围'][0]:.3f}~{soil_geo_range['纬度范围'][1]:.3f}")
        print(f"水体采样范围: 经度 {water_geo_range['经度范围'][0]:.3f}~{water_geo_range['经度范围'][1]:.3f}, "
              f"纬度 {water_geo_range['纬度范围'][0]:.3f}~{water_geo_range['纬度范围'][1]:.3f}")

        comparison_results['geographic_ranges'] = geo_comparison

    # 7. Summary of key findings
    print("\n7. 对比分析总结:")

    key_findings = []

    # Detection rate differences
    if detection_comparison:
        max_detection_diff = max(detection_comparison.items(),
                                key=lambda x: abs(x[1]['差异']))
        key_findings.append(f"检出率差异最大: {max_detection_diff[0]} "
                          f"(土壤{max_detection_diff[1]['土壤检出率(%)']:.1f}% vs "
                          f"水体{max_detection_diff[1]['水体检出率(%)']:.1f}%)")

    # Concentration differences
    if concentration_comparison:
        significant_diff = [k for k, v in concentration_comparison.items() if v['显著性差异'] == '是']
        if significant_diff:
            key_findings.append(f"浓度显著差异污染物: {', '.join(significant_diff)}")
        else:
            key_findings.append("所有污染物浓度在土壤和水体间无显著差异")

    # Composition differences
    if composition_comparison:
        max_comp_diff = max(composition_comparison.items(),
                          key=lambda x: abs(x[1]['占比差异']))
        key_findings.append(f"组成差异最大: {max_comp_diff[0]} "
                          f"(差异{max_comp_diff[1]['占比差异']:.1f}%)")

    print("关键发现:")
    for i, finding in enumerate(key_findings, 1):
        print(f"{i}. {finding}")

    comparison_results['key_findings'] = key_findings

    return comparison_results

def main():
    """Main analysis function"""
    print("开始探索性数据分析...")

    # Load data
    soil_data, water_data = load_data()

    # Basic information analysis
    soil_missing = basic_info_analysis(soil_data, "土壤")
    water_missing = basic_info_analysis(water_data, "水体")

    # Pollutant analysis
    soil_parent_stats, soil_metabolite_stats, soil_detection = pollutant_analysis(soil_data, "土壤")
    water_parent_stats, water_metabolite_stats, water_detection = pollutant_analysis(water_data, "水体")

    # Enhanced pollutant analysis
    soil_composition, soil_ratios = enhanced_pollutant_analysis(soil_data, "土壤")
    water_composition, water_ratios = enhanced_pollutant_analysis(water_data, "水体")

    # Environmental correlation analysis
    soil_env_corr, soil_strong_corr = environmental_correlation_analysis(soil_data, "土壤")
    water_env_corr, water_strong_corr = environmental_correlation_analysis(water_data, "水体")

    # Agricultural impact analysis
    agricultural_impact_analysis(soil_data, "土壤")
    agricultural_impact_analysis(water_data, "水体")

    # Economic impact analysis
    economic_impact_analysis(soil_data, "土壤")
    economic_impact_analysis(water_data, "水体")

    # Professional analysis indicators
    soil_prof_ci, soil_prof_risk, soil_prof_trends = professional_analysis_indicators(soil_data, "土壤")
    water_prof_ci, water_prof_risk, water_prof_trends = professional_analysis_indicators(water_data, "水体")

    # Spatial autocorrelation analysis
    soil_spatial = spatial_autocorrelation_analysis(soil_data, "土壤")
    water_spatial = spatial_autocorrelation_analysis(water_data, "水体")

    # Data quality assessment
    soil_quality = data_quality_assessment(soil_data, "土壤")
    water_quality = data_quality_assessment(water_data, "水体")

    # Comparative analysis between soil and water
    comparison_results = comparative_analysis(soil_data, water_data)

    # Seasonal analysis
    soil_seasons = seasonal_analysis(soil_data, "土壤")
    water_seasons = seasonal_analysis(water_data, "水体")

    # Extended variable analyses
    # Environmental factors
    soil_env = environmental_factors_analysis(soil_data, "土壤")
    water_env = environmental_factors_analysis(water_data, "水体")

    # Vegetation indices (soil only)
    soil_veg = vegetation_indices_analysis(soil_data, "土壤")

    # Agricultural variables
    soil_agri = agricultural_variables_analysis(soil_data, "土壤")
    water_agri = agricultural_variables_analysis(water_data, "水体")

    # Economic variables
    soil_econ = economic_variables_analysis(soil_data, "土壤")
    water_econ = economic_variables_analysis(water_data, "水体")

    # Water usage variables
    soil_water = water_usage_analysis(soil_data, "土壤")
    water_water = water_usage_analysis(water_data, "水体")

    # Land use analysis
    soil_landuse = landuse_analysis(soil_data, "土壤")
    water_landuse = landuse_analysis(water_data, "水体")

    # Geographic analysis
    soil_geo = geographic_analysis(soil_data, "土壤")
    water_geo = geographic_analysis(water_data, "水体")

    # Summary statistics
    summary_statistics(soil_data, water_data)

    # Save all statistics to tables
    tables_dir = save_statistics_tables(soil_data, water_data, soil_parent_stats, soil_metabolite_stats,
                                       water_parent_stats, water_metabolite_stats, soil_detection, water_detection,
                                       soil_veg, soil_agri, soil_econ, soil_water, soil_landuse,
                                       water_agri, water_econ, water_water, water_landuse)

    # Create visualizations
    create_visualizations(soil_data, water_data)

    # Create enhanced visualizations
    create_enhanced_visualizations(soil_data, water_data)

    print(f"\n{'='*60}")
    print("探索性数据分析完成！")
    print(f"统计表格已保存到: {tables_dir}")
    print(f"图表已保存到: {tables_dir.parent / 'eda_plots'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
