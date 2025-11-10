# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scipy",
#   "scikit-learn",
#   "geopandas"
# ]
# ///

"""
Spatiotemporal Distribution Analysis for NNI Prediction
时空分布特征分析 - 为自适应网格配对算法提供参数依据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.spatial import distance_matrix
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
import json
from datetime import datetime
import itertools

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SpatiotemporalAnalyzer:
    """Spatiotemporal distribution analyzer for NNI prediction"""

    def __init__(self):
        self.soil_data = None
        self.water_data = None

    def load_data(self):
        """Load soil and water datasets"""
        script_dir = Path(__file__).parent
        data_path = script_dir.parent / 'datasets'

        self.soil_data = pd.read_csv(data_path / 'soil_data.csv', index_col=0)
        self.water_data = pd.read_csv(data_path / 'water_data.csv', index_col=0)

        print(f"土壤数据集形状: {self.soil_data.shape}")
        print(f"水体数据集形状: {self.water_data.shape}")

        return self.soil_data, self.water_data

    def analyze_geographic_distribution(self):
        """Analyze geographic distribution characteristics"""
        geo_stats = {}

        for dataset_name, data in [('soil', self.soil_data), ('water', self.water_data)]:
            if 'Lon' in data.columns and 'Lat' in data.columns:
                coords = data[['Lon', 'Lat']].dropna()

                geo_stats[dataset_name] = {
                    'total_samples': len(coords),
                    'lon_range': [coords['Lon'].min(), coords['Lon'].max()],
                    'lat_range': [coords['Lat'].min(), coords['Lat'].max()],
                    'lon_center': coords['Lon'].mean(),
                    'lat_center': coords['Lat'].mean(),
                    'lon_std': coords['Lon'].std(),
                    'lat_std': coords['Lat'].std()
                }

                # Calculate geographic extent
                lon_extent = coords['Lon'].max() - coords['Lon'].min()
                lat_extent = coords['Lat'].max() - coords['Lat'].min()
                geo_stats[dataset_name]['geographic_extent'] = {
                    'longitude_span': lon_extent,
                    'latitude_span': lat_extent,
                    'total_area_deg2': lon_extent * lat_extent
                }

        return geo_stats

    def analyze_seasonal_distribution(self):
        """Analyze seasonal distribution patterns"""
        seasonal_stats = {}

        for dataset_name, data in [('soil', self.soil_data), ('water', self.water_data)]:
            if 'Season' in data.columns:
                season_counts = data['Season'].value_counts().to_dict()

                seasonal_stats[dataset_name] = {
                    'season_counts': season_counts,
                    'total_samples': len(data),
                    'seasonal_balance': {season: count/len(data) for season, count in season_counts.items()}
                }

                # Analyze geographic distribution by season
                if 'Lon' in data.columns and 'Lat' in data.columns:
                    seasonal_geo = {}
                    for season in data['Season'].unique():
                        season_data = data[data['Season'] == season]
                        if len(season_data) > 0:
                            seasonal_geo[season] = {
                                'count': len(season_data),
                                'lon_mean': season_data['Lon'].mean(),
                                'lat_mean': season_data['Lat'].mean(),
                                'lon_std': season_data['Lon'].std(),
                                'lat_std': season_data['Lat'].std()
                            }
                    seasonal_stats[dataset_name]['seasonal_geographic'] = seasonal_geo

        return seasonal_stats

    def calculate_sample_density(self, data, bandwidth=0.1):
        """Calculate sample density using Kernel Density Estimation"""
        if 'Lon' not in data.columns or 'Lat' not in data.columns:
            return None

        coords = data[['Lon', 'Lat']].dropna()
        if len(coords) == 0:
            return None

        # Standardize coordinates for KDE
        coords_standardized = (coords - coords.mean()) / coords.std()

        # Fit KDE
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(coords_standardized)

        # Calculate density for each point
        log_dens = kde.score_samples(coords_standardized)
        densities = np.exp(log_dens)

        return {
            'coordinates': coords.values,
            'densities': densities,
            'kde_model': kde,
            'standardization_params': {
                'mean': coords.mean().values,
                'std': coords.std().values
            }
        }

    def identify_spatial_clusters(self, data, eps=0.15, min_samples=3):
        """Identify spatial clusters using DBSCAN"""
        if 'Lon' not in data.columns or 'Lat' not in data.columns:
            return None

        coords = data[['Lon', 'Lat']].dropna()
        if len(coords) < min_samples:
            return None

        # Standardize coordinates
        coords_standardized = (coords - coords.mean()) / coords.std()

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(coords_standardized)

        # Analyze clusters
        unique_clusters = set(clusters)
        cluster_info = {}
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise points
                cluster_info['noise'] = (clusters == -1).sum()
            else:
                cluster_points = coords[clusters == cluster_id]
                cluster_info[f'cluster_{cluster_id}'] = {
                    'count': len(cluster_points),
                    'lon_mean': cluster_points['Lon'].mean(),
                    'lat_mean': cluster_points['Lat'].mean(),
                    'lon_std': cluster_points['Lon'].std(),
                    'lat_std': cluster_points['Lat'].std()
                }

        return {
            'cluster_labels': clusters,
            'cluster_info': cluster_info,
            'n_clusters': len(unique_clusters) - (1 if -1 in unique_clusters else 0),
            'n_noise_points': (clusters == -1).sum()
        }

    def calculate_optimal_grid_parameters(self, geo_stats, seasonal_stats):
        """Calculate optimal grid parameters for spatial pairing"""
        grid_params = {}

        # Analyze both soil and water distributions
        for dataset_name in ['soil', 'water']:
            if dataset_name not in geo_stats:
                continue

            geo = geo_stats[dataset_name]
            samples = geo['total_samples']

            # Calculate grid size based on sample density
            area = geo['geographic_extent']['total_area_deg2']
            sample_density = samples / area

            # Recommended grid size (in degrees)
            if sample_density > 50:  # High density
                grid_size = 0.05  # ~5.5 km
            elif sample_density > 20:  # Medium density
                grid_size = 0.1   # ~11 km
            elif sample_density > 10:  # Low density
                grid_size = 0.2   # ~22 km
            else:  # Very low density
                grid_size = 0.3   # ~33 km

            # Calculate number of grid cells
            lon_cells = int(np.ceil(geo['geographic_extent']['longitude_span'] / grid_size))
            lat_cells = int(np.ceil(geo['geographic_extent']['latitude_span'] / grid_size))

            grid_params[dataset_name] = {
                'recommended_grid_size_deg': grid_size,
                'grid_size_km': grid_size * 111,  # Approximate conversion
                'lon_cells': lon_cells,
                'lat_cells': lat_cells,
                'total_cells': lon_cells * lat_cells,
                'samples_per_cell_avg': samples / (lon_cells * lat_cells),
                'sample_density_per_deg2': sample_density
            }

        # Combined parameters for soil-water pairing
        if 'soil' in grid_params and 'water' in grid_params:
            # Use the smaller grid size for higher resolution
            min_grid_size = min(grid_params['soil']['recommended_grid_size_deg'],
                              grid_params['water']['recommended_grid_size_deg'])

            grid_params['combined'] = {
                'recommended_grid_size_deg': min_grid_size,
                'grid_size_km': min_grid_size * 111,
                'max_soil_cells': grid_params['soil']['total_cells'],
                'max_water_cells': grid_params['water']['total_cells']
            }

        return grid_params

    def analyze_sample_distances(self):
        """Analyze inter-sample distances"""
        distance_stats = {}

        for dataset_name, data in [('soil', self.soil_data), ('water', self.water_data)]:
            if 'Lon' not in data.columns or 'Lat' not in data.columns:
                continue

            coords = data[['Lon', 'Lat']].dropna()
            if len(coords) < 2:
                continue

            # Calculate pairwise distances (in degrees)
            dist_matrix = distance_matrix(coords.values, coords.values)

            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
            distances = dist_matrix[mask]

            # Convert to kilometers (approximate)
            distances_km = distances * 111

            distance_stats[dataset_name] = {
                'min_distance_deg': distances.min(),
                'max_distance_deg': distances.max(),
                'mean_distance_deg': distances.mean(),
                'std_distance_deg': distances.std(),
                'min_distance_km': distances_km.min(),
                'max_distance_km': distances_km.max(),
                'mean_distance_km': distances_km.mean(),
                'std_distance_km': distances_km.std(),
                'median_distance_km': np.median(distances_km)
            }

            # Distance percentiles
            percentiles = [10, 25, 50, 75, 90, 95]
            distance_stats[dataset_name]['percentiles_km'] = {
                f'p{p}': np.percentile(distances_km, p) for p in percentiles
            }

        return distance_stats

    def create_visualizations(self, geo_stats, seasonal_stats, density_results, cluster_results, grid_params, distance_stats):
        """Create comprehensive spatiotemporal visualizations"""
        figures = {}

        # 1. Geographic distribution overview - Single combined plot
        fig1, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig1.suptitle('土壤与水体样本分布对比', fontsize=16, fontweight='bold')

        # Combined soil-water distribution in one plot
        if 'Lon' in self.soil_data.columns and 'Lat' in self.soil_data.columns and \
           'Lon' in self.water_data.columns and 'Lat' in self.water_data.columns:
            soil_coords = self.soil_data[['Lon', 'Lat']].dropna()
            water_coords = self.water_data[['Lon', 'Lat']].dropna()

            # Plot soil samples
            ax.scatter(soil_coords['Lon'], soil_coords['Lat'],
                      c='red', alpha=0.6, s=50, marker='o',
                      label=f'土壤样本 ({len(soil_coords)})', edgecolors='darkred', linewidth=0.5)

            # Plot water samples
            ax.scatter(water_coords['Lon'], water_coords['Lat'],
                      c='blue', alpha=0.6, s=50, marker='^',
                      label=f'水体样本 ({len(water_coords)})', edgecolors='darkblue', linewidth=0.5)

            ax.set_xlabel('经度', fontsize=12)
            ax.set_ylabel('纬度', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)

            # Set equal aspect ratio for better spatial representation
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        figures['geographic_overview'] = fig1

        # 2. Seasonal distribution comparison
        fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig2.suptitle('季节样本分布对比', fontsize=16, fontweight='bold')

        if 'soil' in seasonal_stats and 'seasonal_counts' in seasonal_stats['soil']:
            seasons = list(seasonal_stats['soil']['season_counts'].keys())
            counts = list(seasonal_stats['soil']['season_counts'].values())

            # Create side-by-side bars for soil and water
            if 'water' in seasonal_stats and 'seasonal_counts' in seasonal_stats['water']:
                water_counts = list(seasonal_stats['water']['season_counts'].values())
                x = np.arange(len(seasons))
                width = 0.35

                ax.bar(x - width/2, counts, width, label='土壤样本', color='red', alpha=0.7)
                ax.bar(x + width/2, water_counts, width, label='水体样本', color='blue', alpha=0.7)

                ax.set_xlabel('季节', fontsize=12)
                ax.set_ylabel('样本数量', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(seasons)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures['seasonal_distribution'] = fig2

        # 3. Density analysis
        if density_results:
            fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig3.suptitle('样本密度分析', fontsize=16, fontweight='bold')

            # Soil density
            if 'soil' in density_results:
                soil_density = density_results['soil']
                scatter = axes[0].scatter(soil_density['coordinates'][:, 0],
                                        soil_density['coordinates'][:, 1],
                                        c=soil_density['densities'], cmap='YlOrRd', s=50)
                axes[0].set_title('土壤样本密度分布')
                axes[0].set_xlabel('经度')
                axes[0].set_ylabel('纬度')
                plt.colorbar(scatter, ax=axes[0], label='密度')

            # Water density
            if 'water' in density_results:
                water_density = density_results['water']
                scatter = axes[1].scatter(water_density['coordinates'][:, 0],
                                        water_density['coordinates'][:, 1],
                                        c=water_density['densities'], cmap='YlOrRd', s=50)
                axes[1].set_title('水体样本密度分布')
                axes[1].set_xlabel('经度')
                axes[1].set_ylabel('纬度')
                plt.colorbar(scatter, ax=axes[1], label='密度')

            plt.tight_layout()
            figures['density_analysis'] = fig3

        # 4. Distance analysis
        if distance_stats:
            fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig4.suptitle('样本间距离分析', fontsize=16, fontweight='bold')

            for i, (dataset_name, stats) in enumerate(distance_stats.items()):
                coords = (self.soil_data if dataset_name == 'soil' else self.water_data)[['Lon', 'Lat']].dropna()
                dist_matrix = distance_matrix(coords.values, coords.values)
                mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
                distances = dist_matrix[mask] * 111  # Convert to km

                axes[i].hist(distances, bins=50, alpha=0.7, label=dataset_name)
                axes[i].set_title(f'{dataset_name.capitalize()} 样本间距离分布')
                axes[i].set_xlabel('距离 (km)')
                axes[i].set_ylabel('频次')
                axes[i].axvline(stats['median_distance_km'], color='red',
                               linestyle='--', label=f'中位数: {stats["median_distance_km"]:.1f} km')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            figures['distance_analysis'] = fig4

        return figures

    def save_results(self, geo_stats, seasonal_stats, density_results, cluster_results, grid_params, distance_stats, figures):
        """Save all analysis results"""
        # Get output directory
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'outputs' / 'stage1_spatiotemporal_analysis'

        # Save analysis results as JSON
        results = {
            'geographic_stats': geo_stats,
            'seasonal_stats': seasonal_stats,
            'grid_parameters': grid_params,
            'distance_stats': distance_stats,
            'analysis_timestamp': datetime.now().isoformat()
        }

        # Remove non-serializable objects
        if density_results:
            density_results_clean = {}
            for key, value in density_results.items():
                if isinstance(value, dict):
                    density_results_clean[key] = {
                        k: v for k, v in value.items()
                        if k not in ['kde_model', 'coordinates', 'densities']
                    }
            results['density_summary'] = density_results_clean

        if cluster_results:
            cluster_results_clean = {}
            for key, value in cluster_results.items():
                if isinstance(value, dict):
                    cluster_results_clean[key] = {
                        k: v for k, v in value.items()
                        if k not in ['cluster_labels']
                    }
            results['cluster_summary'] = cluster_results_clean

        with open(output_dir / 'spatiotemporal_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # Save grid parameters separately
        with open(output_dir / 'grid_parameters.json', 'w', encoding='utf-8') as f:
            json.dump(grid_params, f, indent=2, ensure_ascii=False, default=str)

        # Save visualizations
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        for name, fig in figures.items():
            fig.savefig(viz_dir / f'{name}.png', bbox_inches='tight', dpi=300)

        # Save spatial statistics as CSV
        spatial_stats_data = []
        for dataset_name, stats in geo_stats.items():
            spatial_stats_data.append({
                'dataset': dataset_name,
                'total_samples': stats['total_samples'],
                'lon_min': stats['lon_range'][0],
                'lon_max': stats['lon_range'][1],
                'lat_min': stats['lat_range'][0],
                'lat_max': stats['lat_range'][1],
                'longitude_span': stats['geographic_extent']['longitude_span'],
                'latitude_span': stats['geographic_extent']['latitude_span'],
                'total_area_deg2': stats['geographic_extent']['total_area_deg2']
            })

        spatial_stats_df = pd.DataFrame(spatial_stats_data)
        spatial_stats_df.to_csv(output_dir / 'spatial_statistics.csv', index=False)

        print(f"分析结果已保存至: {output_dir}")

    def run_analysis(self):
        """Run complete spatiotemporal analysis"""
        print("=" * 60)
        print("开始时空分布特征分析...")
        print("=" * 60)

        # Load data
        self.load_data()

        # Geographic distribution analysis
        print("分析地理分布特征...")
        geo_stats = self.analyze_geographic_distribution()

        # Seasonal distribution analysis
        print("分析季节分布特征...")
        seasonal_stats = self.analyze_seasonal_distribution()

        # Sample density analysis
        print("计算样本密度...")
        density_results = {
            'soil': self.calculate_sample_density(self.soil_data),
            'water': self.calculate_sample_density(self.water_data)
        }

        # Spatial clustering analysis
        print("识别空间聚类...")
        cluster_results = {
            'soil': self.identify_spatial_clusters(self.soil_data),
            'water': self.identify_spatial_clusters(self.water_data)
        }

        # Distance analysis
        print("分析样本间距离...")
        distance_stats = self.analyze_sample_distances()

        # Calculate optimal grid parameters
        print("计算最优网格参数...")
        grid_params = self.calculate_optimal_grid_parameters(geo_stats, seasonal_stats)

        # Create visualizations
        print("生成可视化图表...")
        figures = self.create_visualizations(geo_stats, seasonal_stats, density_results, cluster_results, grid_params, distance_stats)

        # Save results
        print("保存分析结果...")
        self.save_results(geo_stats, seasonal_stats, density_results, cluster_results, grid_params, distance_stats, figures)

        # Print summary
        self._print_summary(geo_stats, grid_params, distance_stats)

        print("\n时空分布特征分析完成！")

        return {
            'geo_stats': geo_stats,
            'grid_params': grid_params,
            'distance_stats': distance_stats
        }

    def _print_summary(self, geo_stats, grid_params, distance_stats):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("分析结果摘要")
        print("=" * 60)

        # Geographic summary
        print("\n地理分布:")
        for dataset_name, stats in geo_stats.items():
            print(f"  {dataset_name}: {stats['total_samples']}个样本")
            print(f"    经度范围: {stats['lon_range'][0]:.3f}° ~ {stats['lon_range'][1]:.3f}°")
            print(f"    纬度范围: {stats['lat_range'][0]:.3f}° ~ {stats['lat_range'][1]:.3f}°")
            print(f"    覆盖面积: {stats['geographic_extent']['total_area_deg2']:.2f} 平方度")

        # Grid parameters
        print("\n网格参数建议:")
        if 'combined' in grid_params:
            combined = grid_params['combined']
            print(f"  推荐网格大小: {combined['recommended_grid_size_deg']:.3f}° ({combined['grid_size_km']:.1f} km)")
            print(f"  预计土壤网格数: {combined['max_soil_cells']}")
            print(f"  预计水体网格数: {combined['max_water_cells']}")

        # Distance statistics
        print("\n样本间距离统计:")
        for dataset_name, stats in distance_stats.items():
            print(f"  {dataset_name}:")
            print(f"    平均距离: {stats['mean_distance_km']:.1f} km")
            print(f"    中位数距离: {stats['median_distance_km']:.1f} km")
            print(f"    最大距离: {stats['max_distance_km']:.1f} km")
            print(f"    90%样本距离小于: {stats['percentiles_km']['p90']:.1f} km")

        # Recommendations
        print("\n时空配对建议:")
        if 'combined' in grid_params:
            grid_size = grid_params['combined']['grid_size_km']
            print(f"  - 使用 {grid_size:.1f} km 网格进行土壤-水体配对")
            print(f"  - 网格内平均应有充足的土壤样本支持水体预测")
            print(f"  - 建议使用自适应距离阈值进行样本配对")


def main():
    """Main function to run spatiotemporal analysis"""
    analyzer = SpatiotemporalAnalyzer()
    results = analyzer.run_analysis()
    return results


if __name__ == "__main__":
    main()