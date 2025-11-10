# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scipy",
#   "scikit-learn"
# ]
# ///

"""
Water Pollutant Correlation Analysis for NNI Prediction
水体污染物相关性分析 - 为多输出模型分组策略提供依据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
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

class WaterPollutantCorrelationAnalyzer:
    """Water pollutant correlation analyzer for NNI prediction model grouping"""

    def __init__(self):
        self.water_data = None
        # Water pollutants for correlation analysis
        self.water_pollutants = ['THIA', 'IMI', 'CLO', 'ACE', 'DIN', 'parentNNIs',
                                'IMI-UREA', 'DN-IMI', 'DM-ACE', 'CLO-UREA', 'mNNIs']

        # Parent-metabolite relationships
        self.parent_metabolite_map = {
            'THIA': [],
            'IMI': ['IMI-UREA', 'DN-IMI'],
            'CLO': ['CLO-UREA'],
            'ACE': ['DM-ACE'],
            'DIN': [],
            'parentNNIs': ['mNNIs']
        }

    def load_data(self):
        """Load water dataset"""
        script_dir = Path(__file__).parent
        data_path = script_dir.parent / 'datasets'

        self.water_data = pd.read_csv(data_path / 'water_data.csv', index_col=0)
        print(f"水体数据集形状: {self.water_data.shape}")

        return self.water_data

    def get_available_pollutants(self):
        """Get list of available pollutants in the dataset"""
        available = [col for col in self.water_pollutants if col in self.water_data.columns]
        print(f"可用的水体污染物: {available}")
        return available

    def calculate_correlation_matrices(self, pollutants):
        """Calculate correlation matrices for available pollutants"""
        # Filter data to only include available pollutants
        pollutant_data = self.water_data[pollutants].dropna()

        if len(pollutant_data) == 0:
            print("警告: 没有可用的污染物数据进行相关性分析")
            return None

        print(f"相关性分析使用的样本数量: {len(pollutant_data)}")

        # Calculate different correlation coefficients
        pearson_corr = pollutant_data.corr(method='pearson')
        spearman_corr = pollutant_data.corr(method='spearman')
        kendall_corr = pollutant_data.corr(method='kendall')

        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr,
            'data': pollutant_data
        }

    def correlation_significance_test(self, corr_matrix, data):
        """Perform correlation significance test with p-value correction"""
        n = len(data)
        variables = corr_matrix.columns.tolist()

        # Calculate p-values for each correlation
        p_values = pd.DataFrame(index=variables, columns=variables, dtype=float)

        for var1, var2 in itertools.combinations(variables, 2):
            if var1 != var2:
                # Pearson correlation test
                try:
                    corr, p_val = stats.pearsonr(data[var1], data[var2])
                    p_values.loc[var1, var2] = float(p_val)
                    p_values.loc[var2, var1] = float(p_val)
                except:
                    p_values.loc[var1, var2] = 1.0
                    p_values.loc[var2, var1] = 1.0

        # Fill diagonal with 1s (correlation with itself)
        np.fill_diagonal(p_values.values, 1.0)

        # Multiple testing correction (FDR Benjamini-Hochberg)
        from scipy.stats import false_discovery_control

        # Extract upper triangle p-values (excluding diagonal)
        mask = np.triu(np.ones(p_values.shape), k=1).astype(bool)
        p_values_flat = p_values.values[mask]

        # Filter out invalid p-values and apply FDR correction
        if len(p_values_flat) > 0:
            # Ensure all p-values are valid (between 0 and 1)
            valid_p_values = [p for p in p_values_flat if 0 <= p <= 1]

            if valid_p_values:
                corrected_p_flat = false_discovery_control(valid_p_values)

                # Create corrected p-value matrix
                corrected_p = p_values.copy()

                # Fill in corrected values for valid p-values
                valid_idx = 0
                for i in range(p_values.shape[0]):
                    for j in range(i+1, p_values.shape[1]):
                        if mask[i, j] and 0 <= p_values.iloc[i, j] <= 1:
                            corrected_p.iloc[i, j] = corrected_p_flat[valid_idx]
                            corrected_p.iloc[j, i] = corrected_p_flat[valid_idx]
                            valid_idx += 1

                return corrected_p
            else:
                return p_values
        else:
            return p_values

    def hierarchical_clustering(self, corr_matrix, method='average'):
        """Perform hierarchical clustering on correlation matrix"""
        # Convert correlation to distance (use absolute correlation for clustering)
        distance_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(distance_matrix, 0)

        # Hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method=method)

        return linkage_matrix

    def create_disjoint_pollutant_groups(self, corr_matrix, linkage_matrix, threshold=0.7):
        """Create disjoint pollutant groups for multi-output modeling"""
        # Cut dendrogram to form clusters
        clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')

        # Group pollutants by cluster
        variables = corr_matrix.columns.tolist()
        groups = {}

        for i, cluster_id in enumerate(clusters):
            if cluster_id not in groups:
                groups[cluster_id] = []
            groups[cluster_id].append(variables[i])

        # Convert to list and ensure groups are disjoint
        group_list = list(groups.values())

        # Post-process: ensure each pollutant appears in exactly one group
        # This should already be guaranteed by the clustering approach

        return group_list

    def analyze_parent_metabolite_relationships(self, corr_matrix):
        """Analyze specific parent-metabolite correlations"""
        relationships = []

        for parent, metabolites in self.parent_metabolite_map.items():
            if parent in corr_matrix.columns:
                for metabolite in metabolites:
                    if metabolite in corr_matrix.columns:
                        corr_val = corr_matrix.loc[parent, metabolite]
                        relationships.append({
                            'parent': parent,
                            'metabolite': metabolite,
                            'correlation': float(corr_val)
                        })

        return relationships

    def create_visualizations(self, corr_results, pollutant_groups, parent_metabolite_relationships):
        """Create comprehensive visualizations"""
        figures = {}

        # Get correlation matrices
        pearson_corr = corr_results['pearson']
        p_values = self.correlation_significance_test(pearson_corr, corr_results['data'])

        # 1. Correlation matrix heatmap with significance
        fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('水体污染物相关性分析', fontsize=16, fontweight='bold')

        # Pearson correlation
        sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=axes[0, 0])
        axes[0, 0].set_title('Pearson 相关系数矩阵')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Spearman correlation
        sns.heatmap(corr_results['spearman'], annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=axes[0, 1])
        axes[0, 1].set_title('Spearman 相关系数矩阵')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Significant correlations only
        significance_mask = p_values >= 0.05
        sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8},
                   mask=significance_mask, ax=axes[1, 0])
        axes[1, 0].set_title('显著相关 (p<0.05)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Parent-metabolite specific correlations
        if parent_metabolite_relationships:
            parent_meta_df = pd.DataFrame(parent_metabolite_relationships)
            pivot_data = parent_meta_df.pivot(index='parent', columns='metabolite', values='correlation')

            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0,
                           square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=axes[1, 1])
                axes[1, 1].set_title('母体-代谢物相关性')
            else:
                axes[1, 1].text(0.5, 0.5, '无母体-代谢物对数据',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('母体-代谢物相关性')

        plt.tight_layout()
        figures['correlation_matrices'] = fig1

        # 2. Dendrogram and group visualization
        fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('污染物聚类分组分析', fontsize=16, fontweight='bold')

        # Hierarchical clustering dendrogram
        linkage_matrix = self.hierarchical_clustering(pearson_corr)
        dendrogram(linkage_matrix, labels=pearson_corr.columns,
                  leaf_rotation=45, leaf_font_size=10, ax=axes[0])
        axes[0].set_title('层次聚类树状图')
        axes[0].set_xlabel('污染物')
        axes[0].set_ylabel('距离 (1 - |相关系数|)')

        # Group visualization
        group_colors = plt.cm.Set3(np.linspace(0, 1, len(pollutant_groups)))
        pollutant_colors = {}

        for i, group in enumerate(pollutant_groups):
            for pollutant in group:
                pollutant_colors[pollutant] = group_colors[i]

        # Create a matrix showing group assignments
        group_matrix = np.zeros((len(pollutant_groups), len(pearson_corr.columns)))
        group_labels = []

        for i, group in enumerate(pollutant_groups):
            group_labels.append(f"组{i+1}")
            for pollutant in group:
                j = pearson_corr.columns.get_loc(pollutant)
                group_matrix[i, j] = 1

        sns.heatmap(group_matrix, annot=True, cmap='Set3', cbar=False,
                   xticklabels=pearson_corr.columns, yticklabels=group_labels, ax=axes[1])
        axes[1].set_title(f'污染物分组 (阈值=0.7)')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        figures['clustering_groups'] = fig2

        return figures

    def save_results(self, corr_results, pollutant_groups, parent_metabolite_relationships, figures):
        """Save analysis results"""
        # Get output directory
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'outputs' / 'stage1_correlation_analysis'

        # Save correlation matrices
        matrices_dir = output_dir / 'correlation_matrices'
        matrices_dir.mkdir(exist_ok=True)

        for method in ['pearson', 'spearman', 'kendall']:
            corr_matrix = corr_results[method]
            corr_matrix.to_csv(matrices_dir / f'water_{method}_correlation.csv')

        # Save group assignment
        groups_data = {
            'pollutant_groups': pollutant_groups,
            'parent_metabolite_relationships': parent_metabolite_relationships,
            'total_groups': len(pollutant_groups),
            'correlation_threshold': 0.7,
            'analysis_timestamp': datetime.now().isoformat()
        }

        with open(output_dir / 'pollutant_grouping.json', 'w', encoding='utf-8') as f:
            json.dump(groups_data, f, indent=2, ensure_ascii=False)

        # Save visualizations
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        for name, fig in figures.items():
            fig.savefig(viz_dir / f'{name}.png', bbox_inches='tight', dpi=300)

        # Save group assignment as CSV for easy reference
        group_df_data = []
        for i, group in enumerate(pollutant_groups):
            for pollutant in group:
                group_df_data.append({
                    'group': f'Group_{i+1}',
                    'pollutant': pollutant
                })

        group_df = pd.DataFrame(group_df_data)
        group_df.to_csv(output_dir / 'pollutant_group_assignment.csv', index=False)

        print(f"分析结果已保存至: {output_dir}")

    def run_analysis(self, correlation_threshold=0.7):
        """Run complete water pollutant correlation analysis"""
        print("=" * 60)
        print("开始水体污染物相关性分析...")
        print("=" * 60)

        # Load data
        self.load_data()

        # Get available pollutants
        available_pollutants = self.get_available_pollutants()

        if len(available_pollutants) < 2:
            print("错误: 可用污染物数量少于2个，无法进行相关性分析")
            return

        # Calculate correlation matrices
        print("\n计算相关系数矩阵...")
        corr_results = self.calculate_correlation_matrices(available_pollutants)

        if corr_results is None:
            print("相关性分析失败")
            return

        # Perform hierarchical clustering
        print("进行层次聚类分析...")
        linkage_matrix = self.hierarchical_clustering(corr_results['pearson'])

        # Create disjoint pollutant groups
        print("创建污染物分组...")
        pollutant_groups = self.create_disjoint_pollutant_groups(
            corr_results['pearson'], linkage_matrix, threshold=correlation_threshold
        )

        # Analyze parent-metabolite relationships
        print("分析母体-代谢物关系...")
        parent_metabolite_relationships = self.analyze_parent_metabolite_relationships(corr_results['pearson'])

        # Create visualizations
        print("生成可视化图表...")
        figures = self.create_visualizations(corr_results, pollutant_groups, parent_metabolite_relationships)

        # Save results
        print("保存分析结果...")
        self.save_results(corr_results, pollutant_groups, parent_metabolite_relationships, figures)

        # Print summary
        self._print_summary(pollutant_groups, parent_metabolite_relationships)

        print("\n水体污染物相关性分析完成！")

        return {
            'groups': pollutant_groups,
            'parent_metabolite_relationships': parent_metabolite_relationships,
            'correlation_matrices': corr_results
        }

    def _print_summary(self, pollutant_groups, parent_metabolite_relationships):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("分析结果摘要")
        print("=" * 60)

        print(f"\n污染物分组结果 (共{len(pollutant_groups)}组):")
        for i, group in enumerate(pollutant_groups):
            print(f"  组{i+1}: {', '.join(group)}")

        if parent_metabolite_relationships:
            print(f"\n母体-代谢物关系 (共{len(parent_metabolite_relationships)}对):")
            for rel in parent_metabolite_relationships:
                print(f"  {rel['parent']} - {rel['metabolite']}: {rel['correlation']:.3f}")

        # Modeling recommendations
        print(f"\n建模建议:")
        print(f"  - 多输出模型组数: {len(pollutant_groups)}")
        print(f"  - 每组污染物可使用一个多输出模型同时预测")
        print(f"  - 组间污染物相关性较低，建议分别建模")


def main():
    """Main function to run water pollutant correlation analysis"""
    analyzer = WaterPollutantCorrelationAnalyzer()

    # You can adjust the correlation threshold here (default: 0.7)
    results = analyzer.run_analysis(correlation_threshold=0.7)

    return results


if __name__ == "__main__":
    main()