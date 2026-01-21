import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Iterable
from nni_pred.data import get_feature_groups
from nni_pred.evaluation import Metrics, OOFMetrics


class Visualizer:
    def __init__(
        self,
        exp_root: Path,
    ):
        self.exp_root = exp_root
        self.oof_data = {}
        for target_dir in self.exp_root.iterdir():
            if target_dir.is_file():
                continue
            seed_comp_path = target_dir / 'seed_comparison.json'
            if not seed_comp_path.exists():
                continue
            target = target_dir.name
            with seed_comp_path.open() as f:
                seed_comp = json.load(f)
                self.oof_data[target] = (
                    seed_comp['best_model_type'],
                    OOFMetrics.from_json(seed_comp['best_metrics']),
                )

    def plot_cv_metrics(
        self,
        data: dict[str, tuple[str, OOFMetrics]] | None = None,
        metrics_used: Iterable[str] = ['NSE_log', 'RSR_log', 'PBIAS'],
        output_suffix: str | None = None,
    ):
        """
        使用barh和errorbar展示OOFMetrics

        Args:
            data: target为key, tuple[model_type, oof_metrics]作为键
            metrics_used: 展示的指标列表
        """
        if data is None:
            data = self.oof_data
        output_path = (
            self.exp_root / f'metrics_bar_chart{"_" + output_suffix if output_suffix else ""}.png'
        )
        total_plots = len(list(metrics_used))
        rows_mean = []
        rows_std = []
        for target, (_, oof_metrics) in data.items():
            row_mean = {'target': target}
            row_std = {'target': target}
            row_mean.update(
                {metrics: getattr(oof_metrics.mean, metrics) for metrics in metrics_used}
            )
            row_std.update({metrics: getattr(oof_metrics.std, metrics) for metrics in metrics_used})
            rows_mean.append(row_mean)
            rows_std.append(row_std)
        df_mean = pd.DataFrame(rows_mean)
        df_std = pd.DataFrame(rows_std)
        y_pos = np.arange(len(df_mean))

        figshape, figsize = self._create_subplots_shape_and_figsize(total_plots)

        fig, axes = plt.subplots(*figshape, figsize=figsize)
        for ax, metrics in zip(axes, metrics_used, strict=False):
            bar = ax.barh(y_pos, df_mean[metrics], alpha=0.7, edgecolor='black')
            ax.errorbar(
                df_mean[metrics],
                y_pos,
                xerr=df_std[metrics],
                fmt='none',
                ecolor='black',
                capsize=3,
                alpha=0.5,
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_mean['target'], fontsize=10)
            ax.set_xlabel(Metrics.get_metrics_repr(metrics), fontsize=12, fontweight='bold')
            ax.set_title(
                f'{Metrics.get_metrics_repr(metrics)} for best model',
                fontsize=13,
                fontweight='bold',
            )
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.axvline(x=0, color='black', linewidth=0.8)

            # Add value labels on bars
            for b, val, std in zip(bar, df_mean[metrics], df_std[metrics], strict=False):
                label = f'{val:.3f}±{std:.3f}'
                ax.text(
                    val + 0.01, b.get_y() + b.get_height() * 2 / 3, label, va='center', fontsize=8
                )

        fig.suptitle(
            'Nested Spatial Cross-Validation Results',
            fontsize=15,
            fontweight='bold',
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pca_loading(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root / f'pca_loading{"_" + output_suffix if output_suffix else ""}.png'
        )

    def plot_model_performance_box(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root
            / f'model_performance_box{"_" + output_suffix if output_suffix else ""}.png'
        )

    def plot_scatter_identity(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root
            / f'measured_vs_predicted{"_" + output_suffix if output_suffix else ""}.png'
        )

    def plot_shap_summary(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root / f'shap_summary{"_" + output_suffix if output_suffix else ""}.png'
        )

    def plot_shap_importance(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root / f'shap_importance{"_" + output_suffix if output_suffix else ""}.png'
        )

    def plot_shap_interaction(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root / f'shap_interaction{"_" + output_suffix if output_suffix else ""}.png'
        )

    def plot_shap_dependence(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root / f'shap_dependence{"_" + output_suffix if output_suffix else ""}.png'
        )

    def plot_seasonal_shap_comparison(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root
            / f'seasonal_shap_comparison{"_" + output_suffix if output_suffix else ""}.png'
        )

    def plot_subgroup_analysis(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root / f'subgroup_analysis{"_" + output_suffix if output_suffix else ""}.png'
        )

    def _create_subplots_shape_and_figsize(self, total_plots: int) -> tuple:
        if total_plots == 1:
            return (1, 1), (5, 8)
        elif total_plots == 2:
            return (1, 2), (10, 8)
        elif total_plots == 3:
            return (1, 3), (15, 8)
        elif total_plots == 4:
            return (2, 2), (10, 14)
        elif total_plots == 5:
            return (1, 5), (25, 8)
        elif total_plots == 6:
            return (2, 3), (15, 14)
        elif total_plots == 7:
            return (2, 4), (20, 14)
        elif total_plots == 8:
            return (2, 4), (20, 14)
        elif total_plots == 9:
            return (3, 3), (15, 20)
        elif total_plots == 10:
            return (2, 5), (25, 14)
        elif total_plots == 11:
            return (3, 4), (20, 20)
        else:
            raise RuntimeError(f'Too many subplots: {total_plots}.')
