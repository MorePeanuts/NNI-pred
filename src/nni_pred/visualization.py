from shap.maskers import Independent
import shap
import json
import copy
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Iterable
from nni_pred.evaluation import Metrics, OOFMetrics
from nni_pred.transformers import TargetTransformer
from nni_pred.utils import Explorer


class Visualizer:
    def __init__(
        self,
        exp_root: Path,
        use_shap: bool = True,
    ):
        self.exp_root = exp_root
        self.explorer = Explorer(exp_root)
        self.targets = self.explorer.get_targets_list()

        if use_shap:
            self._init_shap_values()

    def _init_shap_values(self):
        self.shap_values = {}
        self.features = {}
        for target in self.targets.copy():
            model_type = self.explorer.get_best_model_type(target)
            # if model_type == 'linear':
            #     self.targets.remove(target)
            #     continue

            pipeline = joblib.load(self.explorer.get_best_model_path(target))
            features = self.explorer.get_features(target)
            cat_cols = self.explorer.var_cls.categorical
            mappings = {col: list(features[col].unique()) for col in cat_cols}
            features_numeric = features.copy()
            for col in cat_cols:
                features_numeric[col] = features[col].map(lambda x: mappings[col].index(x))

            def model_predict(data):
                df = pd.DataFrame(data, columns=features.columns)
                for col in cat_cols:
                    df[col] = df[col].apply(lambda x: mappings[col][int(round(float(x)))])
                return pipeline.predict(df)

            explainer = shap.Explainer(model_predict, features_numeric)
            sh_val = explainer(features_numeric)
            sh_val.data = features.values
            self.shap_values[target] = sh_val

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
            data = {}
            for target in self.targets:
                data[target] = (
                    self.explorer.get_best_model_type(target),
                    self.explorer.get_oof_metrics(target),
                )
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
        for i, metrics in enumerate(metrics_used):
            if figshape[0] > 1 and figshape[1] > 1:
                ax = axes[i // figshape[1]][i % figshape[1]]
            elif figshape[0] == 1 and figshape[1] == 1:
                ax = axes
            else:
                ax = axes[i]
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
        data: dict[str, tuple[str, OOFMetrics]] | None = None,
        targets_used: Iterable[str] | None = None,
        use_log: bool = False,
        output_suffix: str | None = None,
    ):
        """
        plot measured vs predicted plots.
        """
        if data is None:
            data = {}
            for target in self.targets:
                data[target] = (
                    self.explorer.get_best_model_type(target),
                    self.explorer.get_oof_metrics(target),
                )
        output_path = (
            self.exp_root
            / f'measured_vs_predicted{"_" + output_suffix if output_suffix else ""}.png'
        )
        if targets_used is not None:
            for target in list(data.keys()):
                if target not in targets_used:
                    data.pop(target)

        if use_log:
            trans = TargetTransformer(0)
            for target, (_, oof_metrics) in data.items():
                predictions = oof_metrics.oof_predictions
                predictions[f'log_{target}'] = trans.transform(predictions[target])
                predictions[f'log_{target}_pred'] = trans.transform(predictions[f'{target}_pred'])

        total_plots = len(data)
        figshape, figsize = self._create_subplots_shape_and_figsize(total_plots)
        fig, axes = plt.subplots(*figshape, figsize=figsize)
        season_colors = {
            'Dry': '#e74c3c',  # Red
            'Normal': '#f39c12',  # Orange
            'Rainy': '#3498db',  # Blue
        }

        is_first = True
        for i, (target, (_, oof_metrics)) in enumerate(data.items()):
            if figshape[0] > 1 and figshape[1] > 1:
                ax = axes[i // figshape[1]][i % figshape[1]]
            elif figshape[0] == 1 and figshape[1] == 1:
                ax = axes
            else:
                ax = axes[i]
            predictions = oof_metrics.oof_predictions
            r2 = oof_metrics.oof.NSE_log
            y_true = predictions[f'log_{target}'] if use_log else predictions[target]
            y_pred = predictions[f'log_{target}_pred'] if use_log else predictions[f'{target}_pred']

            for season, color in season_colors.items():
                preds = predictions[predictions['Season'] == season]
                ax.scatter(
                    preds[f'log_{target}'] if use_log else preds[target],
                    preds[f'log_{target}_pred'] if use_log else preds[f'{target}_pred'],
                    c=color,
                    label=season if is_first else '',
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.3,
                    s=30,
                )

            max_val = max(y_true.max(), y_pred.max())
            min_val = min(y_true.min(), y_pred.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.5)

            ax.set_title(f'{target}\n(R²={r2:.3f}, (log))', fontsize=11, fontweight='bold')
            ax.set_xlabel('Measured', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')

            if is_first:
                ax.legend(loc='best', fontsize=8)

            is_first = False

        fig.suptitle(
            'Measured vs Predicted (Out-of-Fold)',
            fontsize=16,
            fontweight='bold',
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_shap_summary(
        self,
        shap_values=None,
        data=None,
        targets_used=None,
        output_suffix: str | None = None,
    ):
        if shap_values is None:
            shap_values = self.shap_values.copy()
        output_path = (
            self.exp_root / f'shap_summary{"_" + output_suffix if output_suffix else ""}.png'
        )

        total_plots = len(targets_used) if targets_used else len(shap_values)
        figshape, figsize = self._create_subplots_shape_and_figsize(total_plots)
        fig, axes = plt.subplots(*figshape, figsize=figsize)

        if targets_used:
            for target in list(shap_values.keys()):
                if target not in targets_used:
                    shap_values.pop(target)

        for i, (target, sp_values) in enumerate(shap_values.items()):
            if figshape[0] > 1 and figshape[1] > 1:
                ax = axes[i // figshape[1]][i % figshape[1]]
            elif figshape[0] == 1 and figshape[1] == 1:
                ax = axes
            else:
                ax = axes[i]
            plt.sca(ax)
            features = self.explorer.get_features(target)
            shap.summary_plot(sp_values, features, features.columns, plot_type='dot', show=False)
            ax.set_title(f'{target}', fontsize=13, fontweight='bold')
        plt.suptitle(
            'SHAP Dot Summary Plot',
            fontsize=15,
            fontweight='bold',
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_shap_importance(
        self,
        shap_values=None,
        data=None,
        targets_used=None,
        output_suffix: str | None = None,
    ):
        if shap_values is None:
            shap_values = self.shap_values.copy()
        output_path = (
            self.exp_root / f'shap_importance{"_" + output_suffix if output_suffix else ""}.png'
        )

        total_plots = len(targets_used) if targets_used else len(shap_values)
        figshape, figsize = self._create_subplots_shape_and_figsize(total_plots)
        fig, axes = plt.subplots(*figshape, figsize=figsize)

        if targets_used:
            for target in list(shap_values.keys()):
                if target not in targets_used:
                    shap_values.pop(target)

        for i, (target, sp_values) in enumerate(shap_values.items()):
            if figshape[0] > 1 and figshape[1] > 1:
                ax = axes[i // figshape[1]][i % figshape[1]]
            elif figshape[0] == 1 and figshape[1] == 1:
                ax = axes
            else:
                ax = axes[i]
            plt.sca(ax)
            features = self.explorer.get_features(target)
            shap.summary_plot(sp_values, features, features.columns, plot_type='bar', show=False)
            ax.set_title(f'{target}', fontsize=13, fontweight='bold')
        plt.suptitle(
            'SHAP Importance Plot',
            fontsize=15,
            fontweight='bold',
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_shap_dependence(
        self,
        output_suffix: str | None = None,
    ):
        output_path = (
            self.exp_root / f'shap_dependence{"_" + output_suffix if output_suffix else ""}.png'
        )

    def _create_subplots_shape_and_figsize(self, total_plots: int) -> tuple:
        if total_plots == 1:
            return (1, 1), (5, 8)
        elif total_plots == 2:
            return (1, 2), (8, 15)
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
            return (3, 3), (15, 15)
        elif total_plots == 10:
            return (2, 5), (25, 14)
        elif total_plots == 11:
            return (3, 4), (20, 20)
        else:
            raise RuntimeError(f'Too many subplots: {total_plots}.')
