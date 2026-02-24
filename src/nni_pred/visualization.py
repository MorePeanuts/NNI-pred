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
        targets: list[str] | None = None,
        use_shap: bool = True,
    ):
        self.exp_root = exp_root
        self.explorer = Explorer(exp_root)
        if targets:
            self.targets = targets
        else:
            self.targets = self.explorer.get_targets_list()

        if use_shap:
            self._init_shap_values()

    def reset_targets(self, targets: list[str]):
        self.targets = targets

    def _init_shap_values(self):
        self.shap_values = {}
        self.features = {}
        for target in self.targets.copy():
            shap_path = self.exp_root / target / 'shap_values.joblib'

            # Try to load cached SHAP values
            if shap_path.exists():
                self.shap_values[target] = joblib.load(shap_path)
                continue

            # Calculate SHAP values
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

            # Save SHAP values
            joblib.dump(sh_val, shap_path)

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

    def plot_model_comparison(
        self,
        metric: str = 'NSE_log',
        targets_used: Iterable[str] | None = None,
        output_suffix: str | None = None,
    ):
        """
        Plot model comparison for all three models under the best seed.

        Args:
            metric: Metric to compare (default: 'NSE_log')
            targets_used: Targets to plot (default: all targets)
            output_suffix: Suffix for output filename
        """
        model_types = ['linear', 'rf', 'xgb']
        model_names = {'linear': 'Elastic Net', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
        model_colors = {'linear': '#3498db', 'rf': '#2ecc71', 'xgb': '#e74c3c'}
        best_highlight_color = '#f39c12'

        # Get data for all three models
        model_data = {}
        for mt in model_types:
            model_data[mt] = self.explorer.get_data_for_visualization(mt)

        targets = list(targets_used) if targets_used else self.targets

        # Part 1: Grouped bar chart for metric comparison
        self._plot_model_comparison_bar(
            model_data, targets, metric, model_types, model_names, model_colors,
            best_highlight_color, output_suffix
        )

        # Part 2: Scatter plots for each target (3 subplots per target)
        for target in targets:
            self._plot_model_comparison_scatter(
                model_data, target, model_types, model_names, model_colors,
                best_highlight_color, output_suffix
            )

    def _plot_model_comparison_bar(
        self,
        model_data: dict,
        targets: list[str],
        metric: str,
        model_types: list[str],
        model_names: dict,
        model_colors: dict,
        best_highlight_color: str,
        output_suffix: str | None = None,
    ):
        """Plot grouped bar chart comparing models across targets."""
        output_path = (
            self.exp_root / f'model_comparison_bar{"_" + output_suffix if output_suffix else ""}.png'
        )

        n_targets = len(targets)
        bar_width = 0.25
        x = np.arange(n_targets)

        fig, ax = plt.subplots(figsize=(max(10, n_targets * 1.5), 6))

        # First pass: collect all bar data
        all_bar_data = {}  # {target_idx: [(model_idx, bar_x, bar_top, val, std)]}
        all_tops = []

        for i, mt in enumerate(model_types):
            values = []
            errors = []
            for target in targets:
                _, oof_metrics = model_data[mt][target]
                values.append(getattr(oof_metrics.mean, metric))
                errors.append(getattr(oof_metrics.std, metric))

            # Skip error bar if std > abs(mean)
            errors_to_plot = [e if e <= abs(v) else 0 for v, e in zip(values, errors)]

            for v, e in zip(values, errors_to_plot):
                all_tops.append(v + e if v >= 0 else abs(v))

            bars = ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                label=model_names[mt],
                color=model_colors[mt],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.6,
            )
            ax.errorbar(
                x + i * bar_width,
                values,
                yerr=errors_to_plot,
                fmt='none',
                ecolor='black',
                capsize=3,
                alpha=0.5,
            )

            # Collect bar info
            for j, (bar, val, std, err_plot) in enumerate(zip(bars, values, errors, errors_to_plot, strict=False)):
                if j not in all_bar_data:
                    all_bar_data[j] = []
                bar_top = max(val, 0) + err_plot
                bar_x = bar.get_x() + bar.get_width() / 2
                all_bar_data[j].append((i, bar_x, bar_top, val, std))

        # Set y-axis limits
        max_top = max(all_tops)
        y_max = max_top + 0.25 * max_top
        ax.set_ylim(bottom=None, top=y_max)

        # Minimum vertical spacing between labels (in data units)
        label_height = 0.045 * y_max

        # Second pass: add labels with smart positioning
        for target_idx in range(n_targets):
            group_data = all_bar_data[target_idx]
            # Sort by bar_top height
            sorted_data = sorted(group_data, key=lambda x: x[2])

            # Calculate label positions from lowest to highest
            label_positions = {}
            prev_label_y = -999
            for model_idx, bar_x, bar_top, val, std in sorted_data:
                # Label should be above bar, but also above previous label
                desired_y = bar_top + 0.01 * y_max
                actual_y = max(desired_y, prev_label_y + label_height)
                label_positions[model_idx] = (bar_x, actual_y, val, std)
                prev_label_y = actual_y

            # Draw labels
            for model_idx, (bar_x, label_y, val, std) in label_positions.items():
                label = f'{val:.2f}±{std:.2f}'
                ax.text(
                    bar_x,
                    label_y,
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=7,
                )

        ax.set_xlabel('Target', fontsize=12, fontweight='bold')
        ax.set_ylabel(Metrics.get_metrics_repr(metric), fontsize=12, fontweight='bold')
        ax.set_title(
            f'Model Comparison ({Metrics.get_metrics_repr(metric)})',
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(targets, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linewidth=0.8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_comparison_scatter(
        self,
        model_data: dict,
        target: str,
        model_types: list[str],
        model_names: dict,
        model_colors: dict,
        best_highlight_color: str,
        output_suffix: str | None = None,
    ):
        """Plot scatter plots comparing three models for a single target."""
        output_path = (
            self.exp_root
            / f'model_comparison_scatter_{target}{"_" + output_suffix if output_suffix else ""}.png'
        )

        trans = TargetTransformer(0)
        best_mt = self.explorer.get_best_model_type(target)
        season_colors = {
            'Dry': '#e74c3c',
            'Normal': '#f39c12',
            'Rainy': '#3498db',
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, mt in enumerate(model_types):
            ax = axes[i]
            _, oof_metrics = model_data[mt][target]
            predictions = oof_metrics.oof_predictions.copy()

            # Apply log transform
            predictions[f'log_{target}'] = trans.transform(predictions[target])
            predictions[f'log_{target}_pred'] = trans.transform(predictions[f'{target}_pred'])

            y_true = predictions[f'log_{target}']
            y_pred = predictions[f'log_{target}_pred']
            r2 = oof_metrics.oof.NSE_log

            # Plot by season
            is_first = (i == 0)
            for season, color in season_colors.items():
                preds = predictions[predictions['Season'] == season]
                ax.scatter(
                    preds[f'log_{target}'],
                    preds[f'log_{target}_pred'],
                    c=color,
                    label=season if is_first else '',
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.3,
                    s=40,
                )

            # Identity line
            max_val = max(y_true.max(), y_pred.max())
            min_val = min(y_true.min(), y_pred.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.5)

            # Add (Best) marker to title if this is the best model
            best_marker = ' (Best)' if mt == best_mt else ''
            ax.set_title(
                f'{model_names[mt]}{best_marker}\n(R²={r2:.3f})',
                fontsize=14,
                fontweight='bold',
            )
            ax.set_xlabel('Measured (log)', fontsize=12)
            ax.set_ylabel('Predicted (log)', fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')

        # Add legend to first subplot
        axes[0].legend(loc='best', fontsize=10)

        fig.suptitle(
            f'Model Comparison for {target} (Measured vs Predicted)',
            fontsize=16,
            fontweight='bold',
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

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
                    s=40,
                )

            max_val = max(y_true.max(), y_pred.max())
            min_val = min(y_true.min(), y_pred.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.5)

            ax.set_title(f'{target}\n(R²={r2:.3f}, (log))', fontsize=16, fontweight='bold')
            ax.set_xlabel('Measured', fontsize=16)
            ax.set_ylabel('Predicted', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')

            if is_first:
                ax.legend(loc='best', fontsize=16)

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
            return (1, 1), (8, 8)
        elif total_plots == 2:
            return (1, 2), (8, 15)
        elif total_plots == 3:
            return (1, 3), (18, 6)
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
