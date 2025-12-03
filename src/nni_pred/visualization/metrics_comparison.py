"""
Cross-Validation Metrics Comparison Visualization

This module generates horizontal bar charts comparing model performance
across different pollutants for R², RMSE, and MAE metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_cv_metrics(
    cv_results_path: str,
    output_path: str | None = None,
    figsize: tuple[float, float] = (15, 8),
    show_std: bool = True,
    best_only: bool = True,
) -> None:
    """
    Plot cross-validation metrics comparison (R², RMSE, MAE).

    Creates a 1x3 subplot figure with horizontal bar charts:
    - Left: R² scores (higher is better)
    - Middle: RMSE values (lower is better)
    - Right: MAE values (lower is better)

    Args:
        cv_results_path: Path to nested_cv_results.csv
        output_path: Path to save figure (if None, show instead)
        figsize: Figure size (width, height)
        show_std: Whether to show error bars (std)
        best_only: Whether to show only best models per pollutant
    """
    # Load CV results
    df = pd.read_csv(cv_results_path)

    # Filter to best models only if requested
    if best_only and 'best_model' in df.columns:
        df = df[df['best_model'] == True].copy()  # noqa: E712
        title_suffix = ' (Best Models)'
    else:
        title_suffix = ''

    # Sort by mean R² (descending)
    df = df.sort_values('mean_r2', ascending=True)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Define colors for different metrics
    color_r2 = '#2ecc71'  # Green
    color_rmse = '#e74c3c'  # Red
    color_mae = '#f39c12'  # Orange

    # Y-axis labels (pollutant names)
    if best_only:
        y_labels = df['pollutant'].values
    else:
        # Include model name if showing all models
        y_labels = [f"{row['pollutant']} ({row['model_name']})" for _, row in df.iterrows()]

    y_pos = np.arange(len(df))

    # --- Subplot 1: R² Score ---
    ax1 = axes[0]
    bars1 = ax1.barh(y_pos, df['mean_r2'], color=color_r2, alpha=0.7, edgecolor='black')

    if show_std:
        ax1.errorbar(
            df['mean_r2'],
            y_pos,
            xerr=df['std_r2'],
            fmt='none',
            ecolor='black',
            capsize=3,
            alpha=0.5,
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=10)
    ax1.set_xlabel('R² Score (log space)', fontsize=12, fontweight='bold')
    ax1.set_title(f'R² Score (log){title_suffix}', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=0, color='black', linewidth=0.8)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, df['mean_r2'], df['std_r2'])):
        label = f'{val:.3f}' if not show_std else f'{val:.3f}±{std:.3f}'
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2, label, va='center', fontsize=8)

    # --- Subplot 2: RMSE ---
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, df['mean_rmse'], color=color_rmse, alpha=0.7, edgecolor='black')

    if show_std:
        ax2.errorbar(
            df['mean_rmse'],
            y_pos,
            xerr=df['std_rmse'],
            fmt='none',
            ecolor='black',
            capsize=3,
            alpha=0.5,
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Hide y-axis labels (already shown in left plot)
    ax2.set_xlabel('RMSE (original scale)', fontsize=12, fontweight='bold')
    ax2.set_title(f'RMSE (original){title_suffix}', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.axvline(x=0, color='black', linewidth=0.8)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars2, df['mean_rmse'], df['std_rmse'])):
        label = f'{val:.3f}' if not show_std else f'{val:.3f}±{std:.3f}'
        ax2.text(val + max(df['mean_rmse']) * 0.01, bar.get_y() + bar.get_height() / 2, label, va='center', fontsize=8)

    # --- Subplot 3: MAE ---
    ax3 = axes[2]
    bars3 = ax3.barh(y_pos, df['mean_mae'], color=color_mae, alpha=0.7, edgecolor='black')

    if show_std:
        ax3.errorbar(
            df['mean_mae'],
            y_pos,
            xerr=df['std_mae'],
            fmt='none',
            ecolor='black',
            capsize=3,
            alpha=0.5,
        )

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([])  # Hide y-axis labels
    ax3.set_xlabel('MAE (original scale)', fontsize=12, fontweight='bold')
    ax3.set_title(f'MAE (original){title_suffix}', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.axvline(x=0, color='black', linewidth=0.8)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars3, df['mean_mae'], df['std_mae'])):
        label = f'{val:.3f}' if not show_std else f'{val:.3f}±{std:.3f}'
        ax3.text(val + max(df['mean_mae']) * 0.01, bar.get_y() + bar.get_height() / 2, label, va='center', fontsize=8)

    # Overall title
    fig.suptitle(
        'Nested Spatial Cross-Validation Results',
        fontsize=15,
        fontweight='bold',
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'\nMetrics comparison plot saved to: {output_path}')
    else:
        plt.show()

    plt.close()
