"""
Prediction vs Measured Scatter Plot (Using Out-of-Fold Predictions)

This module generates scatter plots comparing predicted and measured pollutant
concentrations using out-of-fold predictions from nested cross-validation,
with points colored by season.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score


def get_pollutant_categories():
    """
    Categorize pollutants into individual compounds and total concentrations.

    Returns:
        Dictionary with:
        - 'individual': List of 9 individual pollutant names
        - 'total': List of 2 total concentration names
    """
    return {
        'individual': [
            'THIA',      # Thiamethoxam
            'IMI',       # Imidacloprid
            'CLO',       # Clothianidin
            'ACE',       # Acetamiprid
            'DIN',       # Dinotefuran
            'IMI-UREA',  # Imidacloprid-urea
            'DN-IMI',    # Desmethyl-imidacloprid
            'DM-ACE',    # Desmethyl-acetamiprid
            'CLO-UREA',  # Clothianidin-urea
        ],
        'total': [
            'parentNNIs',  # Sum of parent neonicotinoids
            'mNNIs',       # Sum of metabolites
        ],
    }


def plot_prediction_scatter(
    oof_predictions_path: str,
    pollutant: str,
    output_path: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    show_season_r2: bool = True,
    use_best_model: bool = True,
) -> None:
    """
    Plot measured vs predicted values with seasonal coloring.

    Creates a single scatter plot with:
    - X-axis: Measured pollutant concentration
    - Y-axis: Predicted pollutant concentration (from out-of-fold CV)
    - Points colored by season (Dry, Normal, Rainy)
    - 1:1 reference line
    - Overall R² and seasonal R² values

    Args:
        oof_predictions_path: Path to oof_predictions.csv from nested CV training
        pollutant: Pollutant name (e.g., 'THIA', 'IMI', 'parentNNIs')
        output_path: Path to save figure (if None, show instead)
        figsize: Figure size (width, height)
        show_season_r2: Whether to show seasonal R² values in legend
        use_best_model: Whether to use best model predictions (True) or
                        specific model (False, requires specifying model_name)
    """
    # Load OOF predictions
    df = pd.read_csv(oof_predictions_path)

    # Check if pollutant exists (use log-space values)
    true_col = f'{pollutant}_true_log'
    pred_col = f'{pollutant}_pred_best_log' if use_best_model else None

    if true_col not in df.columns:
        raise ValueError(f'Pollutant {pollutant} not found in OOF predictions')

    if pred_col not in df.columns:
        raise ValueError(f'Prediction column {pred_col} not found in OOF predictions')

    # Get true and predicted values (in log space)
    y_true = df[true_col].values
    y_pred = df[pred_col].values

    # Remove NaN values (if any folds failed)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    seasons = df['Season'].values[valid_mask]

    if len(y_true) == 0:
        raise ValueError(f'No valid predictions found for {pollutant}')

    # Calculate overall R²
    r2_overall = r2_score(y_true, y_pred)

    # Get best model name
    best_model_col = f'{pollutant}_best_model'
    best_model = df[best_model_col].iloc[0] if best_model_col in df.columns else 'Best Model'

    # Calculate seasonal R²
    seasonal_r2 = {}
    for season in ['Dry', 'Normal', 'Rainy']:
        mask = seasons == season
        if mask.sum() > 0:
            seasonal_r2[season] = r2_score(y_true[mask], y_pred[mask])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for seasons
    season_colors = {
        'Dry': '#e74c3c',  # Red
        'Normal': '#f39c12',  # Orange
        'Rainy': '#3498db',  # Blue
    }

    # Plot scatter by season
    for season in ['Dry', 'Normal', 'Rainy']:
        mask = seasons == season
        if mask.sum() > 0:
            label = season
            if show_season_r2 and season in seasonal_r2:
                label += f' (R²={seasonal_r2[season]:.3f})'

            ax.scatter(
                y_true[mask],
                y_pred[mask],
                c=season_colors[season],
                label=label,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5,
                s=80,
            )

    # Plot 1:1 line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line', alpha=0.5)

    # Labels and title
    ax.set_xlabel('Measured Concentration (log1p scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Concentration (log1p scale)', fontsize=13, fontweight='bold')

    title = f'{pollutant}: Measured vs Predicted ({best_model})\n(Overall R² = {r2_overall:.3f} in log space, Out-of-Fold CV)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Grid and legend
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Equal aspect ratio for better visual comparison
    ax.set_aspect('equal', adjustable='datalim')

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'\nPrediction scatter plot saved to: {output_path}')
    else:
        plt.show()

    plt.close()


def plot_individual_pollutants_grid(
    oof_predictions_path: str,
    output_path: str | None = None,
    figsize_per_plot: tuple[float, float] = (5, 5),
) -> None:
    """
    Plot prediction scatter for 9 individual pollutants in a 3×3 grid.

    Args:
        oof_predictions_path: Path to oof_predictions.csv from nested CV training
        output_path: Path to save figure (if None, show instead)
        figsize_per_plot: Size of each subplot (width, height)
    """
    # Load OOF predictions
    df = pd.read_csv(oof_predictions_path)

    # Get individual pollutants (3×3 grid)
    categories = get_pollutant_categories()
    pollutants = categories['individual']

    # Check which pollutants are available
    available_pollutants = []
    for p in pollutants:
        if f'{p}_true_log' in df.columns:
            available_pollutants.append(p)

    if len(available_pollutants) == 0:
        print('WARNING: No individual pollutants found in OOF predictions')
        return

    pollutants = available_pollutants

    # Create 3×3 grid
    ncols = 3
    nrows = 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Season colors
    season_colors = {
        'Dry': '#e74c3c',
        'Normal': '#f39c12',
        'Rainy': '#3498db',
    }

    # Plot each pollutant
    for idx, pollutant in enumerate(pollutants):
        ax = axes[idx]

        try:
            # Get true and predicted values (in log space)
            true_col = f'{pollutant}_true_log'
            pred_col = f'{pollutant}_pred_best_log'

            y_true = df[true_col].values
            y_pred = df[pred_col].values
            seasons = df['Season'].values

            # Remove NaN values
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            seasons = seasons[valid_mask]

            # Calculate R²
            r2 = r2_score(y_true, y_pred)

            # Plot by season
            for season in ['Dry', 'Normal', 'Rainy']:
                mask = seasons == season
                if mask.sum() > 0:
                    ax.scatter(
                        y_true[mask],
                        y_pred[mask],
                        c=season_colors[season],
                        label=season if idx == 0 else '',  # Only show legend in first plot
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.3,
                        s=30,
                    )

            # 1:1 line
            max_val = max(y_true.max(), y_pred.max())
            min_val = min(y_true.min(), y_pred.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.5)

            # Title and labels
            ax.set_title(f'{pollutant}\n(R²={r2:.3f}, log space)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Measured (log1p)', fontsize=10)
            ax.set_ylabel('Predicted (log1p)', fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')

            if idx == 0:
                ax.legend(loc='best', fontsize=8)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(pollutant, fontsize=11)

    # Hide unused subplots
    for idx in range(len(pollutants), len(axes)):
        axes[idx].axis('off')

    # Overall title
    fig.suptitle(
        'Individual Pollutants: Measured vs Predicted (Out-of-Fold, Best Models)',
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'\nIndividual pollutants scatter plot saved to: {output_path}')
    else:
        plt.show()

    plt.close()


def plot_total_pollutants_grid(
    oof_predictions_path: str,
    output_path: str | None = None,
    figsize_per_plot: tuple[float, float] = (6, 6),
) -> None:
    """
    Plot prediction scatter for 2 total pollutants (parentNNIs, mNNIs) in a 1×2 grid.

    Args:
        oof_predictions_path: Path to oof_predictions.csv from nested CV training
        output_path: Path to save figure (if None, show instead)
        figsize_per_plot: Size of each subplot (width, height)
    """
    # Load OOF predictions
    df = pd.read_csv(oof_predictions_path)

    # Get total pollutants (1×2 grid)
    categories = get_pollutant_categories()
    pollutants = categories['total']

    # Check which pollutants are available
    available_pollutants = []
    for p in pollutants:
        if f'{p}_true_log' in df.columns:
            available_pollutants.append(p)

    if len(available_pollutants) == 0:
        print('WARNING: No total pollutants found in OOF predictions')
        return

    pollutants = available_pollutants

    # Create 1×2 grid
    ncols = len(pollutants)
    nrows = 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )

    # Ensure axes is iterable
    if ncols == 1:
        axes = [axes]

    # Season colors
    season_colors = {
        'Dry': '#e74c3c',
        'Normal': '#f39c12',
        'Rainy': '#3498db',
    }

    # Plot each pollutant
    for idx, pollutant in enumerate(pollutants):
        ax = axes[idx]

        try:
            # Get true and predicted values (in log space)
            true_col = f'{pollutant}_true_log'
            pred_col = f'{pollutant}_pred_best_log'

            y_true = df[true_col].values
            y_pred = df[pred_col].values
            seasons = df['Season'].values

            # Remove NaN values
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            seasons = seasons[valid_mask]

            # Calculate R²
            r2 = r2_score(y_true, y_pred)

            # Plot by season
            for season in ['Dry', 'Normal', 'Rainy']:
                mask = seasons == season
                if mask.sum() > 0:
                    ax.scatter(
                        y_true[mask],
                        y_pred[mask],
                        c=season_colors[season],
                        label=season,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5,
                        s=60,
                    )

            # 1:1 line
            max_val = max(y_true.max(), y_pred.max())
            min_val = min(y_true.min(), y_pred.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)

            # Title and labels
            ax.set_title(f'{pollutant}\n(R²={r2:.3f}, log space)', fontsize=13, fontweight='bold')
            ax.set_xlabel('Measured (log1p)', fontsize=11)
            ax.set_ylabel('Predicted (log1p)', fontsize=11)
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=10)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(pollutant, fontsize=13)

    # Overall title
    fig.suptitle(
        'Total Pollutant Concentrations: Measured vs Predicted (Out-of-Fold, Best Models)',
        fontsize=16,
        fontweight='bold',
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'\nTotal pollutants scatter plot saved to: {output_path}')
    else:
        plt.show()

    plt.close()
