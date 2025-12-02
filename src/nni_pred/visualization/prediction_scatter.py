"""
Prediction vs Measured Scatter Plot

This module generates scatter plots comparing predicted and measured pollutant
concentrations, with points colored by season.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from sklearn.metrics import r2_score


def plot_prediction_scatter(
    data_path: str,
    model_dir: str,
    pollutant: str,
    output_path: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    show_season_r2: bool = True,
) -> None:
    """
    Plot measured vs predicted values with seasonal coloring.

    Creates a single scatter plot with:
    - X-axis: Measured pollutant concentration
    - Y-axis: Predicted pollutant concentration
    - Points colored by season (Dry, Normal, Rainy)
    - 1:1 reference line
    - Overall R² and seasonal R² values

    Args:
        data_path: Path to processed_data.csv
        model_dir: Directory containing trained models (e.g., models/final_models)
        pollutant: Pollutant name (e.g., 'THIA', 'IMI', 'parentNNIs')
        output_path: Path to save figure (if None, show instead)
        figsize: Figure size (width, height)
        show_season_r2: Whether to show seasonal R² values in legend
    """
    # Load data
    df = pd.read_csv(data_path)

    # Check if pollutant exists
    if pollutant not in df.columns:
        raise ValueError(f'Pollutant {pollutant} not found in data')

    # Load model
    model_path = Path(model_dir) / pollutant / 'model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}\nPlease run final model training first.')

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_cols = model_data['feature_columns']

    # Prepare features and target
    X = df[feature_cols]
    y_true_log = df[pollutant].values  # Log-transformed

    # Predict (model predicts in log space)
    y_pred_log = model.predict(X)

    # Inverse transform to original scale
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    # Get season information
    seasons = df['Season'].values

    # Calculate overall R²
    r2_overall = r2_score(y_true, y_pred)

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
    ax.set_xlabel('Measured Concentration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Concentration', fontsize=13, fontweight='bold')

    title = f'{pollutant}: Measured vs Predicted\n(Overall R² = {r2_overall:.3f})'
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


def plot_all_pollutants_grid(
    data_path: str,
    model_dir: str,
    pollutants: list[str] | None = None,
    output_path: str | None = None,
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (5, 5),
) -> None:
    """
    Plot prediction scatter for multiple pollutants in a grid layout.

    Args:
        data_path: Path to processed_data.csv
        model_dir: Directory containing trained models
        pollutants: List of pollutants to plot (if None, plot all available)
        output_path: Path to save figure (if None, show instead)
        ncols: Number of columns in grid
        figsize_per_plot: Size of each subplot (width, height)
    """
    # Load data
    df = pd.read_csv(data_path)

    # Determine pollutants to plot
    if pollutants is None:
        # Find all available models
        model_dir_path = Path(model_dir)
        pollutants = [
            p.name
            for p in model_dir_path.iterdir()
            if p.is_dir() and (p / 'model.pkl').exists()
        ]

    n_pollutants = len(pollutants)
    nrows = (n_pollutants + ncols - 1) // ncols

    # Create figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )

    # Flatten axes for easier indexing
    if n_pollutants == 1:
        axes = np.array([axes])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

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
            # Load model
            model_path = Path(model_dir) / pollutant / 'model.pkl'
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            model = model_data['model']
            feature_cols = model_data['feature_columns']

            # Prepare data
            X = df[feature_cols]
            y_true = np.expm1(df[pollutant].values)
            y_pred = np.expm1(model.predict(X))
            seasons = df['Season'].values

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
            ax.set_title(f'{pollutant}\n(R²={r2:.3f})', fontsize=11, fontweight='bold')
            ax.set_xlabel('Measured', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')

            if idx == 0:
                ax.legend(loc='best', fontsize=8)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(pollutant, fontsize=11)

    # Hide unused subplots
    for idx in range(n_pollutants, len(axes)):
        axes[idx].axis('off')

    # Overall title
    fig.suptitle(
        'Measured vs Predicted Concentrations (All Pollutants)',
        fontsize=15,
        fontweight='bold',
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'\nGrid scatter plot saved to: {output_path}')
    else:
        plt.show()

    plt.close()
