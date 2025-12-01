#!/usr/bin/env python
"""
Train all pollutant prediction models with nested spatial CV.

This script coordinates the entire training pipeline:
1. Loads processed data
2. Runs nested CV for each pollutant-model combination
3. Selects best model per pollutant
4. Trains final models on full dataset
5. Generates reports and saves results

Usage:
    # Test on single pollutant with small grid (2-4 hours)
    uv run scripts/train_nested_cv.py --pollutants THIA --grid-size small

    # Test on 3 pollutants with medium grid (6-12 hours)
    uv run scripts/train_nested_cv.py --pollutants THIA IMI parentNNIs --grid-size medium

    # Full run on all 11 pollutants (5-6 days)
    uv run scripts/train_nested_cv.py --pollutants all --grid-size full
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import nni_pred
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nni_pred.training import BatchTrainer, FinalModelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train NNI prediction models with nested spatial cross-validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on single pollutant with small grid
  %(prog)s --pollutants THIA --grid-size small

  # Test on 3 pollutants with medium grid
  %(prog)s --pollutants THIA IMI parentNNIs --grid-size medium

  # Full run on all pollutants
  %(prog)s --pollutants all --grid-size full

Grid sizes:
  small:  Reduced hyperparameter grids for fast testing (2-4 hours per pollutant)
  medium: Moderate grids for validation (6-12 hours for 3 pollutants)
  full:   Complete grids for final results (5-6 days for all pollutants)
        """
    )

    parser.add_argument(
        '--pollutants',
        nargs='+',
        default=['all'],
        help='Pollutants to train (space-separated). Use "all" for all 11 pollutants. '
             'Available: THIA IMI CLO ACE DIN parentNNIs IMI-UREA DN-IMI DM-ACE CLO-UREA mNNIs'
    )

    parser.add_argument(
        '--grid-size',
        type=str,
        choices=['small', 'medium', 'full'],
        default='small',
        help='Hyperparameter grid size: small (fast), medium (moderate), full (exhaustive)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to processed_data.csv (default: datasets/processed_data.csv)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: models/nested_cv_results)'
    )

    parser.add_argument(
        '--n-outer',
        type=int,
        default=5,
        help='Number of outer CV folds (default: 5)'
    )

    parser.add_argument(
        '--n-inner',
        type=int,
        default=4,
        help='Number of inner CV folds (default: 4)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Verbosity level: 0 (silent), 1 (normal), 2 (detailed)'
    )

    parser.add_argument(
        '--skip-final-training',
        action='store_true',
        help='Skip final model training on full dataset (only run CV)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent

    if args.data_path is None:
        data_path = project_root / 'datasets' / 'processed_data.csv'
    else:
        data_path = Path(args.data_path)

    if args.output_dir is None:
        output_dir = project_root / 'models' / 'nested_cv_results'
    else:
        output_dir = Path(args.output_dir)

    # Check if data file exists
    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_path}")
        print("\nPlease run the preprocessing script first:")
        print("  uv run scripts/idw_merge.py")
        print("\nThis will generate the required processed_data.csv file.")
        sys.exit(1)

    # Print configuration
    print(f"\n{'='*80}")
    print("NESTED SPATIAL CROSS-VALIDATION FOR NNI PREDICTION")
    print(f"{'='*80}\n")

    print("Configuration:")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Pollutants: {', '.join(args.pollutants)}")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Outer CV folds: {args.n_outer}")
    print(f"  Inner CV folds: {args.n_inner}")
    print(f"  Random state: {args.random_state}")
    print(f"  Verbose: {args.verbose}")

    # Estimate time
    n_pollutants = 11 if args.pollutants == ['all'] else len(args.pollutants)
    if args.grid_size == 'small':
        est_time = f"{n_pollutants * 2}-{n_pollutants * 4} hours"
    elif args.grid_size == 'medium':
        est_time = f"{n_pollutants * 4}-{n_pollutants * 6} hours"
    else:  # full
        est_time = f"{n_pollutants * 12}-{n_pollutants * 14} hours ({n_pollutants * 12 / 24:.1f}-{n_pollutants * 14 / 24:.1f} days)"

    print(f"\nEstimated time: {est_time}")
    print(f"{'='*80}\n")

    # Ask for confirmation if full run
    if args.grid_size == 'full' and n_pollutants > 3:
        response = input(f"This will take approximately {est_time}. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Step 1: Nested CV
    print(f"\n{'='*80}")
    print("STEP 1: NESTED SPATIAL CROSS-VALIDATION")
    print(f"{'='*80}\n")

    trainer = BatchTrainer(
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    try:
        cv_results = trainer.train_all_combinations(
            data_path=str(data_path),
            output_dir=str(output_dir),
            pollutants=args.pollutants,
            grid_size=args.grid_size,
        )

        print(f"\n{'='*80}")
        print("STEP 1 COMPLETE")
        print(f"{'='*80}")
        print(f"\nCross-validation results saved to:")
        print(f"  {output_dir / 'nested_cv_results.csv'}")
        print(f"  {output_dir / 'model_selection_report.txt'}")

    except Exception as e:
        print(f"\nERROR during nested CV: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Final model training
    if not args.skip_final_training:
        print(f"\n{'='*80}")
        print("STEP 2: TRAINING FINAL MODELS ON FULL DATASET")
        print(f"{'='*80}\n")

        final_trainer = FinalModelTrainer(verbose=args.verbose)

        try:
            final_trainer.train_final_models(
                data_path=str(data_path),
                cv_results_path=str(output_dir / 'nested_cv_results.csv'),
                output_dir=str(output_dir.parent),
            )

            print(f"\n{'='*80}")
            print("STEP 2 COMPLETE")
            print(f"{'='*80}")
            print(f"\nFinal models saved to:")
            print(f"  {output_dir.parent / 'final_models'}/")

        except Exception as e:
            print(f"\nERROR during final model training: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING PIPELINE COMPLETE!")
    print(f"{'='*80}\n")

    print("Results saved to:")
    print(f"  CV results: {output_dir / 'nested_cv_results.csv'}")
    print(f"  Detailed results: {output_dir / 'detailed_results.json'}")
    print(f"  Report: {output_dir / 'model_selection_report.txt'}")

    if not args.skip_final_training:
        print(f"  Final models: {output_dir.parent / 'final_models'}/")

    print("\nNext steps:")
    print("  1. Review model_selection_report.txt")
    print("  2. Check cross-validation metrics (R², RMSE, MAE)")

    if not args.skip_final_training:
        print("  3. Use final models for SHAP analysis")
        print("  4. Interpret feature importances")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
