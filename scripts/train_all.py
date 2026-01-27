"""
This script invokes the encapsulated trainer and seed selector for the formal execution of experiments,
 aiming to identify the optimal seeds and training outcomes.

The script does not include visualization steps.
"""

from nni_pred.evaluation import Comparator

import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger
from nni_pred.trainer import Trainer, SeedSelector
from nni_pred.data import VariableGroups


def main():
    logger.remove()

    targets = VariableGroups.targets_parent + VariableGroups.targets_metabolites
    trainer = Trainer(output_path=output_path, param_size=args.size, n_jobs=args.n_jobs)
    comparator = Comparator(indicator=args.indicator, cv_threshold=args.cv_threshold)
    seed_selector = SeedSelector(
        trainer, comparator, max_attempts=args.max_attempts, seed=args.init_seed
    )

    if 'all' in args.targets:
        seed_selector.run_exp(targets)
    else:
        seed_selector.run_exp([target for target in args.targets if target in targets])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-seed', default=42, type=int)
    parser.add_argument('--max-attempts', default=10, type=int)
    parser.add_argument('--targets', nargs='+', default=['all'])
    parser.add_argument('--output', type=str)
    parser.add_argument('--size', default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--n-jobs', default=-1, type=int)
    parser.add_argument('--cv-threshold', default=0.8, type=float)
    parser.add_argument('--indicator', default='NSE_log', type=str)
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        output_path = Path(__file__).parents[1] / f'output/exp_{args.init_seed}_{now}'
    output_path.mkdir(parents=True, exist_ok=True)

    main()
