import argparse
from pathlib import Path
from datetime import datetime
from nni_pred.trainer import Trainer, SeedSelector
from nni_pred.data import get_feature_groups


def main():
    feature_groups = get_feature_groups()
    targets = feature_groups.targets
    trainer = Trainer(output_path=output_path, param_size=args.size)
    seed_selector = SeedSelector(trainer, max_attempts=args.max_attempts, seed=args.init_seed)

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
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        output_path = Path(__file__).parents[1] / f'output/exp_{args.init_seed}_{now}'
    output_path.mkdir(parents=True, exist_ok=True)

    main()
