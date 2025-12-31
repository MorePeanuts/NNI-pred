import json
import argparse
import pandas as pd
from loguru import logger
from pathlib import Path
from nni_pred.trainer import SeedSelector
from nni_pred.evaluation import Comparator, OOFMetrics


def main():
    comparator = Comparator(cv_threshold=args.cv_threshold)
    rows = []
    for target_dir in exp_path.iterdir():
        for seed_dir in target_dir.iterdir():
            comparator.compare_model(seed_dir)
        comparator.compare_seed(target_dir)
        target = target_dir.name
        with (target_dir / 'seed_comparison.json').open() as f:
            s = json.load(f)
            best_metrics = OOFMetrics.from_json(s['best_metrics'])
            row = {'target': target, 'seed': s['best_seed'], 'model': s['best_model_type']}
            row.update(best_metrics.to_format_dict())
            rows.append(row)

    metrics_summary = pd.DataFrame(rows)
    metrics_summary = metrics_summary.set_index('target')
    logger.info(f'Summary all targets.\n{metrics_summary}')
    summary_path = exp_path / 'metrics_summary.csv'
    metrics_summary.to_csv(summary_path, index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', type=str)
    parser.add_argument('--cv-threshold', default=0.5, type=float)
    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    assert exp_path.exists(), f'{exp_path} doesnot exist.'
    main()
