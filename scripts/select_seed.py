import json
import argparse
import pandas as pd
from loguru import logger
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from nni_pred.trainer import Trainer
from nni_pred.evaluation import Comparator, OOFMetrics
from nni_pred.data import MergedVariableGroups, SoilVariableGroups


def main():
    comparator = Comparator(indicator=args.indicator, cv_threshold=args.cv_threshold)
    rows = []

    for target_dir in exp_path.iterdir():
        if not target_dir.is_dir():
            continue

        seed_dirs = [d for d in target_dir.iterdir() if d.is_dir()]
        logger.info(f'Processing {target_dir.name}: {len(seed_dirs)} seeds')

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(comparator.compare_model, seed_dir) for seed_dir in seed_dirs
            ]
            for future in as_completed(futures):
                future.result()

        comparator.compare_seed(target_dir)

        target = target_dir.name
        if not (target_dir / 'seed_comparison.json').exists():
            logger.warning(f'No best seed found in {target_dir}')
            continue
        with (target_dir / 'seed_comparison.json').open() as f:
            s = json.load(f)
            best_metrics = OOFMetrics.from_json(s['best_metrics'])
            row = {'target': target, 'seed': s['best_seed'], 'model': s['best_model_type']}
            row.update(best_metrics.to_format_dict())
            rows.append(row)

        # Final train
        trainer = Trainer(var_cls, output_path=exp_path, param_size=args.size)
        trainer.train(
            row['target'],
            model_type=row['model'],  # type: ignore
            random_state=row['seed'],  # type: ignore
            run_nested_cv=False,
        )

    (exp_path / 'metrics_summary.csv').unlink(missing_ok=True)
    if len(rows) > 0:
        metrics_summary = pd.DataFrame(rows)
        metrics_summary = metrics_summary.set_index('target')
        logger.info(f'Summary all targets.\n{Comparator.format_summary_table(metrics_summary)}')
        summary_path = exp_path / 'metrics_summary.csv'
        metrics_summary.to_csv(summary_path, index=True)
    else:
        logger.warning(f'All targets failed to find best seed in {exp_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', type=str)
    parser.add_argument('--cv-threshold', default=0.8, type=float)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--indicator', default='NSE_log', type=str)
    parser.add_argument('--size', default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--cls', choices=['merged', 'soil'], type=str, default='merged')
    args = parser.parse_args()
    match args.cls:
        case 'merged':
            var_cls = MergedVariableGroups
        case 'soil':
            var_cls = SoilVariableGroups
    exp_path = Path(args.exp_path)
    assert exp_path.exists(), f'{exp_path} doesnot exist.'
    main()
