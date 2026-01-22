import argparse
import joblib
from pathlib import Path
from nni_pred.utils import Explorer
from datetime import datetime
from nni_pred.models import ElasticNetBuilder, RandomForestBuilder, XGBoostBuilder
from nni_pred.data import MergedTabularDataset
from nni_pred.evaluation import Evaluator, Metrics


def main(args):
    exp_path = Path(args.exp_path)
    explorer = Explorer(exp_path)
    random_state = explorer.get_init_seed()
    if args.output:
        output_path = Path(args.output_path)
    else:
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        output_path = Path(__file__).parents[1] / f'output/inf_{random_state}_{now}'
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = MergedTabularDataset()
    X, y_dict, groups = dataset.prepare_data()
    targets = explorer.get_targets_list()
    for target in targets:
        y = y_dict[target].values
        model_path = explorer.get_best_model_path(target)
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        metrics = Metrics.from_predictions(y, y_pred, 1)
        print(f'=============={target}==============')
        print(metrics)
        print('=====================================')
        # TODO: save inference results.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    args = parser.parse_args()
    main(args)
