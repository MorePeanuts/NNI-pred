import json
import argparse
from pathlib import Path
from nni_pred.visualization import Visualizer
from nni_pred.data import MergedTabularDataset


# exp_path = Path(__file__).parents[1] / 'output/exp_42_260127_171406'
# visualizer = Visualizer(exp_path, use_shap=False)
# visualizer.plot_cv_metrics()
# visualizer.plot_scatter_identity(targets_used=None, output_suffix='individual_log', use_log=True)
# visualizer.plot_scatter_identity(
#     targets_used=targets[:5] + targets[6:10],
#     output_suffix='individual',
# )
# visualizer.plot_scatter_identity(
#     targets_used=['parentNNIs', 'mNNIs'], output_suffix='total_log', use_log=True
# )
# visualizer.plot_scatter_identity(targets_used=['parentNNIs', 'mNNIs'], output_suffix='total')
# visualizer.plot_shap_importance(targets_used=None)
# visualizer.plot_shap_summary(targets_used=None)


def main():
    exp_path = Path(args.path)
    visualizer = Visualizer(exp_path, use_shap=True)
    visualizer.plot_cv_metrics()
    visualizer.plot_scatter_identity(output_suffix='log', use_log=True)
    visualizer.plot_shap_importance()
    visualizer.plot_shap_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    main()
