import json
import argparse
from pathlib import Path
from nni_pred.visualization import Visualizer
from nni_pred.data import MergedTabularDataset, MergedVariableGroups, SoilVariableGroups


def main(args):
    if 'soil' in args.path:
        group = SoilVariableGroups
    elif 'merged' in args.path:
        group = MergedVariableGroups

    exp_path = Path(args.path)
    visualizer = Visualizer(exp_path, use_shap=True)
    visualizer.plot_cv_metrics()
    for tgt in group.targets_parent + group.targets_metabolites:
        visualizer.plot_scatter_identity(targets_used=[tgt], output_suffix=tgt, use_log=True)
        visualizer.plot_shap_importance(targets_used=[tgt], output_suffix=tgt)
        visualizer.plot_shap_summary(targets_used=[tgt], output_suffix=tgt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    main(args)
