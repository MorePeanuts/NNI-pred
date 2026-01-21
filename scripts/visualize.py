import json
from pathlib import Path
from nni_pred.visualization import Visualizer


exp_path = Path(__file__).parents[1] / 'output/exp_100_251231_064918'
visualizer = Visualizer(exp_path)
visualizer.plot_cv_metrics()
