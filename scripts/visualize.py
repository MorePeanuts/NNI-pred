import json
from pathlib import Path
from nni_pred.visualization import Visualizer
from nni_pred.data import MergedTabularDataset


exp_path = Path(__file__).parents[1] / 'output/exp_100_251231_064918'
visualizer = Visualizer(exp_path)
# visualizer.plot_cv_metrics()
# targets = [
#     # Parent compounds
#     'THIA',  # Thiamethoxam
#     'IMI',  # Imidacloprid
#     'CLO',  # Clothianidin
#     'ACE',  # Acetamiprid
#     'DIN',  # Dinotefuran
#     'parentNNIs',  # Sum of parent neonicotinoids
#     # Metabolites
#     'IMI-UREA',  # Imidacloprid-urea
#     'DN-IMI',  # Desmethyl-imidacloprid
#     'DM-ACE',  # Desmethyl-acetamiprid
#     'CLO-UREA',  # Clothianidin-urea
#     'mNNIs',  # Sum of metabolites
# ]
# visualizer.plot_scatter_identity(
#     targets_used=targets[:5] + targets[6:10], output_suffix='individual_log', use_log=True
# )
# visualizer.plot_scatter_identity(
#     targets_used=targets[:5] + targets[6:10],
#     output_suffix='individual',
# )
# visualizer.plot_scatter_identity(
#     targets_used=['parentNNIs', 'mNNIs'], output_suffix='total_log', use_log=True
# )
# visualizer.plot_scatter_identity(targets_used=['parentNNIs', 'mNNIs'], output_suffix='total')
visualizer.plot_shap_importance(targets_used=['parentNNIs', 'mNNIs'])
visualizer.plot_shap_summary(targets_used=['parentNNIs', 'mNNIs'])
