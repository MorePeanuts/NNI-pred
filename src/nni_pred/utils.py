from nni_pred.data import MergedTabularDataset, SoilVariableGroups
import json
import pandas as pd
from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from nni_pred.evaluation import OOFMetrics
from nni_pred.data import MergedVariableGroups


@dataclass
class Details:
    best_seed: int
    best_model_type: str
    oof_metrics: OOFMetrics
    best_model_path: Path
    oof_predictions: pd.DataFrame
    features: pd.DataFrame


class Explorer:
    """
    Attributes:
        base_path: experiment output base path or inference output base path.
        etype: `exp` if exlore training output and `inf` while inference.
        targets: A list containing all target variables.
        metrics_summary: ...
        details: list of Details(best_seed, best_model_type, oof_metrics, best_model_path)
    """

    def __init__(self, base_path: Path, etype: Literal['exp', 'inf'] | None = None):
        self.base_path = base_path
        if base_path.name.startswith('exp') or etype == 'exp':
            self.etype = 'exp'
        elif base_path.name.startswith('inf') or etype == 'inf':
            self.etype = 'inf'

        if 'merged' in base_path.name:
            self.var_cls = MergedVariableGroups
        elif 'soil' in base_path.name:
            self.var_cls = SoilVariableGroups

        self.init_seed = int(self.base_path.name.split('_')[2])

        self.targets = []
        self.details: dict[str, Details] = {}
        for path in self.base_path.iterdir():
            if path.is_file():
                if path.name == 'metrics_summary.csv':
                    self.metrics_summary = pd.read_csv(path)
            else:
                seed_comp_path = path / 'seed_comparison.json'
                if not seed_comp_path.exists():
                    continue
                target = path.name
                self.targets.append(target)
                with seed_comp_path.open() as f:
                    seed_comp = json.load(f)
                    best_seed = seed_comp['best_seed']
                    best_model_type = seed_comp['best_model_type']
                    oof_metrics = OOFMetrics.from_json(seed_comp['best_metrics'])
                    best_model_path = Path(
                        path / f'seed_{best_seed}/{best_model_type}_model_for_{target}.joblib'
                    )
                    features = self.var_cls.get_feature_cols()
                    features = oof_metrics.oof_predictions[features]
                    self.details[target] = Details(
                        best_seed,
                        best_model_type,
                        oof_metrics,
                        best_model_path,
                        oof_metrics.oof_predictions,
                        features,
                    )

    def has_target(self, target) -> bool:
        return target in self.targets

    def get_targets_list(self) -> list[str]:
        return self.targets

    def get_best_model_path(self, target) -> Path:
        assert self.has_target(target)
        return self.details[target].best_model_path

    def get_oof_metrics(self, target) -> OOFMetrics:
        assert self.has_target(target)
        return self.details[target].oof_metrics

    def get_oof_predictions(self, target) -> pd.DataFrame:
        return self.details[target].oof_predictions

    def get_init_seed(self) -> int:
        return self.init_seed

    def get_best_model_type(self, target) -> str:
        return self.details[target].best_model_type

    def get_features(self, target) -> pd.DataFrame:
        return self.details[target].features
