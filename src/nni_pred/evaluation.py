import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
from loguru import logger


@dataclass
class Metrics:
    NSE_log: float
    RSR_log: float
    NSE: float
    RSR: float
    PBIAS: float
    KGE: float

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray, offset: float):
        y_true_log = np.log(y_true + offset)
        y_pred_log = np.log(y_pred + offset)
        assert not np.isnan(y_true_log).any(), f'offset={offset}\ny_true={y_true}'
        assert not np.isnan(y_pred_log).any(), f'offset={offset}\ny_pred={y_pred}'
        return cls(
            NSE_log=cls.calc_nse(y_true_log, y_pred_log),
            RSR_log=cls.calc_rsr(y_true_log, y_pred_log),
            NSE=cls.calc_nse(y_true, y_pred),
            RSR=cls.calc_rsr(y_true, y_pred),
            PBIAS=cls.calc_pbias(y_true, y_pred),
            KGE=cls.calc_kge(y_true, y_pred),
        )

    @staticmethod
    def calc_nse(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def calc_rsr(y_true, y_pred):
        """
        RMSE-observations Standard Deviation Ratio (RSR)
        RSR = RMSE / STDEV_obs
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        stdev_obs = np.std(y_true)
        if stdev_obs == 0:
            return np.nan
        return rmse / stdev_obs

    @staticmethod
    def calc_pbias(y_true, y_pred):
        """
        Percent Bias (PBIAS)
        Measure the average trend of model predictions deviating from observed values.
        """
        total_error = np.sum(y_true - y_pred)
        total_obs = np.sum(y_true)
        if total_obs == 0:
            return np.nan
        return (total_error / total_obs) * 100

    @staticmethod
    def calc_kge(y_true, y_pred):
        """
        Kling-Gupta Efficiency (KGE)
        Considering both correlation, bias, and variability
        """
        r = np.corrcoef(y_true, y_pred)[0, 1]
        alpha = np.std(y_pred) / np.std(y_true)
        beta = np.mean(y_pred) / np.mean(y_true)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return kge

    def __repr__(self):
        data = [[k, f'{v:.4f}'] for k, v in asdict(self).items()]
        table = tabulate(data, headers=['Metric', 'Value'], tablefmt='fancy_grid', numalign='right')
        return f'{table}\n'


@dataclass
class FoldInformation:
    fold: int
    best_param: dict
    metrics: Metrics
    best_inner_score: float
    target_col: str
    offset: float

    def __repr__(self):
        return (
            f'Fold: {self.fold}\tBest Inner Score: {self.best_inner_score}\tTarget: {self.target_col}\n'
            + json.dumps(self.best_param, indent=2, ensure_ascii=False)
            + '\n'
        )


@dataclass
class OOFMetrics:
    mean: Metrics
    std: Metrics
    oof: Metrics
    oof_predictions: pd.DataFrame
    fold_infos: list[FoldInformation]

    @classmethod
    def from_fold_information_list(
        cls, fold_infos: list[FoldInformation], oof_predictions: pd.DataFrame
    ):
        target_col = fold_infos[0].target_col
        mean = Metrics(
            NSE_log=np.mean([info.metrics.NSE_log for info in fold_infos]),
            RSR_log=np.mean([info.metrics.RSR_log for info in fold_infos]),
            NSE=np.mean([info.metrics.NSE for info in fold_infos]),
            RSR=np.mean([info.metrics.RSR for info in fold_infos]),
            PBIAS=np.mean([info.metrics.PBIAS for info in fold_infos]),
            KGE=np.mean([info.metrics.KGE for info in fold_infos]),
        )
        std = Metrics(
            NSE_log=np.std([info.metrics.NSE_log for info in fold_infos]),
            RSR_log=np.std([info.metrics.RSR_log for info in fold_infos]),
            NSE=np.std([info.metrics.NSE for info in fold_infos]),
            RSR=np.std([info.metrics.RSR for info in fold_infos]),
            PBIAS=np.std([info.metrics.PBIAS for info in fold_infos]),
            KGE=np.std([info.metrics.KGE for info in fold_infos]),
        )
        mean_offset = np.mean([info.offset for info in fold_infos])
        oof = Metrics.from_predictions(
            oof_predictions[target_col].values,
            oof_predictions[target_col + '_pred'].values,
            mean_offset,
        )
        return cls(
            mean=mean,
            std=std,
            oof=oof,
            oof_predictions=oof_predictions,
            fold_infos=fold_infos,
        )

    def __repr__(self):
        target_col = self.fold_infos[0].target_col
        mean_dict = asdict(self.mean)
        std_dict = asdict(self.std)
        oof_dict = asdict(self.oof)
        data = [[k, oof_dict[k], mean_dict[k], std_dict[k]] for k in oof_dict.keys()]
        title = f'OOF Metrics for {target_col}'
        table = tabulate(
            data, headers=['Metric', 'OOF', 'mean', 'std'], tablefmt='fancy_grid', numalign='right'
        )
        return f'\n####{title}####\n{table}'


class Evaluator:
    def __init__(
        self,
        target_col: str,
        model_name: str,
        k_fold: int = 5,
    ):
        self.k_fold = k_fold
        self.target_col = target_col
        self.model_name = model_name
        self.fold_infos = []

    def update(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        fold: int,
        offset: float,
        best_param: dict,
        best_inner_score: float,
        report: bool = False,
    ):
        y_true = y_true.values
        if not hasattr(self, 'oof_predictions'):
            self.oof_predictions = X.copy()
            self.oof_predictions[self.target_col] = y_true
            self.oof_predictions[self.target_col + '_pred'] = y_pred
        else:
            df = X.copy()
            df[self.target_col] = y_true
            df[self.target_col + '_pred'] = y_pred
            self.oof_predictions = pd.concat([self.oof_predictions, df], ignore_index=True)

        metrics = Metrics.from_predictions(y_true, y_pred, offset)
        fold_info = FoldInformation(
            fold, best_param, metrics, best_inner_score, self.target_col, offset
        )
        self.fold_infos.append(fold_info)

        if fold == self.k_fold:
            self.oof_metrics = OOFMetrics.from_fold_information_list(
                self.fold_infos, self.oof_predictions
            )

        if report:
            print(
                f'----({self.model_name}) Fold {fold}/{self.k_fold} finished for {self.target_col}----'
            )
            print(' Evaluate on OOF (out of fold):')
            print(metrics)
            print(' Fold information:')
            print(fold_info)

    def report(self):
        print(f'====Training completed - Model: {self.model_name}, Target: {self.target_col}====')
        print(self.oof_metrics)
        print()

    def save_result(self, path):
        # TODO:保存训练结果
        with open(path, 'w') as f:
            print(self.oof_metrics, file=f)


class ModelComparator:
    pass


class SeedSelector:
    pass
