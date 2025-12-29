import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
from loguru import logger
from pathlib import Path


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
        # BUG:如果在TargetTransformer中使用训练的offset，可能在此处出现NaN的问题
        # 目前暂时强制将offset设置为1，来避免NaN问题
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
        model_type: str,
        k_fold: int = 5,
    ):
        self.k_fold = k_fold
        self.target_col = target_col
        self.model_name = model_name
        self.model_type = model_type
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
            self.oof_predictions = pd.concat([self.oof_predictions, df])

        metrics = Metrics.from_predictions(y_true, y_pred, offset)
        fold_info = FoldInformation(
            fold, best_param, metrics, best_inner_score, self.target_col, offset
        )
        self.fold_infos.append(fold_info)

        if fold == self.k_fold:
            self.oof_metrics = OOFMetrics.from_fold_information_list(
                self.fold_infos, self.oof_predictions
            )

        logger.info(
            f'----({self.model_name}) Fold {fold}/{self.k_fold} finished for {self.target_col}----'
        )
        logger.trace(f' Evaluate on OOF (out of fold):\n{metrics}')
        logger.trace(f' Fold information:\n{fold_info}')

    def save_result(self, path: Path):
        logger.info(
            f'====Training completed - Model: {self.model_name}, Target: {self.target_col}===='
        )
        logger.info(f'  OOF metrics:\n{self.oof_metrics}')

        # self.oof_predictions: 全部测试集上的预测结果
        # self.fold_infos: 外CV每一折在测试集上的信息，包括最优参数、指标等信息
        # self.oof_metrics: 在全部测试集上的指标
        self.oof_predictions.to_csv(path / f'oof_predictions_of_{self.model_type}.csv', index=True)
        logger.info(
            f'   The prediction results of OOF have been saved in table oof_predictions_of_{self.model_type}.csv'
        )
        with (path / f'fold_infos_of_{self.model_type}.json').open('w') as f:
            json.dump(
                [asdict(fold_info) for fold_info in self.fold_infos],
                f,
                indent=4,
                ensure_ascii=False,
            )
        logger.info(
            f'   The relevant information for each fold has been saved in fold_infos_of_{self.model_type}.json'
        )
        with (path / f'oof_metrics_of_{self.model_type}.json').open('w') as f:
            oof_metrics = asdict(self.oof_metrics)
            oof_metrics.pop('oof_predictions')
            json.dump(oof_metrics, f, indent=4, ensure_ascii=False)
        logger.info(
            f'   The metrics calculated using the predictions on OOF have been saved in oof_metrics_of_{self.model_type}.json'
        )


class Comparator:
    def __init__(self, cv_threshold=0.5):
        self.cv_threshold = cv_threshold

    def compare_model(self, path: Path):
        # NSE_log 的 coefficient of variance < self.cv_threshold 的情况下，取最大值
        for oof_metrics_path in path.glob('oof_metrics_*.json'):
            pass
        pass

    def compare_seed(self):
        pass
