"""
Contains code for metric evaluation and model comparison.
"""

import re
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
from loguru import logger
from pathlib import Path


def get_threshold(target) -> float:
    match target:
        case 'THIA':
            return 0.0206060966888198
        case 'IMI':
            return 0.0404050865952664
        case 'CLO':
            return 0.0305055916420431
        case 'ACE':
            return 0.00434264069423857
        case 'DIN':
            return 0.0467690476366243
        case 'parentNNIs':
            return 0.253244228089568
        case 'IMI-UREA':
            return 0.0283842712949238
        case 'DN-IMI':
            return 0.0326269119891624
        case 'DM-ACE':
            return 0.0425264069423857
        case 'CLO-UREA':
            return 0.0609111832840862
        case 'mNNIs':
            return 0.405979293082156
        case _:
            raise ValueError(f'Unknown target {target}')


@dataclass
class Metrics:
    """
    The most basic set of evaluation metrics, along with the calculation methods for these metrics.
    """

    NSE_log: float
    NSE_log_detected: float
    RSR_log: float
    RSR_log_detected: float
    NSE: float
    NSE_detected: float
    RSR: float
    RSR_detected: float
    PBIAS: float
    PBIAS_detected: float
    KGE: float
    KGE_detected: float
    FNR: float  # false negative rate
    FPR: float  # false positive rate
    TNR: float  # true negative rate
    TPR: float  # true positive rate

    @classmethod
    def from_predictions(
        cls, y_true: np.ndarray, y_pred: np.ndarray, offset: float, threshold: float
    ):
        conf = cls.calc_confusion_matrix(y_true, y_pred, threshold)
        mask = conf['true']
        y_true_log = np.log(y_true + offset)
        y_pred_log = np.log(y_pred + offset)
        assert not np.isnan(y_true_log).any(), f'offset={offset}\ny_true={y_true}'
        assert not np.isnan(y_pred_log).any(), f'offset={offset}\ny_pred={y_pred}'
        return cls(
            NSE_log=cls.calc_nse(y_true_log, y_pred_log),
            NSE_log_detected=cls.calc_nse(y_true_log[mask], y_pred_log[mask]),
            RSR_log=cls.calc_rsr(y_true_log, y_pred_log),
            RSR_log_detected=cls.calc_rsr(y_true_log[mask], y_pred_log[mask]),
            NSE=cls.calc_nse(y_true, y_pred),
            NSE_detected=cls.calc_nse(y_true[mask], y_pred[mask]),
            RSR=cls.calc_rsr(y_true, y_pred),
            RSR_detected=cls.calc_rsr(y_true[mask], y_pred[mask]),
            PBIAS=cls.calc_pbias(y_true, y_pred),
            PBIAS_detected=cls.calc_pbias(y_true[mask], y_pred[mask]),
            KGE=cls.calc_kge(y_true, y_pred),
            KGE_detected=cls.calc_kge(y_true[mask], y_pred[mask]),
            FNR=conf['FN'] / y_pred.size,
            FPR=conf['FP'] / y_pred.size,
            TNR=conf['TN'] / y_pred.size,
            TPR=conf['TP'] / y_pred.size,
        )

    def to_format_dict(self):
        return {
            'NSE (log)': f'{self.NSE_log:.4f}',
            'NSE (log, detected)': f'{self.NSE_log_detected:.4f}',
            'RSR (log)': f'{self.RSR_log:.4f}',
            'RSR (log, detected)': f'{self.RSR_log_detected:.4f}',
            'NSE': f'{self.NSE:.4f}',
            'NSE (detected)': f'{self.NSE_detected:.4f}',
            'RSR': f'{self.RSR:.4f}',
            'RSR (detected)': f'{self.RSR_detected:.4f}',
            'PBIAS (%)': f'{self.PBIAS:.4f}',
            'PBIAS (%, detected)': f'{self.PBIAS_detected:.4f}',
            'KGE': f'{self.KGE:.4f}',
            'KGE (detected)': f'{self.KGE_detected:.4f}',
            'FNR': f'{self.FNR:.4f}',
            'FPR': f'{self.FPR:.4f}',
            'TNR': f'{self.TNR:.4f}',
            'TPR': f'{self.TPR:.4f}',
        }

    @staticmethod
    def get_metrics_repr(ind):
        match ind:
            case 'NSE_log' | 'NSE_log_detected':
                return 'NSE (log)'
            case 'RSR_log' | 'RSR_log_detected':
                return 'RSR (log)'
            case 'NSE' | 'NSE_detected':
                return 'NSE'
            case 'RSR' | 'RSR_detected':
                return 'RSR'
            case 'PBIAS' | 'PBIAS_detected':
                return 'PBIAS (%)'
            case 'KGE' | 'KGE_detected':
                return 'KGE'
            case 'FNR' | 'FPR' | 'TNR' | 'TPR':
                return ind

    @staticmethod
    def calc_confusion_matrix(y_true, y_pred, threshold):
        actual_pos = y_true > threshold
        actual_neg = y_true <= threshold
        predict_pos = y_pred > threshold
        predict_neg = y_pred <= threshold

        return {
            'TP': np.sum(actual_pos & predict_pos),
            'TN': np.sum(actual_neg & predict_neg),
            'FP': np.sum(actual_neg & predict_pos),
            'FN': np.sum(actual_pos & predict_neg),
            'true': predict_pos & actual_pos,
        }

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
    """
    This class records the information of one fold in the outer cross-validation, including
     the optimal hyperparameters identified by the inner cross-validation, evaluation metrics
     on the test set, and other related information.
    """

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
    """
    Out-of-fold evaluation metrics include outer cross-validation, the average and standard
     deviation of metrics on the test set for each fold, and retain all prediction results
     on the test set to calculate out-of-fold metrics, while also saving relevant information
     for each fold.
    """

    mean: Metrics
    std: Metrics
    oof: Metrics
    oof_predictions: pd.DataFrame
    fold_infos: list[FoldInformation]

    @classmethod
    def from_fold_information_list(
        cls, fold_infos: list[FoldInformation], oof_predictions: pd.DataFrame
    ) -> 'OOFMetrics':
        target_col = fold_infos[0].target_col
        mean = Metrics(
            NSE_log=np.mean([info.metrics.NSE_log for info in fold_infos]),
            NSE_log_detected=np.mean([info.metrics.NSE_log_detected for info in fold_infos]),
            RSR_log=np.mean([info.metrics.RSR_log for info in fold_infos]),
            RSR_log_detected=np.mean([info.metrics.RSR_log_detected for info in fold_infos]),
            NSE=np.mean([info.metrics.NSE for info in fold_infos]),
            NSE_detected=np.mean([info.metrics.NSE_detected for info in fold_infos]),
            RSR=np.mean([info.metrics.RSR for info in fold_infos]),
            RSR_detected=np.mean([info.metrics.RSR_detected for info in fold_infos]),
            PBIAS=np.mean([info.metrics.PBIAS for info in fold_infos]),
            PBIAS_detected=np.mean([info.metrics.PBIAS_detected for info in fold_infos]),
            KGE=np.mean([info.metrics.KGE for info in fold_infos]),
            KGE_detected=np.mean([info.metrics.KGE_detected for info in fold_infos]),
            FNR=np.mean([info.metrics.FNR for info in fold_infos]),
            FPR=np.mean([info.metrics.FPR for info in fold_infos]),
            TNR=np.mean([info.metrics.TNR for info in fold_infos]),
            TPR=np.mean([info.metrics.TPR for info in fold_infos]),
        )
        std = Metrics(
            NSE_log=np.std([info.metrics.NSE_log for info in fold_infos]),
            NSE_log_detected=np.std([info.metrics.NSE_log_detected for info in fold_infos]),
            RSR_log=np.std([info.metrics.RSR_log for info in fold_infos]),
            RSR_log_detected=np.std([info.metrics.RSR_log_detected for info in fold_infos]),
            NSE=np.std([info.metrics.NSE for info in fold_infos]),
            NSE_detected=np.std([info.metrics.NSE_detected for info in fold_infos]),
            RSR=np.std([info.metrics.RSR for info in fold_infos]),
            RSR_detected=np.std([info.metrics.RSR_detected for info in fold_infos]),
            PBIAS=np.std([info.metrics.PBIAS for info in fold_infos]),
            PBIAS_detected=np.std([info.metrics.PBIAS_detected for info in fold_infos]),
            KGE=np.std([info.metrics.KGE for info in fold_infos]),
            KGE_detected=np.std([info.metrics.KGE_detected for info in fold_infos]),
            FNR=np.std([info.metrics.FNR for info in fold_infos]),
            FPR=np.std([info.metrics.FPR for info in fold_infos]),
            TNR=np.std([info.metrics.TNR for info in fold_infos]),
            TPR=np.std([info.metrics.TPR for info in fold_infos]),
        )
        mean_offset = np.mean([info.offset for info in fold_infos])
        oof = Metrics.from_predictions(
            oof_predictions[target_col].values,
            oof_predictions[target_col + '_pred'].values,
            mean_offset,
            get_threshold(target_col),
        )
        return cls(
            mean=mean,
            std=std,
            oof=oof,
            oof_predictions=oof_predictions,
            fold_infos=fold_infos,
        )

    @classmethod
    def from_json(cls, json_obj) -> 'OOFMetrics':
        mean = Metrics(**json_obj['mean'])
        std = Metrics(**json_obj['std'])
        oof = Metrics(**json_obj['oof'])
        oof_predictions = pd.DataFrame(json_obj['oof_predictions'])
        oof_predictions = oof_predictions.set_index('ID')
        fold_infos = [FoldInformation(**fold_info) for fold_info in json_obj['fold_infos']]
        return cls(mean, std, oof, oof_predictions, fold_infos)

    def to_json(self) -> dict:
        res = asdict(self)
        res['oof_predictions'] = res['oof_predictions'].reset_index().to_dict(orient='records')
        return res

    def calc_coefficient_of_variation(self, indicator):
        mean = getattr(self.mean, indicator)
        std = getattr(self.std, indicator)
        logger.trace(f'mean={mean}, std={std}')
        return std / mean

    def to_format_dict(self) -> dict:
        mean_dict = self.mean.to_format_dict()
        std_dict = self.std.to_format_dict()
        return {k: f'{mean_dict[k]} ± {std_dict[k]}' for k in mean_dict.keys()}

    @staticmethod
    def format_table(mean: Metrics, std: Metrics, oof: Metrics, target: str):
        mean_dict = asdict(mean)
        std_dict = asdict(std)
        oof_dict = asdict(oof)
        data = [[k, oof_dict[k], mean_dict[k], std_dict[k]] for k in oof_dict.keys()]
        title = f'OOF Metrics for {target}'
        table = tabulate(
            data, headers=['Metric', 'OOF', 'mean', 'std'], tablefmt='fancy_grid', numalign='right'
        )
        return f'\n####{title}####\n{table}'

    def __repr__(self):
        target_col = self.fold_infos[0].target_col
        table_str = self.format_table(self.mean, self.std, self.oof, target_col)
        return table_str


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
            self.oof_predictions['fold'] = fold
            self.oof_predictions[self.target_col] = y_true
            self.oof_predictions[self.target_col + '_pred'] = y_pred
        else:
            df = X.copy()
            df['fold'] = fold
            df[self.target_col] = y_true
            df[self.target_col + '_pred'] = y_pred
            self.oof_predictions = pd.concat([self.oof_predictions, df])

        # NOTE: Build Metrics 1
        metrics = Metrics.from_predictions(y_true, y_pred, offset, get_threshold(self.target_col))
        fold_info = FoldInformation(
            fold, best_param, metrics, best_inner_score, self.target_col, offset
        )
        self.fold_infos.append(fold_info)

        if fold == self.k_fold:
            # NOTE: Build Metrics 2
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
        logger.trace(f'  OOF metrics:{self.oof_metrics}')
        # self.oof_predictions.to_csv(path / f'oof_predictions_of_{self.model_type}.csv', index=True)
        # logger.info(
        #     f'   The prediction results of OOF have been saved in table oof_predictions_of_{self.model_type}.csv'
        # )
        with (path / f'oof_metrics_of_{self.model_type}.json').open('w') as f:
            oof_metrics = self.oof_metrics.to_json()
            json.dump(oof_metrics, f, indent=4, ensure_ascii=False)
        logger.info(
            f'   The metrics calculated using the predictions on OOF have been saved in oof_metrics_of_{self.model_type}.json'
        )


class Comparator:
    """
    Compare the effects of different models and different random seeds.
    """

    def __init__(self, indicator='NSE_log', cv_threshold=0.8):
        self.indicator = indicator
        self.cv_threshold = cv_threshold

    def compare_model(self, path: Path):
        """
        Under the same seed experiment, select the one with the best metrics among different models.
        """
        if (path / 'model_comparison.json').exists():
            (path / 'model_comparison.json').unlink()
        indicator_map = {}
        metrics_map = {}
        cv_records = []
        for oof_metrics_path in path.glob('oof_metrics_of_*.json'):
            mt = re.search(r'^oof_metrics_of_(.+)\.json$', oof_metrics_path.name)
            if mt:
                model_type = mt.group(1)
            else:
                raise RuntimeError(f'model type doesnot found in {oof_metrics_path.name}')
            with oof_metrics_path.open() as f:
                oof_metrics = json.load(f)
            oof_metrics = OOFMetrics.from_json(oof_metrics)
            try:
                cv = oof_metrics.calc_coefficient_of_variation(self.indicator)
                cv_records.append(cv)
                logger.trace(f'Coefficient of variation of {path} is {cv}')
                if cv <= self.cv_threshold and cv >= 0:
                    indicator_map[model_type] = getattr(oof_metrics.mean, self.indicator)
                    metrics_map[model_type] = oof_metrics
            except Exception:
                logger.exception('Exception when calculate coefficient of variation.')
                pass

        if len(indicator_map) == 0:
            logger.trace(f'No valid (cv_threshold={self.cv_threshold}) model. All cv: {cv_records}')
            return

        best_model_type = max(indicator_map, key=indicator_map.get)  # type: ignore
        with (path / 'model_comparison.json').open('w') as f:
            json.dump(
                {
                    'best_model_type': best_model_type,
                    self.indicator: indicator_map[best_model_type],
                    'metrics': metrics_map[best_model_type].to_json(),
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

    def compare_seed(self, path: Path):
        """
        Among different seeds, select the best seed based on the performance of the best model under each seed.
        """
        if (path / 'seed_comparison.json').exists():
            (path / 'seed_comparison.json').unlink()
        seed_indicator_map = {}
        for seed_dir in path.iterdir():
            if not seed_dir.is_dir():
                continue
            mt = re.search(r'^seed_(\d+)', seed_dir.name)
            if mt:
                seed = int(mt.group(1))
            else:
                logger.trace(f'Pass {seed_dir}, seed not found.')
                continue
            model_comparison_path = seed_dir / 'model_comparison.json'
            if not model_comparison_path.exists():
                logger.trace(f'Pass {seed_dir}, model comparison not found.')
                continue
            with model_comparison_path.open() as f:
                model_comparison = json.load(f)
                seed_indicator_map[seed] = model_comparison[self.indicator]

        if len(seed_indicator_map) == 0:
            logger.trace(f'No valid (cv_threshold={self.cv_threshold}) seed.')
            return

        best_seed = max(seed_indicator_map, key=seed_indicator_map.get)  # type: ignore
        with (path / f'seed_{best_seed}/model_comparison.json').open() as f:
            model_comparison = json.load(f)
            best_model_type = model_comparison['best_model_type']
            best_metrics = model_comparison['metrics']

        with (path / 'seed_comparison.json').open('w') as f:
            json.dump(
                {
                    'best_seed': best_seed,
                    'best_model_type': best_model_type,
                    'best_metrics': best_metrics,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

    @staticmethod
    def format_summary_table(df: pd.DataFrame) -> str:
        return tabulate(df.T, headers='keys', tablefmt='fancy_grid', showindex=True)
