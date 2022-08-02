from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
import pandas as pd


@dataclass
class Model(ABC):

    test_fraction = 0.2

    def __post_init__(self):
        model_path = Path(f'outputs')
        if not model_path.exists():
            model_path.mkdir()
        model_path = Path(f'outputs/{self.model_name}')
        if not model_path.exists():
            model_path.mkdir()

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def train(self, train_x: pd.DataFrame, train_y: pd.DataFrame,
              val_x: pd.DataFrame, val_y: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_x: pd.DataFrame, test_y: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, model_save_path):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_save_path):
        raise NotImplementedError

    @staticmethod
    def get_performance_metrics(true_changepoints: List[int], model_changepoints: List[int]) -> Optional[dict]:

        if not (model_changepoints and true_changepoints):
            return None

        # on average how many days ahead can de model predict the changepoint
        result = {'nearest_changepoint': 0}
        # also get the f1 score for various distance thresholds
        tolerances = sorted([2, 5, 10, 20])
        for tl in tolerances:
            result[f'tp_{tl}'] = 0
            result[f'fp_{tl}'] = 0

        for tcp in true_changepoints:
            distance_to_nearest_model_cp = min([abs(tcp - mcp) for mcp in model_changepoints])
            result['nearest_changepoint'] += distance_to_nearest_model_cp
            for tl in tolerances:
                if distance_to_nearest_model_cp <= tl:
                    result[f'tp_{tl}'] += 1

        for mcp in model_changepoints:
            distance_to_nearest_true_cp = min([abs(tcp - mcp) for tcp in true_changepoints])
            for tl in tolerances:
                if distance_to_nearest_true_cp > tl:
                    result[f'fp_{tl}'] += 1

        for tl in tolerances:
            denominator = (result[f'tp_{tl}'] + result[f'fp_{tl}'])
            result[f'precision_{tl}'] = result[f'tp_{tl}'] / denominator if denominator > 0 else 0
            result[f'recall_{tl}'] = result[f'tp_{tl}'] / len(model_changepoints)
            denominator = (result[f'precision_{tl}'] + result[f'recall_{tl}'])
            result[f'f1_{tl}'] = 2 * (result[f'precision_{tl}'] * result[f'recall_{tl}']
                                      ) / denominator if denominator > 0 else 0

        result['nearest_changepoint'] /= len(true_changepoints)
        return result
