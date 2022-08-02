import unittest

import pandas as pd
from dataclasses import dataclass
from models.base import Model

@dataclass()
class SomeModel(Model):

    model_name = "test"

    def evaluate(self, test_data, model_save_path):
        pass

    def train(self, train_x: pd.DataFrame, train_y: pd.DataFrame,
              val_x: pd.DataFrame, val_y: pd.DataFrame):
        pass

    def load_model(self, model_name):
        pass

    def save_model(self, model_name):
        pass


class TestModel(unittest.TestCase):

    def test_get_performance_metrics(self):
        model = SomeModel()
        model_cp = [1, 10, 20]
        true_cp = [1, 10, 20]
        result = model.get_performance_metrics(true_cp, model_cp)
        # thresholds are 2, 5, 10, 20
        self.assertEqual(result['tp_2'], 3)
        self.assertEqual(result['tp_5'], 3)
        self.assertEqual(result['tp_10'], 3)
        self.assertEqual(result['tp_20'], 3)
        self.assertEqual(result['fp_2'], 0)
        self.assertEqual(result['fp_5'], 0)
        self.assertEqual(result['fp_10'], 0)
        self.assertEqual(result['fp_20'], 0)
        self.assertEqual(result['f1_2'], 1)
        self.assertEqual(result['f1_5'], 1)
        self.assertEqual(result['f1_10'], 1)
        self.assertEqual(result['f1_20'], 1)
        self.assertEqual(result['precision_2'], 1)
        self.assertEqual(result['precision_5'], 1)
        self.assertEqual(result['precision_10'], 1)
        self.assertEqual(result['precision_20'], 1)
        self.assertEqual(result['recall_2'], 1)
        self.assertEqual(result['recall_5'], 1)
        self.assertEqual(result['recall_10'], 1)
        self.assertEqual(result['recall_20'], 1)
        self.assertEqual(result['nearest_changepoint'], 0)

        model_cp = []
        true_cp = [1, 10, 20]
        result = model.get_performance_metrics(true_cp, model_cp)
        assert result is None

        model_cp = [2, 14]
        true_cp = [1, 10, 20]
        result = model.get_performance_metrics(true_cp, model_cp)
        self.assertEqual(result['tp_2'], 1)
        self.assertEqual(result['tp_5'], 2)
        self.assertEqual(result['tp_10'], 3)
        self.assertEqual(result['tp_20'], 3)
        self.assertEqual(result['fp_2'], 1)
        self.assertEqual(result['fp_5'], 0)
        self.assertEqual(result['fp_10'], 0)
        self.assertEqual(result['fp_20'], 0)

        for t in [2, 5, 10, 10]:
            with self.subTest(t):
                self.assertEqual(result[f"recall_{t}"], result[f'tp_{t}']/len(model_cp))
                self.assertEqual(result[f"precision_{t}"], result[f'tp_{t}'] / (result[f'tp_{t}'] + result[f'fp_{t}']))
                self.assertEqual(result[f"f1_{t}"],
                                 (2 * (result[f"precision_{t}"] * result[f"recall_{t}"]) /
                                  (result[f"precision_{t}"] + result[f"recall_{t}"])))
        self.assertEqual(result['nearest_changepoint'],  (1+4+6)/3)
