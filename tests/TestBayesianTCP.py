import unittest
from unittest import mock
import numpy as np
import pandas as pd

from utils.dataset import ChangePointDataset
from models.bayesian_model import BayesianCPDModel, PosteriorTypes, HazardTypes, Definition, StudentTPosterior


class TestBayesianTCP(unittest.TestCase):

    def test_definition(self):
        definition = Definition(name='hello', params={'mean': 100, 'variance': 200})
        assert definition.name == 'hello'
        assert definition.mean == 100
        assert definition.variance == 200

    def test_student_t_posterior(self):
        posterior = StudentTPosterior()
        fake_data = np.array([1, 2, 3, 4, 5])
        posterior.estimate_parameters(fake_data)
        assert posterior.definition.degrees_freedom == 4

    @mock.patch.object(BayesianCPDModel, 'save_model')
    def test_bayesian_changepoint_detection(self, mock_bcpd):

        bcpd = BayesianCPDModel(
            posterior_definition=Definition(name=PosteriorTypes.student_t,
                                            params={'mean': np.array([50]),
                                                    'degrees_freedom': np.array([1]),
                                                    'var': np.array([1])}),
            hazard_definition=Definition(name=HazardTypes.constant,
                                            params={'lambda_': 100}),
            delay=10
        )

        normal_signal = np.random.normal(loc=50, scale=1, size=1000)
        normal_signal[20:80] += 30
        normal_signal[250:500] += 30
        normal_signal[500:750] -= 30
        y = [20, 80, 250, 500, 750]
        dataset = ChangePointDataset(inputs=pd.DataFrame({'a': normal_signal}),
                                     changepoints=pd.DataFrame({"changepoints": y}),
                                     name="test")
        train_x, train_y, val_x, val_y, _, _ = dataset.split_train_val_test(test_fraction=0.2)
        metrics_train, _ = bcpd.train(train_x, train_y, val_x, val_y)
        for k, v in metrics_train.items():
            if 'precision' in k or 'recall' in k:
                with self.subTest(k):
                    self.assertEqual(v, 1)
