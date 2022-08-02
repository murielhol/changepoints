import pandas as pd
import unittest
from unittest import mock
from utils.dataset import ChangePointDataset, ChangePointDatasetGenerator


class TestChangePointDatasetGenerator(unittest.TestCase):

    def test_generator(self):

        annotations = {'a': pd.DataFrame({"changepoints": [1]}),
                       'b': pd.DataFrame({"changepoints": [2]}),
                       'c': pd.DataFrame({"changepoints": [3]})
                       }

        datasets = [pd.DataFrame({'a': [1]}),
                    pd.DataFrame({'b': [1]}),
                    pd.DataFrame({'c': [1]})]

        with mock.patch.object(ChangePointDatasetGenerator, "_read_in_dataframe", side_effect=datasets):
            with mock.patch.object(ChangePointDatasetGenerator, "_get_annotations", return_value=annotations):
                cpg = ChangePointDatasetGenerator()
                a = next(cpg)
                b = next(cpg)
                c = next(cpg)
                assert a.inputs.equals(datasets[0])
                assert b.inputs.equals(datasets[1])
                assert c.inputs.equals(datasets[2])
                assert a.changepoints.equals(annotations['a'])
                assert b.changepoints.equals(annotations['b'])
                assert c.changepoints.equals(annotations['c'])
                with self.assertRaises(StopIteration):
                    next(cpg)
                assert isinstance(a, ChangePointDataset)
