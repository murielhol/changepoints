from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Iterator, Tuple
import json

from load_dataset import TimeSeries


@dataclass()
class ChangePointDataset:
    changepoints: pd.DataFrame
    inputs: pd.DataFrame
    name: str
    num_features: int = field(init=False)

    def __post_init__(self):
        self.num_features = len(self.inputs.columns)

    def split_train_val_test(self, test_fraction: float
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                        pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        train_split = int((1 - test_fraction * 2) * len(self.inputs))
        val_split = int((1 - test_fraction) * len(self.inputs))
        train_x = self.inputs.iloc[:train_split]
        val_x = self.inputs.iloc[train_split:val_split]
        test_x = self.inputs.iloc[val_split:]

        train_ts_max = train_x.index.max()
        val_ts_max = val_x.index.max()

        train_y = self.changepoints[self.changepoints["changepoints"] <= train_ts_max]
        val_y = self.changepoints[self.changepoints["changepoints"].between(train_ts_max, val_ts_max, inclusive="right")]
        test_y = self.changepoints[self.changepoints["changepoints"] > val_ts_max]
        return train_x, train_y, val_x, val_y, test_x, test_y


@dataclass()
class ChangePointDatasetGenerator:

    generator: Iterator[ChangePointDataset] = field(init=False)

    def __post_init__(self):

        def _generator():
            annotation_df_dict = self._get_annotations()
            for dataset_name in annotation_df_dict.keys():
                df = self._read_in_dataframe(dataset_name)
                yield ChangePointDataset(inputs=df,
                                         changepoints=annotation_df_dict[dataset_name],
                                         name=dataset_name)
        self.generator = _generator()

    def __next__(self):
        return next(self.generator)

    def _read_in_dataframe(self, dataset_name: str) -> pd.DataFrame:
        # return pd.DataFrame({dataset_name: [1]})
        ts = TimeSeries.from_json(f'datasets/{dataset_name}/{dataset_name}.json')
        assert (ts.df.index == ts.t).all()
        return ts.df

    def _get_annotations(self) -> Dict[str, pd.DataFrame]:
        # return  {'a': pd.DataFrame({"changepoints": [1]}),
        #                'b': pd.DataFrame({"changepoints": [2]}),
        #                'c': pd.DataFrame({"changepoints": [3]})
        #                }
        '''
        raw_annotations format is :

        {"<dataset>": {
              "annotator_id": [
                  <change point index>
                  ]}}

        and this class makes this:

        { <dataset> : dataframe }
        with dataframe holing the changepoint and the number of annotators
        '''

        raw_annotations = json.load(open('annotations.json', 'r'))
        annotation_df_dict = {}
        for dataset_name, dataset in raw_annotations.items():
            changepoints = [cp for _, cplist in dataset.items() for cp in cplist]
            counter = Counter(changepoints)

            annotation_df_dict[dataset_name] = pd.DataFrame(counter.items(),
                                                            columns=['changepoints', 'votes'])
        return annotation_df_dict



