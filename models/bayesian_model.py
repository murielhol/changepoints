import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from models.base import Model
from dataclasses import dataclass, field
from models.bayesian_helpers import Definition, PosteriorTypes, \
    HazardTypes, BayesianOnlineChangepointProcess1D, ConstantHazard, GaussianHazard, \
    StudentTPosterior


@dataclass()
class BayesianCPDModel(Model):

    posterior_definition: Definition
    hazard_definition: Definition
    delay: int = 10
    process_map: Dict[str, BayesianOnlineChangepointProcess1D] = field(init=False)

    def __post_init__(self):
        Model.__post_init__(self)
        assert self.delay >= 1
        self.process_map = {}

    @property
    def model_name(self) -> str:
        return f"{self.posterior_definition.name}_{self.hazard_definition.name}_{self.delay}_{self.test_fraction}"

    def train(self, train_x: pd.DataFrame, train_y: pd.DataFrame,
              val_x: pd.DataFrame, val_y: pd.DataFrame):

        self.initialize_params(train_x)
        cp_train = self.get_change_points(train_x)
        cp_validation = self.get_change_points(val_x)
        metrics_train = self.get_performance_metrics(true_changepoints=train_y["changepoints"].to_list(),
                                                     model_changepoints=cp_train)
        metrics_val = self.get_performance_metrics(true_changepoints=val_y["changepoints"].to_list(),
                                                   model_changepoints=cp_validation)
        print(f'train metrics \n {metrics_train} \n validation metrics \n {metrics_val}')
        self.save_model(self.model_name)
        self.plot_train_results( cp_train, cp_validation, train_x, train_y, val_x, val_y)
        return metrics_train, metrics_val

    def evaluate(self, test_x: pd.DataFrame, test_y: pd.DataFrame):
        self.load_model(self.model_name)
        cp_test = self.get_change_points(test_x)
        metrics_test = self.get_performance_metrics(true_changepoints=test_y["changepoints"].to_list(),
                                                    model_changepoints=cp_test)
        print(f'test metrics: \n {metrics_test}')

    def initialize_params(self, data: pd.DataFrame):

        use_first_n_samples_to_estimate = 20

        for col in data.columns:
            hazard = self.get_hazard_from_definition(self.hazard_definition)
            posterior = self.get_posterior_from_definition(self.posterior_definition)
            posterior.estimate_parameters(data.loc[:use_first_n_samples_to_estimate][col].values)
            process = BayesianOnlineChangepointProcess1D(hazard, posterior)
            self.process_map[col] = process

    def get_change_points(self, data: pd.DataFrame) -> List[int]:

        changepoint_indexes = []
        threshold = 0.20

        for col in data.columns:
            process = self.process_map[col]
            for idx, datum in data[col].items():

                process.update(datum)
                changepoint_detected = len(process.growth_probs) > self.delay and \
                                           process.growth_probs[self.delay] >= threshold

                if changepoint_detected and idx > self.delay:
                    changepoint_indexes.append(idx - self.delay + 1)
                    process.prune(500)
            self.process_map[col] = process

        return list(set(changepoint_indexes))

    def save_model(self, model_name: str):
        model_save_path = Path("outputs") / Path(model_name)
        for col, bayesian_online_cp_process_1d in self.process_map.items():
            column_path = model_save_path / Path(col)
            if not column_path.is_dir():
                column_path.mkdir(parents=True)
            with open(column_path / Path('posterior_params.json'), "w") as outfile:
                json.dump(self.posterior_definition.to_json(), outfile)
            with open(column_path / Path('hazard_params.json'), "w") as outfile:
                json.dump(self.hazard_definition.to_json(), outfile)

    def load_model(self, model_name: str):
        model_save_path = Path("outputs") / Path(model_name)
        model_save_path.glob()
        with open(model_save_path / Path(col) / Path('posterior_params.json'), "r") as infile:
            posterior_definition: Definition = Definition(**json.load(infile))
            posterior = self.get_posterior_from_definition(posterior_definition)
        with open(model_save_path / Path(col) / Path('hazard_params.json'), "r") as infile:
            hazard_definition: Definition = Definition(**json.load(infile))
            hazard = self.get_hazard_from_definition(hazard_definition)
        self.process_map[col] = \
            BayesianOnlineChangepointProcess1D(hazard=hazard, posterior=posterior)

    @staticmethod
    def get_posterior_from_definition(definition: Definition):
        if definition.name == PosteriorTypes.student_t:
            return StudentTPosterior(definition)
        else:
            raise NotImplementedError(f"Unknown posterior {definition.name}")

    @staticmethod
    def get_hazard_from_definition(definition: Definition):
        if definition.name == HazardTypes.constant:
            return ConstantHazard(definition)
        elif definition.name == HazardTypes.gaussian:
            return GaussianHazard(definition)
        else:
            raise NotImplementedError(f"Unknown hazard {definition.name}")
