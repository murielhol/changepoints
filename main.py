from dataclasses import dataclass
import numpy as np
from utils.dataset import ChangePointDatasetGenerator
from models.bayesian_model import BayesianCPDModel, Definition, PosteriorTypes, HazardTypes
from models.base import Model


@dataclass()
class JobRunner:

    task: str
    test_fraction: float

    def run(self):

        generator = ChangePointDatasetGenerator()
        while True:
            try:
                dataset = next(generator)
                if self.task == "train":
                    train_x, train_y, val_x, val_y, _, _ = dataset.split_train_val_test(self.test_fraction)
                    self.train(train_x, train_y, val_x, val_y)
                elif self.task == "test":
                    _, _, _, _, test_x, test_y = dataset.split_train_val_test(self.test_fraction)
            except StopIteration:
                break

    @property
    def model(self) -> Model:
        return BayesianCPDModel(
            posterior_definition=Definition(name=PosteriorTypes.student_t),
            hazard_definition=Definition(name=HazardTypes.constant,
                                         params={'lambda_': 10}),
            delay=20,
        )

    def train(self, train_x, train_y, val_x, val_y):
        self.model.train(train_x, train_y, val_x, val_y)


if __name__ == "__main__":
    jr = JobRunner(task="train", test_fraction=0.2)
    jr.run()
