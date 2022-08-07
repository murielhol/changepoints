import json
from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import scipy.stats
from scipy.stats import stats

from dataclasses import dataclass, asdict




@dataclass()
class Definition:

    def __init__(self, name: str, params: Optional[dict] = None):
        if params is None:
            params = {}
        self.name = name
        self.__dict__.update(params)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)


class PosteriorTypes:
    student_t = 'student_t'


class HazardTypes:
    constant = 'constant'
    gaussian = 'gaussian'


'''

all credits to
-> -> ->    https://github.com/projectaligned/chchanges/blob/master/chchanges/bayesian_online.py


'''


@dataclass()
class Posterior(ABC):
    """
    Abstract class defining the interface for the Posterior distribution.
    See Equation 2 of https://arxiv.org/abs/0710.3742
    In the Bayesian Online Changepoint Detection algorithm, the Posterior
        P(x_t | r_{t-1}, x_{t-1}^(r))
    specifies the probability of sampling the next detected data point from the distribution
    associated with the current regime.
    """
    definition: Optional[Definition] = None

    @abstractmethod
    def estimate_parameters(self, data: np.array):
        """
        :param data: dataset to estimate parameters from
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, data: Union[float, np.ndarray]) -> np.ndarray:
        """
        Probability density function for the distribution at data.
        If the distribution is d-dimensional, then the data array should have length d.
        If the pruned parameter history has length l, then the returned array should have length l.
        :param data: the data point for which we want to know the probability of sampling from
            the posterior distribution
        :return: the probability of sampling the datapoint from each distribution in the
            pruned parameter history.
        """
        raise NotImplementedError

    @abstractmethod
    def update_theta(self, data: Union[float, np.ndarray]) -> None:
        """
        Use new data to update the posterior distribution.
        The vector of parameters which define the distribution is called theta, hence the name.
        Note that it is important to filter faulty data and outliers before updating theta in
        order to maintain the stability of the distribution.
        :param data: the datapoint which we want to use to update the distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def prune(self, t: int) -> None:
        """
        Remove the parameter history before index t in order to save memory.
        :param t: the index to prune at, e.g. the index of a changepoint.
        """
        raise NotImplementedError


@dataclass()
class Hazard(ABC):
    """
    Abstract class defining the interface for the Hazard function.
    See Equation 5 of https://arxiv.org/abs/0710.3742
    The hazard provides information on how the occurrence of previous
        changepoints affects the probability of subsequent changepoints.
    """

    definition: Definition

    @abstractmethod
    def __call__(self, gap: int) -> float:
        """
        Compute the hazard for a gap between changepoints of a given size.
        :param gap: the number of datapoints since the last changepoint
        :return: the value of the hazard function.
        """
        raise NotImplementedError


class BayesianOnlineChangepointProcess1D:

    def __init__(self, hazard: Hazard, posterior: Posterior):
        """
        Performs Bayesian Online Changepoint Detection as defined in https://arxiv.org/abs/0710.3742
        :param hazard: The hazard provides information on how the occurrence of previous
            changepoints affects the probability of subsequent changepoints.
        :param posterior: The posterior determines the probability of observing a certain data point
            given the data points observed so far.
        """
        self.hazard = hazard
        self.posterior = posterior

        # The start index marks the beginning of the current run,
        # and then end index marks the end of the current run,
        # where a run consists of all the data points between two changepoints.
        self.start = 0
        self.end = 0

        # The (len(growth_probs) - i)-th value in the growth probabilities array records the
        # probability that there is a changepoint between the ith datum and the i+1th datum.
        # See Step 1. of https://arxiv.org/abs/0710.3742 Algorithm 1
        self.growth_probs = np.array([1.])

    def update(self, datum: np.ndarray):
        """
        Update the run probabilities based on the new data point
        :param datum: the new data point
        """
        # Observe New Datum:
        # See Step 2. of https://arxiv.org/abs/0710.3742 Algorithm 1

        # run indicates the number of data points since the last changepoint
        run = self.end - self.start
        self.end += 1

        # Allocate enough space, and reduce number of resizings.
        if len(self.growth_probs) == run + 1:
            self.growth_probs = np.hstack([self.growth_probs, np.zeros_like(self.growth_probs)])
            # self.growth_probs = np.resize(self.growth_probbs, (run + 1) * 2)

        # Evaluate Predictive Probability:
        # See Step 3. of https://arxiv.org/abs/0710.3742 Algorithm 1
        # I.e. Determine the probability of observing the datum,
        # for each of the past posterior parameter sets.
        pred_probs = self.posterior.pdf(datum)

        # Evaluate the hazard function for this run length
        hazard_value = self.hazard(run + 1)

        # Calculate Changepoint Probability:
        # See Step 5. of https://arxiv.org/abs/0710.3742 Algorithm 1
        # index 0 of growth probabilities corresponds to a run length of zero, i.e. a changepoint
        try:
            cp_prob = np.sum(self.growth_probs[0:run + 1] * pred_probs * hazard_value)
        except:
            pass
        # Calculate Growth Probabilities:
        # See Step 4. of https://arxiv.org/abs/0710.3742 Algorithm 1
        # self.growth_probs[i] corresponds to the probability of a run length of i,
        # hence after the new datum, the probability mass at i moves to i + 1,
        # scaled by (1 - hazard), since hazard is the baseline probability of a changepoint
        # and scaled by the relative likelihood of measuring the given data point for each of the
        # past posterior parameter sets.
        self.growth_probs[1:run + 2] = (self.growth_probs[0:run + 1] * pred_probs
                                        * (1 - hazard_value))

        # Store changepoint probability
        self.growth_probs[0] = cp_prob

        # Calculate Evidence, Determine Run Length Distribution
        # See Steps 6. and 7. of https://arxiv.org/abs/0710.3742 Algorithm 1
        # Intuitively, if a new data point is highly unlikely to fall in the past distribution,
        # then the corresponding predictive probability is very small.
        # Then, if the predictive probability at index i is very small, then growth
        # probabilities after index i will be very small.
        # And so, after normalizing, growth probabilities before index i will be much larger, and
        # so the first couple of points after index i should then exceed the threshold,
        # until the distribution parameters reflect the new data-generating distribution.
        self.growth_probs[0:run + 2] /= np.sum(self.growth_probs[0:run + 2])

        # Update Sufficient Statistics:
        # See Step 8. of https://arxiv.org/abs/0710.3742 Algorithm 1
        # Update the parameters for each possible run length.
        self.posterior.update_theta(datum)

    def prune(self, t) -> None:
        """
        Remove history older than t indices in the past in order to save memory.
        """
        if t < self.end:
            self.posterior.prune(t)
            self.growth_probs = self.growth_probs[:t + 1]
            self.start = self.end - t


@dataclass
class ConstantHazard(Hazard):
    """
    See Equation 5 of https://arxiv.org/abs/0710.3742
    "In the special case is where Pgap(g) is a discrete exponential (geometric) distribution with
    timescale λ, the process is memoryless and the hazard function is constant at H(τ) = 1/λ."
    """

    def __call__(self, gap: int) -> np.ndarray:
        """
        Evaluate the hazard function
        :param gap: the number of indices since the last event.
        :return: simply a constant array of length gap.
        """
        return np.full(gap, 1. / self.definition.lambda_)


@dataclass
class GaussianHazard(Hazard):

    def __call__(self, gap: int) -> np.ndarray:
        r = np.array(gap)
        # Hazard function = pdf/survival = pdf/(1-cdf)
        pdf = stats.norm.pdf(r, self.definition.mu, self.definition.sigma)
        survival = 1 - stats.norm.cdf(r, self.definition.mu, self.definition.sigma)
        hazard_arr = pdf / survival
        return hazard_arr


@dataclass()
class StudentTPosterior(Posterior):
    """
    The Student's T distribution is the predictive posterior to the normal distribution in the case where
    both the variance and mean are unknown:
    see https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf section 3
    and https://en.wikipedia.org/wiki/Posterior_predictive_distribution
    Student's T predictive posterior.
    https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous_t.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
    """

    def estimate_parameters(self, data: np.array):
        _, loc, scale = scipy.stats.t.fit(data)
        self.set_parameters(degrees_freedom=np.array([1]),
                            mean=np.array([loc]),
                            var=np.array([scale**2]),
                            )

    def set_parameters(self, degrees_freedom: np.array, mean: np.array, var: np.array):
        self.definition = Definition(name=PosteriorTypes.student_t,
                                      params={'var': var,
                                              'degrees_freedom': degrees_freedom,
                                              'mean': mean})

    def pdf(self, data: float) -> np.ndarray:
        """
        The probability density function for the Student's T of the predictive posterior.
        Note that t.pdf(x, df, loc, scale) is identically equivalent to t.pdf(y, df) / scale
        with y = (x - loc) / scale. So increased self.var corresponds to increased scale
        which in turn corresponds to a flatter distribution.
        :param data: the data point for which we want to know the probability of sampling from
            the posterior distribution.
        :return: the probability of sampling the datapoint from each distribution in the
            pruned parameter history.
        """
        return scipy.stats.t.pdf(x=data, df=self.definition.degrees_freedom, loc=self.definition.mean,
                                 scale=np.sqrt(2. * self.definition.var * (self.definition.degrees_freedom + 1) / self.definition.degrees_freedom ** 2))

    def update_theta(self, data: float) -> None:
        """
        Use new data to update the predictive posterior distribution.
        The vector of parameters which define the distribution is called theta, hence the name.
        Note that it is important to filter faulty data and outliers before updating theta in
        order to maintain the stability of the distribution.
        This update rule is based on page 9 of https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        specifically equations 86, 87, 88, and 89 for the case where we are updating with one data point at a time
        so that m=1.

        This fact that m=1 also allows us to eliminate the alpha and kappa variables in favor of a combined df variable,
        further note that I have renamed beta -> var and mu -> mean in this implementation.
        :param data: the datapoint which we want to use to update the distribution.
        """
        next_var = 0.5 * (data - self.definition.mean) ** 2 * self.definition.degrees_freedom / (
                self.definition.degrees_freedom + 1.)
        self.definition.var = np.concatenate(([self.definition.var[0]], self.definition.var + next_var))
        self.definition.mean = np.concatenate(([self.definition.mean[0]],
                                               (self.definition.degrees_freedom * self.definition.mean + data
                                                ) / (self.definition.degrees_freedom + 1)))
        self.definition.degrees_freedom = np.concatenate(([self.definition.degrees_freedom[0]],
                                                          self.definition.degrees_freedom + 1.))

    def prune(self, t: int) -> None:
        """
        Remove the parameter history before index t.
        :param t: the index to prune at, e.g. the index of a changepoint.
        """
        self.definition.mean = self.definition.mean[:t + 1]
        self.definition.degrees_freedom = self.definition.degrees_freedom[:t + 1]
        self.definition.var = self.definition.var[:t + 1]
