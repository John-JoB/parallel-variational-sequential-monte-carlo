import pydpf
from pydpf import Module
import torch
from sympy.functions.elementary.benchmarks.bench_exp import timeit_exp_subs
from sympy.physics.units import second
from torch import Tensor
import einops
from warnings import warn
from math import floor
import torch.backends as tb
from math import log


class ParallelKalmanFilter(Module):
    def __init__(self,
                 prior_mean: Tensor|Module,
                 prior_cov: Tensor|Module,
                 dynamic_matrix: Tensor|Module,
                 dynamic_drift: Tensor|Module,
                 dynamic_cov: Tensor|Module,
                 observation_matrix: Tensor|Module,
                 observation_offset: Tensor|Module,
                 observation_cov: Tensor|Module,):
        super().__init__()
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.dynamic_matrix = dynamic_matrix
        self.dynamic_drift = dynamic_drift
        self.dynamic_cov = dynamic_cov
        self.observation_matrix = observation_matrix
        self.observation_offset = observation_offset
        self.observation_cov = observation_cov

    def forward(self, time_extent, observation, **data):
        t = torch.arange(1, time_extent+1)
        data['observation'] = observation
        time_zero_data = {}
        for label, tensor in data.items():
            if label == "series_metadata":
                time_zero_data[label] = tensor
                continue
            time_zero_data[label] = tensor[0]

        positive_time_data = {}
        for label, tensor in data.items():
            if label == "series_metadata":
                positive_time_data[label] = tensor
                continue
            positive_time_data[label] = tensor[1:]

        if isinstance(self.prior_mean, Tensor):
            prior_mean = self.prior_mean.unsqueeze(0)
        else:
            prior_mean = self.prior_mean(**time_zero_data)
        if isinstance(self.prior_cov, Tensor):
            prior_cov = self.prior_cov.unsqueeze(0)
        else:
            prior_cov = self.prior_cov(**time_zero_data)
        if isinstance(self.dynamic_matrix, Tensor):
            dynamic_matrix = pydpf.multiple_unsqueeze(self.dynamic_matrix, 2, 0)
        else:
            dynamic_matrix = self.dynamic_matrix(**positive_time_data).transpose(0,1).contiguous()
        if isinstance(self.dynamic_drift, Tensor):
            dynamic_drift = pydpf.multiple_unsqueeze(self.dynamic_drift, 2, 0)
        else:
            dynamic_drift = self.dynamic_drift(**positive_time_data).transpose(0,1).contiguous()
        if isinstance(self.dynamic_cov, Tensor):
            dynamic_cov = pydpf.multiple_unsqueeze(self.dynamic_cov, 2, 0)
        else:
            dynamic_cov = self.dynamic_cov(**positive_time_data).transpose(0,1).contiguous()
        if isinstance(self.observation_matrix, Tensor):
            observation_matrix = pydpf.multiple_unsqueeze(self.observation_matrix, 2, 0)
        else:
            observation_matrix = self.observation_matrix(**positive_time_data).transpose(0,1).contiguous()
        if isinstance(self.observation_offset, Tensor):
            observation_offset = pydpf.multiple_unsqueeze(self.observation_offset, 2, 0)
        else:
            observation_offset = self.observation_offset(**positive_time_data).transpose(0,1).contiguous()
        if isinstance(self.observation_cov, Tensor):
            observation_cov = pydpf.multiple_unsqueeze(self.observation_cov, 2, 0)
        else:
            observation_cov = self.observation_cov(**positive_time_data).transpose(0,1).contiguous()
        batched_observation = observation.transpose(0,1).contiguous()

        initial_dynamic_matrix = dynamic_matrix[:, 0]
        second_observation_matrix = observation_matrix[:, 1]
        initial_predictive_mean = torch.bmm(initial_dynamic_matrix, prior_mean.unsqueeze(-1)).squeeze() + dynamic_drift[:, 0]
        initial_predictive_cov = torch.einsum("b i j, b j k, b l k -> b i l", initial_dynamic_matrix, prior_cov, initial_dynamic_matrix) + dynamic_cov[:, 0]
        initial_obs_predictive_cov = torch.einsum("b i j, b j k, b l k -> b i l", second_observation_matrix, initial_predictive_cov, second_observation_matrix) + observation_cov[:, 0]
        cholesky_initial_obs_predictive_cov = torch.cholesky(initial_obs_predictive_cov)
        initial_kalman_gain = torch.cholesky_solve(cholesky_initial_obs_predictive_cov, torch.einsum("b i j, b k j -> b i k", initial_predictive_cov, second_observation_matrix))
        initial_post_predictive_mean = (initial_predictive_mean
                                        + torch.bmm(initial_kalman_gain,
                                                    (batched_observation[:, 1, :, None] - torch.bmm(second_observation_matrix, initial_predictive_mean.unsqueeze(-1)) - observation_offset[:, 1, :, None])
                                                    ).squeeze())
        initial_post_predictive_cov = initial_predictive_cov - torch.einsum("b i j, b j k, b l k -> b i l", initial_kalman_gain, initial_predictive_cov, second_observation_matrix)

        remaining_dynamic_matrix = dynamic_matrix[:, 1:]
        remaining_observation_matrix = observation_matrix[:, 2:]
        post_predictive_cov = torch.einsum("b i j, b j k, b l k -> b i l", remaining_observation_matrix, , remaining_observation_matrix)
        kalman_gain =

class ParallelKalmanSmoother(Module):
    def __init__(self,
                 prior_mean: Tensor|Module,
                 prior_cov: Tensor|Module,
                 dynamic_matrix: Tensor|Module,
                 dynamic_drift: Tensor|Module,
                 dynamic_cov: Tensor|Module,
                 observation_matrix: Tensor|Module,
                 observation_offset: Tensor|Module,
                 observation_cov: Tensor|Module,):
        super().__init__()
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.dynamic_matrix = dynamic_matrix
        self.dynamic_drift = dynamic_drift
        self.dynamic_cov = dynamic_cov
        self.observation_matrix = observation_matrix
        self.observation_offset = observation_offset
        self.observation_cov = observation_cov


    def forward(self):
