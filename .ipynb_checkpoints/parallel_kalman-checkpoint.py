import pydpf
from pydpf import Module
import torch
from torch import Tensor
import einops
from warnings import warn
from math import floor
from math import log
import opt_einsum


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

    @staticmethod
    def unsqueeze_to_dim(a, b):
        if isinstance(b, Tensor):
            b = b.dim
        if a.dim == b:
            return a
        if a.dim > b:
            raise ValueError(f"Dimension of the first tensor must be less than the second tensor found dims {a.dim}, {b}")
        return pydpf.multiple_unsqueeze(a, b - a.dim, 0)

    def initialise_kalman_parameters(self, data):
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
            prior_mean = self.prior_mean
        else:
            prior_mean = self.prior_mean(**time_zero_data)
        prior_mean = ParallelKalmanFilter.unsqueeze_to_dim(prior_mean, 2)

        if isinstance(self.prior_cov, Tensor):
            prior_cov = self.prior_cov.unsqueeze(0)
        else:
            prior_cov = self.prior_cov(**time_zero_data)
        prior_cov = ParallelKalmanFilter.unsqueeze_to_dim(prior_cov, 3)

        if isinstance(self.dynamic_matrix, Tensor):
            dynamic_matrix = self.dynamic_matrix
        else:
            dynamic_matrix = self.dynamic_matrix(**positive_time_data)
        dynamic_matrix = ParallelKalmanFilter.unsqueeze_to_dim(dynamic_matrix, 4)

        if isinstance(self.dynamic_drift, Tensor):
            dynamic_drift = pydpf.multiple_unsqueeze(self.dynamic_drift, 2, 0)
        else:
            dynamic_drift = self.dynamic_drift(**positive_time_data)
        dynamic_drift =  ParallelKalmanFilter.unsqueeze_to_dim(dynamic_drift, 3)

        if isinstance(self.dynamic_cov, Tensor):
            dynamic_cov = pydpf.multiple_unsqueeze(self.dynamic_cov, 2, 0)
        else:
            dynamic_cov = self.dynamic_cov(**positive_time_data)
        dynamic_cov = ParallelKalmanFilter.unsqueeze_to_dim(dynamic_cov, 4)

        if isinstance(self.observation_matrix, Tensor):
            observation_matrix = pydpf.multiple_unsqueeze(self.observation_matrix, 2, 0)
        else:
            observation_matrix = self.observation_matrix(**positive_time_data)
        observation_matrix = ParallelKalmanFilter.unsqueeze_to_dim(observation_matrix, 4)

        if isinstance(self.observation_offset, Tensor):
            observation_offset = pydpf.multiple_unsqueeze(self.observation_offset, 2, 0)
        else:
            observation_offset = self.observation_offset(**positive_time_data).transpose(0,1)
        observation_offset = ParallelKalmanFilter.unsqueeze_to_dim(observation_offset, 3)

        if isinstance(self.observation_cov, Tensor):
            observation_cov = pydpf.multiple_unsqueeze(self.observation_cov, 2, 0)
        else:
            observation_cov = self.observation_cov(**positive_time_data).transpose(0,1).contiguous()
        observation_cov= ParallelKalmanFilter.unsqueeze_to_dim(observation_cov, 4)

        return prior_mean, prior_cov, dynamic_matrix, dynamic_drift, dynamic_cov, observation_matrix, observation_offset, observation_cov

    def forward(self, time_extent, observation, **data):
        t = torch.arange(0, time_extent+1)
        data['observation'] = observation
        data["t"] = t
        prior_mean, prior_cov, dynamic_matrix, dynamic_drift, dynamic_cov, observation_matrix, observation_offset, observation_cov = self.initialise_kalman_parameters(data)

        dynamic_matrix = torch.cat((torch.zeros_like(dynamic_matrix[0:1]), dynamic_matrix), dim=0).contiguous()
        dynamic_drift = torch.cat((prior_mean.unsqueeze(0), dynamic_drift), dim=0).contiguous()
        dynamic_cov = torch.cat((prior_cov.unsqueeze(0), dynamic_cov), dim=0).contiguous()
        transp_obs_matrix = observation_matrix.transpose(-1,-2)
        innovation_cov = observation_cov + observation_matrix @ dynamic_cov @ transp_obs_matrix
        inv_innovation_cov_mult_obs_matrix = torch.linalg.solve(transp_obs_matrix, innovation_cov, left=False)
        kalman_gain = dynamic_cov @ inv_innovation_cov_mult_obs_matrix
        dynamic_weighting = (pydpf.multiple_unsqueeze(torch.eye(dynamic_matrix.size(-1), dtype=dynamic_matrix.dtype, device=dynamic_matrix.device), 2, 0) - kalman_gain @ observation_matrix)
        cond_dyn_matrix = dynamic_weighting @ dynamic_matrix
        cond_dyn_cov = dynamic_weighting @ dynamic_cov
        obs_surprise = (observation.unsqueeze(1).unsqueeze(-1) - observation_matrix @ dynamic_drift[..., None] - observation_offset[..., None])
        cond_dyn_offset = dynamic_drift + (kalman_gain @ obs_surprise)
        prop_inv_innovation_cov_mult_obs_matrix = dynamic_matrix.transpose(-1,-2) @ inv_innovation_cov_mult_obs_matrix
        likelihood_information_vec = prop_inv_innovation_cov_mult_obs_matrix @ obs_surprise
        likelihood_information_cov = prop_inv_innovation_cov_mult_obs_matrix @ observation_matrix @ dynamic_matrix








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
