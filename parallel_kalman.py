import pydpf
from pydpf import Module
import torch
from torch import Tensor
from math import log
from parallel_scan import parallel_associative_scan

def apply_time_extent(time_extent, data):
    new_data = {}
    for k, v in data.items():
        if isinstance(v, Tensor):
            new_data[k] = v[:time_extent + 1]
    return new_data

class ParallelKalmanFilter(Module):
    def __init__(self, prior_model, dynamic_model, observation_model):
        super().__init__()
        self.prior_model = prior_model
        self.dynamic_model = dynamic_model
        self.observation_model = observation_model
        self.prior_mean = prior_model.mean
        self.dynamic_matrix = dynamic_model.mean_fun.weight
        self.observation_matrix = observation_model.mean_fun.weight
        self.dynamic_drift = dynamic_model.mean_fun.bias
        self.observation_offset = observation_model.mean_fun.bias
        self.prior_chole_cov = prior_model.cholesky_covariance
        self.dynamic_chole_cov = dynamic_model.dist.cholesky_covariance
        self.observation_chole_cov = observation_model.dist.cholesky_covariance

    @pydpf.cached_property
    def prior_cov(self):
        return self.prior_chole_cov @ self.prior_chole_cov.T

    @pydpf.cached_property
    def dynamic_cov(self):
        return self.dynamic_chole_cov @ self.dynamic_chole_cov.T

    @pydpf.cached_property
    def observation_cov(self):
        return self.observation_chole_cov @ self.observation_chole_cov.T


    @staticmethod
    def unsqueeze_to_dim(a, b):
        if isinstance(b, Tensor):
            b = b.dim()
        if a.dim() == b:
            return a
        if a.dim() > b:
            raise ValueError(f"Dimension of the first tensor must be less than the second tensor found dims {a.dim()}, {b}")
        return pydpf.multiple_unsqueeze(a, b - a.dim(), 0)

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
            prior_cov = self.prior_cov
        else:
            prior_cov = self.prior_cov(**time_zero_data)
        prior_cov = ParallelKalmanFilter.unsqueeze_to_dim(prior_cov, 3)

        if isinstance(self.dynamic_matrix, Tensor):
            dynamic_matrix = self.dynamic_matrix
        else:
            dynamic_matrix = self.dynamic_matrix(**positive_time_data)
        dynamic_matrix = ParallelKalmanFilter.unsqueeze_to_dim(dynamic_matrix, 4)

        if isinstance(self.dynamic_drift, Tensor):
            dynamic_drift = self.dynamic_drift
        else:
            dynamic_drift = self.dynamic_drift(**positive_time_data)
        dynamic_drift =  ParallelKalmanFilter.unsqueeze_to_dim(dynamic_drift, 3)

        if isinstance(self.dynamic_cov, Tensor):
            dynamic_cov = self.dynamic_cov
        else:
            dynamic_cov = self.dynamic_cov(**positive_time_data)
        dynamic_cov = ParallelKalmanFilter.unsqueeze_to_dim(dynamic_cov, 4)

        if isinstance(self.observation_matrix, Tensor):
            observation_matrix = self.observation_matrix
        else:
            observation_matrix = self.observation_matrix(**positive_time_data)
        observation_matrix = ParallelKalmanFilter.unsqueeze_to_dim(observation_matrix, 4)

        if isinstance(self.observation_offset, Tensor):
            observation_offset = self.observation_offset
        else:
            observation_offset = self.observation_offset(**positive_time_data).transpose(0,1)
        observation_offset = ParallelKalmanFilter.unsqueeze_to_dim(observation_offset, 3)

        if isinstance(self.observation_cov, Tensor):
            observation_cov = self.observation_cov
        else:
            observation_cov = self.observation_cov(**positive_time_data).transpose(0,1).contiguous()
        observation_cov= ParallelKalmanFilter.unsqueeze_to_dim(observation_cov, 4)

        return prior_mean, prior_cov, dynamic_matrix, dynamic_drift, dynamic_cov, observation_matrix, observation_offset, observation_cov

    @staticmethod
    def combination_op(left, right):
        a_left, b_left, c_left, nu_left, j_left = left
        a_right, b_right, c_right, nu_right, j_right = right
        id = pydpf.multiple_unsqueeze(torch.eye(a_left.size(-1), dtype=a_left.dtype, device=a_left.device), 2, 0)
        t1 = id + c_left @ j_right
        t2 = id + j_right @ c_left
        prob_factor = torch.linalg.solve(t1, a_right, left=False)
        likelihood_factor = torch.linalg.solve(t2, a_left.transpose(-1,-2), left=False)
        a_new = prob_factor @ a_left
        b_new = prob_factor @ (b_left + c_left @ nu_right) + b_right
        c_new = prob_factor @ c_left @ a_right.transpose(-1,-2) + c_right
        nu_new = likelihood_factor @ (nu_right - j_right @ b_left) + nu_left
        j_new = likelihood_factor @ j_right @ a_left + j_left
        return a_new, b_new, c_new, nu_new, j_new


    @staticmethod
    def propagate_dist(mean, cov, condition_matrix, offset, conditional_cov):
        new_mean = condition_matrix @ mean + offset
        new_cov = condition_matrix @ cov @ condition_matrix.transpose(-1,-2) + conditional_cov
        return new_mean, new_cov

    @staticmethod
    def expand_time_and_batch(tensor, match):
        return tensor.expand((match.size(0) - 1, match.size(1), tensor.size(2), tensor.size(3)))

    @staticmethod
    def expand_batch(tensor, match):
        return tensor.expand((match.size(1), tensor.size(1), tensor.size(2)))

    def forward(self, time_extent, observation, need_predictive=False, **data):
        t = torch.arange(0, time_extent+1)
        if "observation" not in data:
            data['observation'] = observation
        if "t" not in data:
            data["t"] = t
        data = apply_time_extent(time_extent, data)
        prior_mean, prior_cov, dynamic_matrix, dynamic_drift, dynamic_cov, observation_matrix, observation_offset, observation_cov = self.initialise_kalman_parameters(data)
        #More efficient to special case the t=0 because it is common for all elements t>0 to be the same
        first_dynamic_matrix = torch.zeros_like(dynamic_matrix[0])
        first_dynamic_drift = prior_mean
        first_dynamic_cov = prior_cov
        first_observation_matrix = observation_matrix[0]
        first_observation_offset = observation_offset[0]
        first_observation_cov = observation_cov[0]

        first_cond_dyn_matrix = first_dynamic_matrix
        first_innovation_cov = first_observation_cov + first_observation_matrix @ first_dynamic_cov @ first_observation_matrix.transpose(-1,-2)
        first_inv_innovation_cov_mult_obs_matrix = torch.linalg.solve(first_innovation_cov, first_observation_matrix.transpose(-2, -1), left=False)
        first_kalman_gain = first_dynamic_cov @ first_inv_innovation_cov_mult_obs_matrix
        first_cond_dyn_offset = first_dynamic_drift[..., None] + first_kalman_gain @ (observation[0][..., None] - first_observation_matrix @ first_dynamic_drift[..., None] - first_observation_offset[..., None])
        first_dynamic_weighting = torch.eye(dynamic_drift.shape[-1], dtype=dynamic_drift.dtype, device=dynamic_drift.device).unsqueeze(0) - first_kalman_gain @ first_observation_matrix
        first_cond_dyn_cov = first_dynamic_weighting @ first_dynamic_cov

        if observation_matrix.size(0) > 1:
            observation_matrix = observation_matrix[1:]
            observation_offset = observation_offset[1:]
            observation_cov = observation_cov[1:]


        transp_obs_matrix = observation_matrix.transpose(-1,-2)
        innovation_cov = observation_cov + observation_matrix @ dynamic_cov @ transp_obs_matrix
        inv_innovation_cov_mult_obs_matrix = torch.linalg.solve(innovation_cov, transp_obs_matrix, left=False)
        kalman_gain = dynamic_cov @ inv_innovation_cov_mult_obs_matrix
        dynamic_weighting = (pydpf.multiple_unsqueeze(torch.eye(dynamic_matrix.size(-1), dtype=dynamic_matrix.dtype, device=dynamic_matrix.device), 2, 0) - kalman_gain @ observation_matrix)
        cond_dyn_matrix = dynamic_weighting @ dynamic_matrix
        cond_dyn_cov = dynamic_weighting @ dynamic_cov
        obs_surprise = observation[1:][..., None] - observation_matrix @ dynamic_drift[..., None] - observation_offset[..., None]
        cond_dyn_offset = dynamic_drift[..., None] + (kalman_gain @ obs_surprise)
        prop_inv_innovation_cov_mult_obs_matrix = dynamic_matrix.transpose(-1,-2) @ inv_innovation_cov_mult_obs_matrix
        likelihood_information_vec = prop_inv_innovation_cov_mult_obs_matrix @ obs_surprise
        likelihood_information_cov = prop_inv_innovation_cov_mult_obs_matrix @ observation_matrix @ dynamic_matrix

        cond_dyn_matrix = self.expand_time_and_batch(cond_dyn_matrix, observation)
        cond_dyn_cov = self.expand_time_and_batch(cond_dyn_cov, observation)
        likelihood_information_vec = self.expand_time_and_batch(likelihood_information_vec, observation)
        likelihood_information_cov = self.expand_time_and_batch(likelihood_information_cov, observation)
        first_cond_dyn_matrix = self.expand_batch(first_cond_dyn_matrix, observation)
        first_cond_dyn_cov = self.expand_batch(first_cond_dyn_cov, observation)

        cond_dyn_matrix = torch.concat((first_cond_dyn_matrix.unsqueeze(0), cond_dyn_matrix), dim=0).contiguous()
        cond_dyn_offset = torch.concat((first_cond_dyn_offset.unsqueeze(0), cond_dyn_offset), dim=0).contiguous()
        cond_dyn_cov = torch.concat((first_cond_dyn_cov.unsqueeze(0), cond_dyn_cov), dim=0).contiguous()
        likelihood_information_vec = torch.concat((torch.zeros_like(likelihood_information_vec[0:1]), likelihood_information_vec), dim=0).contiguous()
        likelihood_information_cov = torch.concat((torch.zeros_like(likelihood_information_cov[0:1]), likelihood_information_cov), dim=0).contiguous()

        _, kalman_means, kalman_covs, _, _ = parallel_associative_scan(ParallelKalmanFilter.combination_op, cond_dyn_matrix, cond_dyn_offset, cond_dyn_cov, likelihood_information_vec, likelihood_information_cov)

        state_predictive_means, state_predictive_covs = ParallelKalmanFilter.propagate_dist(kalman_means[:-1], kalman_covs[:-1], dynamic_matrix, dynamic_drift.unsqueeze(-1), dynamic_cov)
        batched_prior_mean = self.expand_batch(prior_mean.unsqueeze(-1), observation)
        batched_prior_cov = self.expand_batch(prior_cov, observation)

        state_predictive_means_ap = torch.cat((batched_prior_mean.unsqueeze(0), state_predictive_means), dim=0)
        state_predictive_covs_ap = torch.cat((batched_prior_cov.unsqueeze(0), state_predictive_covs), dim=0)
        obs_predictive_means, obs_predictive_covs = ParallelKalmanFilter.propagate_dist(state_predictive_means_ap, state_predictive_covs_ap, observation_matrix, observation_offset.unsqueeze(-1), observation_cov)
        L_cov = torch.linalg.cholesky(obs_predictive_covs)
        scaled_residual = torch.linalg.solve_triangular(L_cov, obs_predictive_means - observation[..., None], upper=False)
        exp_term = torch.sum(scaled_residual**2, dim=(-1,-2))
        linear_term = torch.sum(torch.log(torch.diagonal(L_cov, dim1=-2, dim2=-1)), dim=-1)
        log_likelihood_factors = -0.5*(exp_term + L_cov.size(-1)*log(2*torch.pi)) - linear_term
        if need_predictive:
            return kalman_means.squeeze(-1), kalman_covs, log_likelihood_factors, state_predictive_means, state_predictive_covs
        return kalman_means.squeeze(-1), kalman_covs, log_likelihood_factors


class ParallelKalmanSmoother(ParallelKalmanFilter):
    def __init__(self, prior_model, dynamic_model, observation_model):
        super().__init__(prior_model, dynamic_model, observation_model)

    @staticmethod
    def combination_op(right, left):
        left_e, left_g, left_l = left
        right_e, right_g, right_l = right
        new_e = left_e @ right_e
        new_g, new_l = ParallelKalmanFilter.propagate_dist(right_g, right_l, left_e, left_g, left_l)
        return new_e, new_g, new_l


    def forward(self, time_extent, observation, **data):
        t = torch.arange(0, time_extent + 1)
        data['observation'] = observation
        data["t"] = t
        data = apply_time_extent(time_extent, data)
        prior_mean, prior_cov, dynamic_matrix, dynamic_drift, dynamic_cov, observation_matrix, observation_offset, observation_cov = self.initialise_kalman_parameters(data)
        kalman_filter = ParallelKalmanFilter(self.prior_model, self.dynamic_model, self.observation_model)
        filter_means, filter_covs, likelihood_factors, predictive_means, predictive_covs = kalman_filter(time_extent, need_predictive=True, **data)
        backwards_dyn_matrix = torch.linalg.solve(predictive_covs, filter_covs[:-1] @ dynamic_matrix.transpose(-1,-2), left=False)
        backwards_dyn_drift = filter_means[:-1][..., None] - backwards_dyn_matrix @ predictive_means
        backwards_dyn_cov = filter_covs[:-1] - backwards_dyn_matrix @ dynamic_matrix @ filter_covs[:-1]
        backwards_dyn_matrix = torch.concat((backwards_dyn_matrix, torch.zeros_like(backwards_dyn_matrix[0:1])), dim=0).contiguous()
        backwards_dyn_drift = torch.concat((backwards_dyn_drift, filter_means[-1:][...,None]), dim=0).contiguous()
        backwards_dyn_cov = torch.concat((backwards_dyn_cov, filter_covs[-1:]), dim=0).contiguous()
        _, smoothed_mean, smoothed_cov = parallel_associative_scan(ParallelKalmanSmoother.combination_op, torch.flip(backwards_dyn_matrix, (0,)), torch.flip(backwards_dyn_drift, (0,)), torch.flip(backwards_dyn_cov, (0,)))
        return torch.flip(smoothed_mean.squeeze(-1), (0,)), torch.flip(smoothed_cov, (0,)), likelihood_factors

