import pydpf
import torch
from experiments.common.training import ExperimentRun


class Kalman_mean(pydpf.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean, **kwargs):
        return mean

    def batch_dict(self, **kwargs):
        return 1

class Kalman_MSE(pydpf.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean, ground_truth, **data):
        return torch.sum((mean - ground_truth) ** 2, dim=-1)

    def batch_dict(self, **kwargs):
        return 1

class Kalman_covariance(pydpf.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cov, **kwargs):
        return cov

    def batch_dict(self, **kwargs):
        return 1

class Kalman_log_likelihood_factors(pydpf.Module):
    def __init__(self):
        super().__init__()

    def forward(self, likelihood_factor, **kwargs):
        return likelihood_factor

    def batch_dict(self, **kwargs):
        return 1

class Kalman_log_likelihood(pydpf.Module):
    def __init__(self):
        super().__init__()

    def forward(self, likelihood_factor, **kwargs):
        return torch.sum(likelihood_factor, dim=0)

    def batch_dict(self, **kwargs):
        return 0

class KalmanRun(ExperimentRun):
    def __init__(self, kalman, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.kalman = kalman

    @staticmethod
    def apply_time_extent(time_extent, data):
        new_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                new_data[k] = v[:time_extent + 1]
        return new_data

    def run(self, mode, run_info, **data):
        data = self.apply_time_extent(run_info[mode]["time_extent"], data)
        if isinstance(self.kalman, pydpf.KalmanFilter):
            mean, cov, likelihood_factor = self.kalman(run_info[mode]["time_extent"], data["observation"])
        else:
            mean, cov, likelihood_factor = self.kalman(run_info[mode]["time_extent"], **data)
        kalman_outputs = {"mean": mean, "cov": cov, "likelihood_factor": likelihood_factor}
        outputs = {}
        batch_dict = {}
        for name, fun in run_info[mode]["output_function"].items():
            outputs[name] = fun(**{**kalman_outputs, **data})
            batch_dict[name] = fun.batch_dict(**{**kalman_outputs, **data})
        return outputs, batch_dict