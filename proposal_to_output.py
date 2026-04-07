import torch

from pydpf import Module
from torch import Tensor
from math import log



class ProposalRunner(Module):
    def __init__(self, proposal):
        super().__init__()
        self.proposal = proposal

    @staticmethod
    def print_grad(grad):
        print("Gradient for state:")
        print(grad[3::4])
        print("---")

    def forward(self, n_particles: int,
                time_extent: int,
                aggregation_function: Module | dict,
                observation: Tensor,
                *,
                gradient_regulariser: None = None,
                ground_truth: Tensor | None = None,
                control: Tensor | None = None,
                time: Tensor | None = None,
                series_metadata: Tensor | None = None) -> Tensor|dict:

        observation = observation[:time_extent + 1]
        if ground_truth is not None:
            ground_truth = ground_truth[:time_extent + 1]
        if control is not None:
            control = control[:time_extent + 1]
        if time is not None:
            time = time[:time_extent + 1]
        state, prop_density = self.proposal(n_particles, observation=observation, control=control, time=time, series_metadata=series_metadata)
        weights = torch.zeros_like(prop_density) - log(n_particles)
        time_zero_l = torch.nan
        kernels = torch.nan
        if isinstance(aggregation_function, dict):
            output = {}
            for name, function in aggregation_function.items():
                output[name] = function(weight=weights, kernel = kernels, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
            return output
        return aggregation_function(weight=weights, kernel = kernels, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
