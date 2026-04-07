import torch

from pydpf import Module
from torch import Tensor
import pydpf
import einops
from parallel_smoother_new import ParallelSmoother



class Truncated(Module):
    def __init__(self, proposal, SSM):
        super().__init__()
        self.proposal = proposal
        self.SSM = SSM

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
        batched_data = ParallelSmoother._get_batched_dict(ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
        state_repeat = einops.repeat(state[1:], 't b n d -> (t b) (n m) d', m=n_particles)
        prev_state_repeat = einops.repeat(state[:-1], 't b n d -> (t b) (m n) d', m=n_particles)
        obs_score = self.SSM.observation_model.score(**batched_data)
        del (batched_data["state"])
        t_zero_data = ParallelSmoother._get_time_zero_data(state=state, observation=observation, control=control, time=time, series_metadata=series_metadata)
        prior_density = self.SSM.prior_model.log_density(**t_zero_data)
        dynamic_density = self.SSM.dynamic_model.log_density(state=state_repeat, prev_state=prev_state_repeat, **batched_data)
        obs_score = einops.rearrange(obs_score, "(t b) n -> t b n", t=time_extent + 1)
        prop_density = einops.rearrange(prop_density, "t b n -> t b n", t=time_extent + 1)
        dynamic_density = einops.rearrange(dynamic_density, "(t b) (m n) -> t b n m", t=time_extent, m=n_particles)
        dynamic_density = torch.logsumexp(dynamic_density, dim=-1)
        dynamic_density = torch.cat([prior_density.unsqueeze(0), dynamic_density], dim=0)
        weights = obs_score + dynamic_density - prop_density
        weights, l = pydpf.normalise(weights)

        time_zero_l = torch.nan
        kernels = torch.nan
        if isinstance(aggregation_function, dict):
            output = {}
            for name, function in aggregation_function.items():
                output[name] = function(weight=weights, kernel = kernels, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
            return output
        return aggregation_function(weight=weights, kernel = kernels, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
