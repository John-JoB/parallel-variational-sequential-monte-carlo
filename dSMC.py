import pydpf
import torch
from torch import Tensor
from parallel_smoother_new import ParallelSmoother
import einops
from parallel_scan import parallel_associative_reduce
from math import log

class dSMC(pydpf.Module):

    def __init__(self, proposal, SSM, resampling_generator):
        super().__init__()
        self.proposal = proposal
        self.SSM = SSM
        self.resampling_generator = resampling_generator
        self.combine = dSMC.combination_operator_help(SSM.dynamic_model.log_density, resampling_generator)

    @staticmethod
    def combination_operator_help(dynamic_model, generator):
        def combination_operator(left, right):
            current_left_edge = left[0][..., -1, :]
            current_right_edge = right[0][..., 0, :]
            n = current_right_edge.shape[-2]
            flat_left_edge = einops.repeat(current_left_edge, 't b n ... -> (t b) (n m) ...', m = n)
            flat_right_edge = einops.repeat(current_right_edge, 't b m ... -> (t b) (n m) ...', n = n)
            dynamic_densities = dynamic_model(prev_state = flat_left_edge, state = flat_right_edge)
            left_L = left[1]
            right_L = right[1]
            weight = einops.rearrange(dynamic_densities, '(t b) ... -> t b ...', t = current_left_edge.size(0), b= current_left_edge.size(1)) + einops.repeat(right[2], 't b m ... -> t b (n m) ...', n = n)
            norm_weight, centre_L = pydpf.normalise(weight)
            centre_L = centre_L - 2*log(n)
            #Use multinomial resampling because particles are not order independent
            resampled_indices = torch.multinomial(torch.exp(norm_weight).flatten(0,-2), n, replacement=True, generator=generator).detach()
            resampled_indices = einops.rearrange(resampled_indices, "(t b) n -> t b n", t = current_left_edge.size(0), b = current_left_edge.size(1), n = n)
            resampled_indices_right = resampled_indices % n
            resampled_indices_left = resampled_indices // n
            resampled_state_left = pydpf.batched_select(left[0], resampled_indices_left)
            resampled_state_right = pydpf.batched_select(right[0], resampled_indices_right)
            resampled_state = torch.cat([resampled_state_left, resampled_state_right], dim=-2)
            return resampled_state, left_L + right_L + centre_L.squeeze(-1), pydpf.batched_select(left[2], resampled_indices_left)
        return combination_operator

    @staticmethod
    def reshaper(last_left):
        if last_left.ndim == 5:
            return torch.nn.functional.pad(last_left, (0, 0, 0, last_left.shape[-2]), mode='constant', value=torch.nan)
        return last_left


    def forward(self, n_particles: int,
                time_extent: int,
                aggregation_function: pydpf.Module | dict,
                observation: Tensor,
                *,
                gradient_regulariser: torch.autograd.Function | None = None,
                ground_truth: Tensor | None = None,
                control: Tensor | None = None,
                time: Tensor | None = None,
                series_metadata: Tensor | None = None) -> Tensor | dict:


        observation = observation[:time_extent + 1]
        if ground_truth is not None:
            ground_truth = ground_truth[:time_extent + 1]
        if control is not None:
            control = control[:time_extent + 1]
        if time is not None:
            time = time[:time_extent + 1]
        state, prop_density = self.proposal(n_particles, observation=observation, control=control, time=time, series_metadata=series_metadata)
        batched_data = ParallelSmoother._get_batched_dict(ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
        obs_score = self.SSM.observation_model.score(**batched_data)
        del (batched_data["state"])
        t_zero_data = ParallelSmoother._get_time_zero_data(state=state, observation=observation, control=control, time=time, series_metadata=series_metadata)
        prior_density = self.SSM.prior_model.log_density(**t_zero_data)
        obs_score = einops.rearrange(obs_score, "(t b) n -> t b n", t=time_extent + 1)

        kernels = obs_score - prop_density
        kernels[0] = kernels[0] + prior_density
        weight, likelihood_zero = pydpf.normalise(kernels[0])
        likelihood_zero = likelihood_zero - log(state.size(-2))
        out_state, elbo, _ = parallel_associative_reduce(self.combine, self.reshaper, False, state.unsqueeze(-2), torch.zeros((kernels.size(0), kernels.size(1)), device = kernels.device), kernels)
        out_state = out_state.permute(2, 0, 1, 3)
        out_state = out_state[:time_extent + 1]
        elbo = elbo.squeeze() + likelihood_zero.squeeze()
        weight = weight.unsqueeze(0).expand(time_extent + 1, -1, -1)
        if isinstance(aggregation_function, dict):
            output = {}
            for name, function in aggregation_function.items():
                output[name] = function(weight=weight, kernel=kernels, elbo=elbo, initial_likelihood=likelihood_zero, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation,
                                        state=out_state)
            return output
        return aggregation_function(weight=kernels[0], kernel=kernels, elbo=elbo, initial_likelihood=likelihood_zero, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
