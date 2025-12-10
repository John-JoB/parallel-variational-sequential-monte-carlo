from pydpf import KernelDPF
from pydpf import Module
import pydpf
from torch import Tensor
import torch
import numpy as np
from math import log

class MDPS(Module):

    @staticmethod
    def _get_time_data(t: int, **data) -> dict:
        time_dict = {k: v[t] for k, v in data.items() if k != 'series_metadata' and v is not None}
        time_dict['t'] = t
        if data['time'] is not None and t > 0:
            time_dict['prev_time'] = data['time'][t - 1]
        if data['series_metadata'] is not None:
            time_dict['series_metadata'] = data['series_metadata']
        return time_dict

    def __init__(self, SSM,  backward_SSM, combination_model, forward_kernel, backward_kernel, forward_resample_kernel, backward_resample_kernel):
        super().__init__()
        self.SSM = SSM
        self.forward_kernel = forward_kernel
        self.backward_kernel = backward_kernel
        self.forward_filter = KernelDPF(SSM, forward_resample_kernel)
        self.backward_filter = KernelDPF(backward_SSM, backward_resample_kernel)
        self.combination_model = combination_model

    def _get_existing_data(self, **input_data):
        return {k:v for k,v in input_data.items() if v is not None}

    def forward(self, n_particles: int,
                time_extent: int,
                aggregation_function: Module | dict,
                observation: Tensor,
                *,
                gradient_regulariser: torch.autograd.Function | None = None,
                ground_truth: Tensor | None = None,
                control: Tensor | None = None,
                time: Tensor | None = None,
                series_metadata: Tensor | None = None) -> Tensor | dict:
        output_dict = False
        if isinstance(aggregation_function,dict):
            output_dict = True
            output = {}
            self.aggregation_function = torch.nn.ModuleDict(aggregation_function)
        data = self._get_existing_data(ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata)
        observation = observation[:time_extent+1]
        forward_res = self.forward_filter(n_particles=n_particles, time_extent=time_extent, observation=observation, aggregation_function={"state": pydpf.State(), "weight": pydpf.Weight()}, gradient_regulariser=gradient_regulariser, **data)
        forward_state = forward_res["state"]
        forward_weight = forward_res["weight"]
        backward_data = {k:torch.flip(v, (0,)) for k,v in data.items()}
        backward_state = self.backward_filter(n_particles=n_particles, time_extent=time_extent, observation=torch.flip(observation, (0,)), aggregation_function=pydpf.State(), gradient_regulariser=gradient_regulariser, **backward_data)
        #Don't actually sample the particles, just take all of them. This is unbiased due to GMIS Elvira 2015
        #And we can ignore the importance weight as we want p(x_t | y_0:t-1) (and backwards equivalent)
        combined_particle_state = torch.cat([forward_state, backward_state], dim=-2)
        weight_size = tuple([forward_state.size(i) for i in range(combined_particle_state.dim() - 1)])
        #Can be generalised for resamplers that induce importance weights, will do if need arises
        integrated_forward_weight = self.forward_kernel.log_density(combined_particle_state, forward_state, torch.full(weight_size, -np.log(forward_state.size(-2)), device=combined_particle_state.device))
        integrated_backward_weight = self.backward_kernel.log_density(combined_particle_state, backward_state, torch.full(weight_size, -np.log(backward_state.size(-2)), device=combined_particle_state.device))
        combined_integrated_weight = torch.logaddexp(integrated_forward_weight, integrated_backward_weight) - np.log(2)
        combination_weights = self.combination_model.score(combined_particle_state, observation, integrated_forward_weight, integrated_backward_weight, **data)
        #This line seems wrong, correct application of REINFORCE would detach the particles here but this follows the original paper's code.
        output_weights = combination_weights - combined_integrated_weight.detach()
        #Could be vectorised, but this is hardly going to be the performance bottleneck
        output_weights, elbo_factors = pydpf.normalise(output_weights)
        elbo = torch.sum(torch.logsumexp(forward_weight, dim = -1), dim=0) - log(n_particles)
        if output_dict:
            output = {}
            for k, v in aggregation_function.items():
                output[k] = v(weight=output_weights, kernel=None, elbo=elbo, initial_likelihood=None, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=combined_particle_state)
        else:
                output = aggregation_function(weight=output_weights, kernel=None, elbo=elbo, initial_likelihood=None, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=combined_particle_state)
        return output

