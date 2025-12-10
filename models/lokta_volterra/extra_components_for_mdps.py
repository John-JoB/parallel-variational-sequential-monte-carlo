import torch
import pydpf
from einops import repeat
from models.generic_nets.FCNN import FCNN


class backward_prior(pydpf.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.device = generator.device
        self.maxes = pydpf.multiple_unsqueeze(torch.tensor([5, 7], device=self.device),2, 0 )
        self.mins = pydpf.multiple_unsqueeze(torch.tensor([0, 0], device=self.device), 2, 0)

    def sample(self, n_particles, batch_size, **data):
        rand = torch.rand((batch_size, n_particles, 2), device=self.device, generator=self.generator)
        return rand * (self.maxes - self.mins) + self.mins

class CombinationWeights(pydpf.Module):
    def __init__(self, device, dim=2):
        super().__init__()
        self.particle_encoder = FCNN(dim, 8, 8, "Relu", None, 3, device = device)
        self.weight_function = FCNN(8 + dim + 2, 1, 8, "Relu", "sigmoid", 3, device = device)
        self.max_weight = 1.0
        self.min_weight = 0.00001

    def score(self, state, observation, integrated_forward_weight, integrated_backward_weight, **data):
        expanded_obs = repeat(observation, "t b d -> t b p d", p = state.size(2))
        encoded_state = self.particle_encoder(state)
        fun_input = torch.cat([encoded_state, expanded_obs, integrated_forward_weight.unsqueeze(-1), integrated_backward_weight.unsqueeze(-1)], dim=-1)
        weights = self.weight_function(fun_input)
        weights = weights * (self.max_weight - self.min_weight) + self.min_weight
        # This measurement function is unrealistic it will assign weights too evenly, but prob helps stability. And is taken from the original paper.
        return torch.log(weights.squeeze())
