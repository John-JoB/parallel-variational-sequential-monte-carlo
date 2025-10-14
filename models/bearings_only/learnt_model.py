import torch
import pydpf
from pydpf import Module
from einops import repeat

from models.generic_nets.FCNN import FCNN

class DynamicsModel(Module):
    def __init__(self, generator):
        super().__init__()
        self.dist = pydpf.StandardGaussian(4, generator)
        self.encoder = FCNN(4, 8, 8, "Relu", None, 3, device = generator.device)
        self.forward_transform = FCNN(12, 4, 8, "Relu", "sigmoid", 3, device = generator.device)
        self.scale_factor = torch.tensor([[[5., 5., 2., 2.]]], dtype=torch.float32, device = generator.device)


    def sample(self, prev_state, **data):
        transformed_state = torch.cat([prev_state[..., 0:2], torch.sin(prev_state[..., -1:]), torch.cos(prev_state[..., -1:])], dim=-1)
        encoded_state = self.encoder(transformed_state)
        noise = self.dist.sample((encoded_state.size(0), encoded_state.size(1)))
        transform_input = torch.cat([encoded_state, noise], dim=-1)
        residual_state = ((self.forward_transform(transform_input) * 2.) - 1.) * self.scale_factor
        propagated_state = transformed_state + residual_state
        return torch.cat([propagated_state[..., 0:2], torch.atan2(propagated_state[..., 2:3], propagated_state[..., 3:])], dim=-1).contiguous()


class ObservationEncoder(Module):
    def __init__(self, device):
        super().__init__()
        self.net = FCNN(1, 8, 8, "Relu", None, 2, device = device)

    def forward(self, *args, observation, **data):
        return self.net.forward(observation)


class ObservationModel(Module):
    def __init__(self, device):
        super().__init__()
        self.particle_encoder = FCNN(4, 8, 8, "Relu", None, 2, device = device)
        self.weight_function = FCNN(16, 1, 8, "Relu", "sigmoid", 2, device = device)
        self.max_weight = 1.0
        self.min_weight = 0.00001

    def score(self, observation, state, **data):
        expanded_obs = repeat(observation, "b d -> b p d", p=state.size(1))
        transformed_state = torch.cat([state[..., 0:2], torch.sin(state[..., -1:]), torch.cos(state[..., -1:])], dim=-1)
        encoded_state = self.particle_encoder(transformed_state.contiguous())
        state_obs_comb = torch.cat([encoded_state, expanded_obs], dim=-1)
        weights = self.weight_function(state_obs_comb)
        weights = weights * (self.max_weight - self.min_weight) + self.min_weight
        #This measurement function is unrealistic it will assign weights too evenly, but prob helps stability
        return torch.log(weights.squeeze())


class CombinationWeights(Module):
    def __init__(self, device):
        super().__init__()
        self.particle_encoder = FCNN(4, 8, 8, "Relu", None, 3, device = device)
        self.weight_function = FCNN(18, 1, 8, "Relu", "sigmoid", 3, device = device)
        self.max_weight = 1.0
        self.min_weight = 0.00001

    def score(self, state, observation, integrated_forward_weight, integrated_backward_weight, **data):
        transformed_state = torch.cat([state[..., 0:2], torch.sin(state[..., -1:]), torch.cos(state[..., -1:])], dim=-1)
        expanded_obs = repeat(observation, "t b d -> t b p d", p = state.size(2))
        encoded_state = self.particle_encoder(transformed_state)
        fun_input = torch.cat([encoded_state, expanded_obs, integrated_forward_weight.unsqueeze(-1), integrated_backward_weight.unsqueeze(-1)], dim=-1)
        weights = self.weight_function(fun_input)
        weights = weights * (self.max_weight - self.min_weight) + self.min_weight
        # This measurement function is unrealistic it will assign weights too evenly, but prob helps stability
        return torch.log(weights.squeeze())

class forward_prior(Module):
    def __init__(self, generator):
        super().__init__()
        device = generator.device
        position_noise_cov = torch.eye(2, device=device) * 1e-4
        position_noise_dist = pydpf.MultivariateGaussian(torch.zeros(2, device=device), position_noise_cov, generator=generator)
        angle_noise_dist = pydpf.VonMises(torch.zeros(1, device=device), torch.full((1,), 100, device=device), generator=generator)
        self.noise_dist = pydpf.CompoundDistribution([position_noise_dist, angle_noise_dist], generator=generator)

    def sample(self, n_particles, batch_size, series_metadata, **data):
            true_initial_position = series_metadata
            noise = self.noise_dist.sample((batch_size, n_particles))
            return noise + true_initial_position.unsqueeze(-2)

class backward_prior(Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.device = generator.device
        self.maxes = pydpf.multiple_unsqueeze(torch.tensor([10, 10, torch.pi], device=self.device),2, 0 )
        self.mins = pydpf.multiple_unsqueeze(torch.tensor([-10, -10, -torch.pi], device=self.device), 2, 0)

    def sample(self, n_particles, batch_size, **data):
        rand = torch.rand((batch_size, n_particles, 3), device=self.device, generator=self.generator)
        return rand * (self.maxes - self.mins) + self.mins

