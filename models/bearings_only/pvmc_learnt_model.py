import torch
from pydpf import Module
import pydpf
import einops
from models.generic_nets.FCNN import FCNN
from models.generic_nets.conv import ConvEncoder
import itertools

def _wrap_angle(angle):
    return (angle - torch.pi) % (2 * torch.pi) - torch.pi

def make_space_kernel(starting_pos_var, starting_angle_conc, generator):
    device = generator.device
    gaussian_part = pydpf.MultivariateGaussian(torch.zeros(2, device=device), torch.nn.Parameter(torch.eye(2, device=device) * starting_pos_var, requires_grad=True), diagonal_cov=True, generator=generator)
    vonmises_part = pydpf.VonMises(torch.zeros(1, device=device), torch.nn.Parameter(torch.tensor(starting_angle_conc, device=device), requires_grad=True), generator)
    compound_dist = pydpf.CompoundDistribution([gaussian_part, vonmises_part], generator)
    return compound_dist

class PriorModel(Module):
    def __init__(self, generator):
        super().__init__()
        self.tolerance_dist = make_space_kernel(10., 2., generator)

    def log_density(self, state, series_metadata, **data):
        res = state - series_metadata.unsqueeze(-2)
        res[..., -1] = _wrap_angle(res[..., -1])
        return self.tolerance_dist.log_density(res)

    def sample(self, **data):
        pass

class DynamicModelOld(Module):
    def __init__(self, generator):
        super().__init__()
        self.encoder = FCNN(4, 4, 8, "Relu", "tanh", 7, device=generator.device)
        self.scale_factor = torch.tensor([[[5., 5., 2., 2.]]], dtype=torch.float32, device=generator.device)
        self.weight_function = FCNN(8, 1, 8, "Relu", "sigmoid", 3, device = generator.device)
        self.max_weight = 1.0
        self.min_weight = 0.00001

    def log_density(self, state, prev_state, **data):
        transformed_state = torch.cat([prev_state[..., 0:2], torch.sin(prev_state[..., -1:]), torch.cos(prev_state[..., -1:])], dim=-1)
        encoded_state = self.encoder(transformed_state) + transformed_state
        concat = torch.cat([encoded_state, state[..., 0:2], torch.sin(state[..., -1:]), torch.cos(state[..., -1:])], dim=-1)
        weights = self.weight_function(concat)
        weights = weights * (self.max_weight - self.min_weight) + self.min_weight
        # This measurement function is unrealistic it will assign weights too evenly, but prob helps stability
        return torch.log(weights.squeeze())

    def sample(self, **data):
        pass

class DynamicModel(Module):
    def __init__(self, generator):
        super().__init__()
        self.tolerance_dist = make_space_kernel(10., 2., generator)
        self.encoder = FCNN(4, 4, 8, "Relu", "Sigmoid", 7, device=generator.device)
        self.scale_factor = torch.tensor([[[5., 5., 2., 2.]]], dtype=torch.float32, device=generator.device)

    def log_density(self, state, prev_state, **data):
        transformed_state = torch.cat([prev_state[..., 0:2], torch.sin(prev_state[..., -1:]), torch.cos(prev_state[..., -1:])], dim=-1)
        encoded_state = ((self.encoder(transformed_state) * 2) - 1) * self.scale_factor + transformed_state
        mean_state = torch.cat([encoded_state[..., 0:2], torch.atan2(encoded_state[..., 2:3], encoded_state[..., 3:])], dim=-1).contiguous()
        res = state - mean_state
        res[..., -1] = _wrap_angle(res[..., -1])
        return self.tolerance_dist.log_density(res)

    def sample(self, **data):
        pass

class ObservationModel(Module):
    def __init__(self, device):
        super().__init__()
        self.particle_encoder = FCNN(4, 8, 8, "Relu", None, 2, device = device)
        self.weight_function = FCNN(16, 1, 8, "Relu", "sigmoid", 2, device = device)
        self.max_weight = 1.0
        self.min_weight = 0.00001

    def score(self, observation, state, **data):
        expanded_obs = einops.repeat(observation, "b d -> b p d", p=state.size(1))
        transformed_state = torch.cat([state[..., 0:2], torch.sin(state[..., -1:]), torch.cos(state[..., -1:])], dim=-1)
        encoded_state = self.particle_encoder(transformed_state.contiguous())
        state_obs_comb = torch.cat([encoded_state, expanded_obs], dim=-1)
        weights = self.weight_function(state_obs_comb)
        weights = weights * (self.max_weight - self.min_weight) + self.min_weight
        #This measurement function is unrealistic it will assign weights too evenly, but prob helps stability
        return torch.log(weights.squeeze())

class ProposalModel(Module):
    def __init__(self, generator):
        super().__init__()
        device = generator.device
        starting_var = torch.tensor([[5.,0.,0.],[0.,5.,0.],[0.,0.,torch.pi/4]], device=device)
        self.dist = pydpf.MultivariateGaussian(torch.zeros(3, device=device), torch.nn.Parameter(starting_var, requires_grad=True), diagonal_cov=True, generator=generator)
        self.net = FCNN(12, 4, 8, "Relu", "id", 3, device = device)

    def forward(self, n_particles, observation, series_metadata, **data):
        pos_encode = einops.repeat(torch.arange(len(observation), device=observation.device), "t -> t b 1", b = observation.size(1))
        expanded_series_metadata = einops.repeat(series_metadata,  "b d -> t b d", t=len(observation))
        net_input = torch.cat([pos_encode, expanded_series_metadata, observation], dim=-1)
        net_out = self.net(net_input)
        mean_state = torch.cat([net_out[..., 0:2], torch.atan2(net_out[..., 2:3], net_out[..., 3:])], dim=-1).contiguous()
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        state = sample + mean_state.unsqueeze(-2)
        state[..., -1] = _wrap_angle(state[..., -1])
        return state, self.dist.log_density(sample)


class ObservationEncoder(Module):
    def __init__(self, device):
        super().__init__()
        layer = [{"type": "conv", "in_channels": 1, "out_channels": 8, "kernel_size": 5, "left_input_size": 5, "right_input_size": 5, "kernel_offset": 0, "activation": "relu"},
                  {"type": "linear", "in_features": 8, "out_features": 8, "activation": "relu"},
                  {"type": "linear", "in_features": 8, "out_features": 8, "activation": "relu"},
                  {"type": "dropout", "p": 0.5}]
        layer_two = [{"type": "conv", "in_channels": 8, "out_channels": 8, "kernel_size": 5, "left_input_size": 5, "right_input_size": 5, "kernel_offset": 0, "activation": "relu"},
                  {"type": "linear", "in_features": 8, "out_features": 8, "activation": "relu"},
                  {"type": "linear", "in_features": 8, "out_features": 8, "activation": "relu"},
                  {"type": "dropout", "p": 0.5}]
        final_layer = [{"type": "conv", "in_channels": 8, "out_channels": 8, "kernel_size": 5, "left_input_size": 5, "right_input_size": 5, "kernel_offset": 0, "activation": "relu"},
                  {"type": "linear", "in_features": 8, "out_features": 8, "activation": "relu"},
                  {"type": "linear", "in_features": 8, "out_features": 8, "activation": "tanh"}]

        self.net = ConvEncoder(list(itertools.chain.from_iterable([layer, layer_two, final_layer])), device=device)

    def forward(self, *args, observation, **data):
        return self.net(observation)

