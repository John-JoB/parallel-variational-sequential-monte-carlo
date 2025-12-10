import einops
import pydpf
import torch
from models.generic_nets.FCNN import FCNN
from models.generic_nets.conv import ConvEncoder

class LearnedDynamicModel(pydpf.Module):
    def __init__(self, step_size = 3/256, depth = 2, hidden_dim = 32, device = torch.device('cpu'), generator = torch.default_generator):
        super().__init__()
        self.depth = depth
        self.a = FCNN(2, 2, hidden_dim, n_hidden_layers= depth - 1, activation_function="swish", output_function="id", device = device)
        self.b = FCNN(2, 2, hidden_dim, n_hidden_layers= depth - 1, activation_function="swish", output_function="id", device = device)
        self.step_size = torch.tensor(step_size, device=device)
        self.root_step_size = torch.sqrt(self.step_size)
        self.dist = pydpf.StandardGaussian(2, generator)

    def sample(self, prev_state, **data):
        drift = self.a(prev_state)
        sd = torch.exp(torch.nn.functional.tanh(self.b(prev_state)))
        rng = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        return prev_state + drift * self.step_size + sd * self.root_step_size * rng

    def log_density(self, prev_state, state, **data):
        drift = self.a(prev_state)
        sd = torch.exp(torch.nn.functional.tanh(self.b(prev_state)))
        normalised_state = (state - prev_state - drift * self.step_size)/(sd * self.root_step_size)
        return self.dist.log_density(normalised_state) - torch.log(sd * self.root_step_size).sum(dim=-1)

class proposal_model(pydpf.Module):
    def __init__(self, device=torch.device('cpu'), generator = torch.default_generator):
        super().__init__()
        self.gen = generator
        layers = []
        in_dims = [2, 16, 32, 32, 32]
        for di in range(len(in_dims) - 1):
            layers += [{"type": "conv", "in_channels": in_dims[di], "out_channels": in_dims[di+1], "kernel_size": 7, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "relu"},
                       {"type": "linear", "in_features": in_dims[di + 1], "out_features": in_dims[di + 1], "bias": True, "device": device, "activation": "relu"}]
        layers += [{"type": "conv", "in_channels": in_dims[-1], "out_channels": 4, "kernel_size": 5, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "id"}]
        self.conv = ConvEncoder(layers, self.gen.device)
        self.dist = pydpf.StandardGaussian(2, generator)

    def forward(self, n_particles, observation, **data):
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles)).detach()
        observation = (observation / 10) - .5
        mean_sd = self.conv(observation)
        log_sd = torch.nn.functional.tanh(mean_sd[:, :, None, 2:])
        sd = torch.exp(log_sd)
        state = sample * sd + mean_sd[:, :, None, :2]
        sample_stl = (state - mean_sd[:, :, None, :2].detach())/sd.detach()
        return state, self.dist.log_density(sample_stl) - torch.sum(log_sd.detach(), dim=-1)
