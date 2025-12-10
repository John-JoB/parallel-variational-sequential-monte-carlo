
import pydpf
import torch
from experiments.SPX.main import device
from models.generic_nets.FCNN import FCNN#, ResNet
from models.generic_nets.Normalizing_flow import RealNVP_cond, NormalizingFlowModel_cond
from models.generic_nets.conv import ConvEncoder

class StudentsT(pydpf.distributions.Distribution):
    def __init__(self, dim, df, generator, loc=0., scale = 1.):
        super().__init__(generator)
        self.df = df
        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df, device = device)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, device = device)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device = device)
        self.dist = torch.distributions.StudentT(df, loc, scale)
        self.dim = dim

    def sample(self, sample_size):
        return self.dist.sample((*sample_size, self.dim))

    def log_density(self, sample):
        return self.dist.log_prob(sample).sum(dim=-1)




class observation_model(pydpf.Module):
    def __init__(self, dx, generator, n_layers, hidden_dim, n_kernels):
        super().__init__()
        self.gen = generator
        self.tanh_scale = torch.nn.Parameter(torch.atanh(torch.log(torch.tensor(1., device=generator.device) / 3)/2))
        self.kernel = pydpf.StandardGaussian(1, generator)
        self.dist = pydpf.KernelMixture(self.kernel, generator, pydpf.MultinomialResampler(generator))
        self.net = FCNN(dx, n_kernels*2, hidden_dim, activation_function="swish", output_function="id", n_hidden_layers=n_layers, device=generator.device)
        self.temperature = torch.nn.Parameter(torch.tensor(5., device=generator.device), requires_grad=True)

    @pydpf.cached_property
    def log_scale(self):
        return torch.tanh(self.tanh_scale) * 2


    def sample(self, state, **data):
        dist_info = self.net(state)
        locs = dist_info[..., :dist_info.size(-1)//2].unsqueeze(-1)
        weights = dist_info[..., dist_info.size(-1)//2:] / self.temperature
        weights = torch.softmax(weights, dim=-1)
        locs = locs.squeeze(-1)
        idx = torch.multinomial(weights.flatten(0,1), 1, True, generator=self.gen).reshape(weights.size(0), weights.size(1), -1)
        chosen_locs = pydpf.batched_select(locs, idx)
        sample = self.dist.kernel.sample((state.size(0), state.size(1))) * torch.exp(self.log_scale)
        return sample + chosen_locs

    def score(self, observation, state, **data):
        dist_info = self.net(state)
        locs = dist_info[..., :dist_info.size(-1) // 2]
        weights = dist_info[..., dist_info.size(-1) // 2:] / self.temperature
        weights = torch.log_softmax(weights, dim=-1)
        normalised = (observation.unsqueeze(-1) - locs)/torch.exp(self.log_scale)
        log_density = self.kernel.log_density(normalised.unsqueeze(-1))
        return torch.logsumexp(log_density + weights, dim=-1) - self.log_scale

class prior_model(pydpf.Module):
    def __init__(self, dx, generator):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(5., device=generator.device))
        self.dx = dx
        self.dist = StudentsT(dx, 5., generator=generator)

    def sample(self,  batch_size, n_particles, **data):
        return self.dist.sample((batch_size, n_particles)) * self.scale

    def log_density(self, state, **data):
        return self.dist.log_density(state / self.scale) - torch.log(self.scale)


class dynamic_model(pydpf.Module):
    def __init__(self, dx, generator, n_layers, depth_per_layer, hidden_dim):
        super().__init__()
        self.dx = dx
        self.gen = generator
        real_nvps = []
        for i in range(n_layers):
            real_nvps += [RealNVP_cond(dx, hidden_dim, FCNN, dx, self.gen, False, depth_per_layer)]
        self.prior_dist = StudentsT(dx, 8., generator)
        self.flow = NormalizingFlowModel_cond(self.prior_dist, real_nvps, self.gen.device)

    def sample(self, prev_state, **data):
        return self.flow.sample((prev_state.size(0), prev_state.size(1)), prev_state)

    def log_density(self, state, prev_state, **data):
        x = self.flow.log_density(state, prev_state)
        return x


class proposal_model(pydpf.Module):
    def __init__(self, dx, generator):
        super().__init__()
        self.dx = dx
        self.gen = generator
        layers = []
        in_dims = [1, 16, 16, 16]
        for di in range(len(in_dims) - 1):
            layers += [{"type": "conv", "in_channels": in_dims[di], "out_channels": in_dims[di+1], "kernel_size": 7, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "relu"},]
        layers += [{"type": "conv", "in_channels": in_dims[-1], "out_channels": 2*dx, "kernel_size": 7, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "id"}]
        self.conv = ConvEncoder(layers, self.gen.device)
        self.dist = pydpf.StandardGaussian(dx, generator)

    def forward(self, n_particles, observation, series_metadata, **data):
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        mean_sd = self.conv(observation)
        log_sd = torch.nn.functional.tanh(mean_sd[:, :, None, self.dx:])
        sd = torch.exp(log_sd)
        mean = mean_sd[:, :, None, :self.dx]
        state = sample * sd + mean
        #stl
        sample_stl = (state - mean.detach()) / sd.detach()
        return state, (self.dist.log_density(sample_stl) - torch.sum(log_sd.detach(), dim=-1))


class dmm_proposal_model(pydpf.Module):
    def __init__(self, dx, generator):
        super().__init__()
        device = generator.device
        self.dist = pydpf.StandardGaussian(dx, generator)
        self.dx = dx
        self.net = torch.nn.LSTM(1, 32, 1, bidirectional=True, device=device)
        self.carry_layer = torch.nn.Linear(self.dx, 32, device=device)
        self.mean_layer = torch.nn.Linear(32, self.dx, device=device)
        self.std_layer = torch.nn.Linear(32, self.dx, device=device)

    def mean(self, hc):
        return self.mean_layer(hc)

    def std(self, hc):
        return torch.sqrt(torch.nn.functional.softplus(self.std_layer(hc)))

    def forward(self, n_particles, observation, series_metadata, **data):
        out, _ = self.net(observation)
        h_sum = (out[..., 32:] + out[..., :32]).unsqueeze(-2)
        hc = h_sum[0] / 2
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        mean = self.mean(hc)
        std = self.std(hc)
        state = [sample[0] * std + mean]
        c_state = state[0]
        log_density = [self.dist.log_density(sample[0]) - torch.sum(torch.log(std), dim=-1)]
        for i in range(1, observation.size(0)):
            hc = (h_sum[i] + torch.nn.functional.tanh(self.carry_layer(c_state)))/3
            mean = self.mean(hc)
            std = self.std(hc)
            state.append(sample[i] * std + mean)
            log_density.append(self.dist.log_density(sample[i]) - torch.sum(torch.log(std), dim=-1))
        return torch.stack(state), torch.stack(log_density)

