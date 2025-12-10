from functools import cached_property

import einops
import pydpf
import torch
from torch import dtype


from experiments.SPX.main import device
from models.generic_nets.FCNN import FCNN#, ResNet
from models.generic_nets.Normalizing_flow import RealNVP_cond, NormalizingFlowModel_cond, very_simple_cond
from models.generic_nets.conv import ConvEncoder
import normflows as nf
import numpy as np


class NN_Switching(pydpf.Module):

    def __init__(self, n_models, recurrent_length, dyn, device):
        super().__init__()
        self.device = device
        self.r_length = recurrent_length
        self.n_models = n_models
        self.forget = torch.nn.Sequential(torch.nn.Linear(n_models, recurrent_length), torch.nn.Sigmoid())
        self.self_forget = torch.nn.Sequential(torch.nn.Linear(recurrent_length, recurrent_length), torch.nn.Sigmoid())
        self.scale = torch.nn.Sequential(torch.nn.Linear(n_models, recurrent_length), torch.nn.Sigmoid())
        self.to_reccurrent = torch.nn.Sequential(torch.nn.Linear(n_models, recurrent_length), torch.nn.Tanh())
        self.output_layer = torch.nn.Sequential(torch.nn.Linear(recurrent_length, recurrent_length), torch.nn.Tanh(), torch.nn.Linear(recurrent_length, n_models))
        self.probs = torch.ones(n_models) / n_models
        self.dyn = dyn

    def init_state(self, batches, n_samples):
        i_models = torch.multinomial(self.probs, batches * n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        if self.r_length > 0:
            return torch.concat((i_models, torch.zeros((batches, n_samples, self.r_length), device=self.device)), dim=2)
        else:
            return i_models

    def forward(self, x_t_1, t):
        old_model = x_t_1[:, :, 0].to(int).unsqueeze(2)
        one_hot = torch.zeros((old_model.size(0), old_model.size(1), self.n_models), device=self.device)
        one_hot = torch.scatter(one_hot, 2, old_model, 1)
        old_recurrent = x_t_1[:, :, 1:]
        c = old_recurrent * self.self_forget(old_recurrent)
        c *= self.forget(one_hot)
        c += self.scale(one_hot) * self.to_reccurrent(one_hot)
        if self.dyn == 'Boot':
            probs = torch.abs(self.output_layer(c))
            probs = probs / torch.sum(probs, dim=2, keepdim=True)
            return torch.concat((torch.multinomial(probs.reshape(-1, self.n_models), 1, True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
        return torch.concat((torch.multinomial(self.probs, x_t_1.size(0) * x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)

    def get_log_probs(self, x_t, x_t_1):
        models = x_t[:, :, 0].to(int)
        probs = torch.abs(self.output_layer(x_t[:, :, 1:])) + 1e-7
        probs = probs / torch.sum(probs, dim=2, keepdim=True)
        log_probs = pydpf.batched_select(probs.reshape(-1, self.n_models), models.flatten()).reshape(x_t.size(0), x_t.size(1))
        return torch.log(log_probs + 1e-7)

class StudentsT(pydpf.distributions.Distribution):
    def __init__(self, dim, df, generator, loc=0., scale = 1.):
        super().__init__(generator)
        self.df = df
        self.dist = torch.distributions.StudentT(torch.tensor(df, device = device), torch.tensor(loc, device = device), torch.tensor(scale, device = device))
        self.dim = dim

    def sample(self, sample_size):
        return self.dist.sample((*sample_size, self.dim))

    def log_density(self, sample):
        return self.dist.log_prob(sample).sum(dim=-1)



class CustomUniform(nf.distributions.BaseDistribution):
    """
    Multivariate uniform distribution
    """

    def __init__(self, shape, low=-1.0, high=1.0, device=torch.device('cpu')):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          low: Lower bound of uniform distribution
          high: Upper bound of uniform distribution
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.d = np.prod(shape)
        self.log_prob_val = -self.d * np.log(high - low)
        self.low = torch.tensor(low, device=device)
        self.high = torch.tensor(high, device=device)

        self.device = device

    def forward(self, num_samples=1, context=None):
        eps = torch.rand(
            (num_samples,) + self.shape, device=self.device
        )
        z = self.low + (self.high - self.low) * eps
        log_p = self.log_prob_val * torch.ones(num_samples, device=self.device)
        return z, log_p

    def log_prob(self, z, context=None):
        log_p = self.log_prob_val * torch.ones(z.shape[0], device=z.device)
        out_range = torch.logical_or(z < self.low, z > self.high)
        ind_inf = torch.any(torch.reshape(out_range, (z.shape[0], -1)), dim=-1)
        log_p[ind_inf] = -np.inf
        return log_p


class observation_model(pydpf.Module):
    def __init__(self, dx, layers, hidden_dim, generator):
        super().__init__()
        self.dx = dx
        self.net = FCNN(dx, 1, hidden_dim, activation_function="tanh", output_function="id", n_hidden_layers=layers - 1, device=generator.device)
        self.dist = pydpf.StandardGaussian(1, generator)


    def sample(self, state, **data):
        mean_sd = self.net(state)
        return mean_sd

    def score(self, state, observation, **data):
        mean_sd = self.net(state)
        standardised = (observation.unsqueeze(1) - mean_sd)
        return self.dist.log_density(standardised)


class observation_model2(pydpf.Module):
    def __init__(self, dx, generator, n_layers, hidden_dim, n_kernels):
        super().__init__()
        self.gen = generator
        self.kernel = pydpf.MultivariateGaussian(torch.tensor([0.], device = generator.device), torch.eye(1, device = generator.device), generator=generator)
        self.dist = pydpf.KernelMixture(self.kernel, generator, pydpf.MultinomialResampler(generator))
        self.net = FCNN(dx, n_kernels*3, hidden_dim, activation_function="tanh", output_function="id", n_hidden_layers=n_layers, device=generator.device)
        self.temperature = torch.nn.Parameter(torch.tensor(5., device=generator.device), requires_grad=True)

    def sample(self, state, **data):
        dist_info = self.net(state)
        locs = dist_info[..., :dist_info.size(-1)//3]
        sds = dist_info[..., dist_info.size(-1)//3:(2*dist_info.size(-1))//3]
        weights = dist_info[..., (2*dist_info.size(-1))//3:] / self.temperature
        weights = torch.softmax(weights, dim=-1)
        idx = torch.multinomial(weights.flatten(0,1), 1, True, generator=self.gen).reshape(weights.size(0), weights.size(1), -1)
        chosen_locs = pydpf.batched_select(locs, idx)
        chosen_sds = pydpf.batched_select(sds, idx)
        sample = self.dist.kernel.sample((state.size(0), state.size(1)))
        return sample * torch.exp(chosen_sds) + chosen_locs

    def score(self, observation, state, **data):
        dist_info = self.net(state)
        observation = observation.unsqueeze(1)
        locs = dist_info[..., :dist_info.size(-1) // 3]
        log_sds = dist_info[..., dist_info.size(-1) // 3:(2 * dist_info.size(-1)) // 3]
        weights = dist_info[..., (2 * dist_info.size(-1)) // 3:] / self.temperature
        weights = torch.log_softmax(weights, dim=-1)
        standardised = (observation - locs)/torch.exp(log_sds)
        log_density = self.kernel.log_density(standardised.unsqueeze(-1)) - log_sds
        return torch.logsumexp(log_density + weights, dim=-1)

class observation_model3(pydpf.Module):
    def __init__(self, dx, generator, n_layers, hidden_dim, n_kernels):
        super().__init__()
        self.gen = generator
        self.kernel = pydpf.MultivariateGaussian(torch.tensor([0.], device = generator.device), torch.nn.Parameter(torch.eye(1, device = generator.device)/3, requires_grad=False), generator=generator)
        self.dist = pydpf.KernelMixture(self.kernel, generator, pydpf.MultinomialResampler(generator))
        self.net = FCNN(dx, n_kernels*2, hidden_dim, activation_function="relu", output_function="id", n_hidden_layers=n_layers, device=generator.device)
        self.temperature = torch.nn.Parameter(torch.tensor(5., device=generator.device), requires_grad=True)

    def sample(self, state, **data):
        dist_info = self.net(state)
        locs = dist_info[..., :dist_info.size(-1)//2]
        weights = dist_info[..., dist_info.size(-1)//2:] / self.temperature
        weights = torch.softmax(weights, dim=-1)
        idx = torch.multinomial(weights.flatten(0,1), 1, True, generator=self.gen).reshape(weights.size(0), weights.size(1), -1)
        chosen_locs = pydpf.batched_select(locs, idx)
        sample = self.dist.kernel.sample((state.size(0), state.size(1)))
        return sample + chosen_locs

    def score(self, observation, state, **data):
        dist_info = self.net(state)
        locs = dist_info[..., :dist_info.size(-1) // 2].unsqueeze(-1)
        weights = dist_info[..., dist_info.size(-1) // 2:] / self.temperature
        weights = torch.log_softmax(weights, dim=-1)
        return self.dist.log_density(observation.unsqueeze(1), locs, weights)


class observation_model4(pydpf.Module):
    def __init__(self, dx, generator, n_layers, depth_per_layer, hidden_dim):
        super().__init__()
        self.dx = dx
        self.gen = generator
        simple_flow = []
        for i in range(n_layers):
            simple_flow += [very_simple_cond(1, hidden_dim, FCNN, dx, self.gen, depth_per_layer)]

        cov = torch.eye(1, device=generator.device)
        mean = torch.zeros(1, device=generator.device)
        self.prior_dist = pydpf.MultivariateGaussian(mean, cov, diagonal_cov=True, generator=self.gen)
        self.flow = NormalizingFlowModel_cond(self.prior_dist, simple_flow, self.gen.device)

    def sample(self, state, **data):
        return self.flow.sample((state.size(0), state.size(1)), state)

    def score(self, state, observation, **data):
        return self.flow.log_density(einops.repeat(observation, "b d -> b p d", p = state.size(1)), state)

class observation_model5(pydpf.Module):
    def __init__(self, dx, generator, n_layers, depth_per_layer, hidden_dim):
        super().__init__()
        self.dx = dx
        self.gen = generator
        simple_flow = []
        for i in range(n_layers):
            simple_flow += [very_simple_cond(1, hidden_dim, FCNN, dx, self.gen, depth_per_layer)]

        cov = torch.eye(1, device=generator.device)
        mean = torch.zeros(1, device=generator.device)
        self.prior_dist = pydpf.MultivariateGaussian(mean, cov, diagonal_cov=True, generator=self.gen)
        self.flow = NormalizingFlowModel_cond(self.prior_dist, simple_flow, self.gen.device)

    def sample(self, state, **data):
        return self.flow.sample((state.size(0), state.size(1)), state)

    def score(self, state, observation, **data):
        return self.flow.log_density(einops.repeat(observation, "b d -> b p d", p = state.size(1)), state)

class observation_model6(pydpf.Module):
    def __init__(self, dx, generator, depth, hidden_dim):
        super().__init__()
        self.dx = dx
        self.gen = generator
        self.dist = FCNN(dx, 1, hidden_dim, activation_function="tanh", output_function="id", n_hidden_layers=depth, device=generator.device)


    def sample(self, state, **data):
        return self.dist(state)

    def score(self, state, observation, **data):
        return observation.unsqueeze(1) - self.dist(state)


class prior_model(pydpf.Module):
    def __new__(cls, dx:int, generator, use_vix= False, hidden_dim=None, depth=None):
        device = generator.device
        #prior_mean = torch.nn.Parameter(torch.zeros(dx, device=generator.device), requires_grad=True)
        #prior_cov =  torch.nn.Parameter(torch.eye(dx, device=device))
        if not use_vix:
            return StudentsT(dx, 3., generator=generator)
        return super().__new__(cls)

    def __init__(self, dx, generator,  use_vix = False, hidden_dim=None, depth=None):
        super().__init__()
        self.dx = dx
        self.dist = StudentsT(dx, 5., generator=generator)
        self.net = FCNN(1, dx, hidden_dim, activation_function="relu", output_function="id", n_hidden_layers=depth - 1, device=generator.device)

    def sample(self,  batch_size, n_particles, series_metadata, **data):
        return self.dist.sample((batch_size, n_particles)) + self.net(series_metadata).unsqueeze(1)

    def log_density(self, state, series_metadata, **data):
        return self.dist.log_density(state - self.net(series_metadata).unsqueeze(1))

class dummy_prior(pydpf.Module):
    def __init__(self, dx, generator):
        super().__init__()
        self.dist = StudentsT(dx, 1., generator, loc=0, scale=3)

    def sample(self, batch_size, n_particles, **data):
        return self.dist.sample((batch_size, n_particles))

    def log_density(self, state, **data):
        return self.dist.log_density(state)


class very_simple_dynamic(pydpf.Module):
    def __new__(cls, dx, generator):
        weight = torch.randn((dx, dx), generator=generator, device=generator.device) * (2/dx)
        bias = torch.randn((dx), generator=generator, device=generator.device) * (2/dx)
        cov = torch.eye(dx, device=generator.device)
        return pydpf.LinearGaussian(torch.nn.Parameter(weight, requires_grad=True), torch.nn.Parameter(bias, requires_grad=True), cholesky_covariance=torch.nn.Parameter(cov, requires_grad=True), constrain_spectral_radius=1., generator=generator)

class simple_switching(pydpf.Module):
    def __init__(self, dx, generator, n_reg):
        super().__init__()
        self.dx = dx
        self.dists = torch.nn.ModuleList()
        for i in range(n_reg):
            weight = torch.nn.Parameter([torch.randn((dx, dx), generator=generator, device=generator.device) * (2/dx)], requires_grad=True)
            cov = torch.nn.Parameter(torch.eye(dx, device=generator.device)/5, requires_grad=True)
            self.dists.append(pydpf.LinearGaussian(weight, torch.zeros(dx, device=generator.device), cholesky_covariance=cov, constrain_spectral_radius=0.99, generator=generator))

    def sample(self, prev_state, **data):
        regimes = prev_state[..., 0].to(int)
        out = torch.empty_like(prev_state)

        for i, dist in enumerate(self.dists):

            mask = (regimes == i)
            if mask.any():
                print('hi')



class dynamic_model(pydpf.Module):
    def __init__(self, dx, generator, n_layers, depth_per_layer, hidden_dim):
        super().__init__()
        self.dx = dx
        self.gen = generator
        real_nvps = []
        for i in range(n_layers):
            real_nvps += [RealNVP_cond(dx, hidden_dim, FCNN, dx, self.gen, False, depth_per_layer)]
        #self.prior_dist = pydpf.MultivariateGaussian(mean, cov, diagonal_cov=True, generator=self.gen)
        self.prior_dist = StudentsT(dx,5., generator)
        self.flow = NormalizingFlowModel_cond(self.prior_dist, real_nvps, self.gen.device)

    def sample(self, prev_state, **data):
        return self.flow.sample((prev_state.size(0), prev_state.size(1)), prev_state)

    def log_density(self, state, prev_state, **data):
        x = self.flow.log_density(state, prev_state)
        return x



class dynamic_model2(pydpf.Module):
    def __init__(self, dx, layers, hidden_dim, generator):
        super().__init__()
        self.dx = dx
        self.net = FCNN(dx, 2*dx, hidden_dim, activation_function="tanh", output_function="id", n_hidden_layers=layers - 1, device=generator.device)
        self.dist = pydpf.StandardGaussian(dx, generator)

    def sample(self, prev_state, **data):
        mean_sd = self.net(prev_state)
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        return (sample * mean_sd[..., self.dx:] + mean_sd[..., :self.dx] + prev_state)

    def log_density(self, prev_state, state, **data):
        mean_sd = self.net(prev_state)
        # print(mean_sd[0])
        sd = torch.abs(mean_sd[..., self.dx:])
        sd = torch.clamp(sd, 0.01)
        standardised = (state - mean_sd[..., :self.dx] - prev_state) / torch.abs(sd)
        log_sd = torch.sum(torch.log(torch.abs(sd.squeeze()) + 1e-8), dim=-1)
        return self.dist.log_density(torch.clamp(standardised, -5, 5)) - log_sd

class dynamic_model3(pydpf.Module):
    def __init__(self, dx, generator, n_layers, hidden_dim, n_kernels):
        super().__init__()
        self.gen = generator
        self.dx = dx
        self.n_kernels = n_kernels
        self.kernel = pydpf.MultivariateGaussian(torch.tensor([0.]*dx, device = generator.device), torch.nn.Parameter(torch.eye(dx, device = generator.device)/3, requires_grad=False), generator=generator)
        self.dist = pydpf.KernelMixture(self.kernel, generator, pydpf.MultinomialResampler(generator))
        self.net = FCNN(dx, n_kernels*(dx+1), hidden_dim, activation_function="relu", output_function="id", n_hidden_layers=n_layers, device=generator.device)
        self.temperature = torch.nn.Parameter(torch.tensor(5., device=generator.device), requires_grad=True)

    def sample(self, prev_state, **data):
        dist_info = self.net(prev_state)
        locs = dist_info[..., self.n_kernels:]
        locs = einops.rearrange(locs, "b i (j k) -> b i j k", j = self.n_kernels)
        weights = dist_info[..., :self.n_kernels] / self.temperature
        weights = torch.softmax(weights, dim=-1)
        idx = torch.multinomial(weights.flatten(0,1), 1, True, generator=self.gen).reshape(weights.size(0), weights.size(1), -1)
        chosen_locs = pydpf.batched_select(locs, idx)
        sample = self.dist.kernel.sample((prev_state.size(0), prev_state.size(1)))
        return sample + chosen_locs

    def log_density(self, state, prev_state, **data):
        dist_info = self.net(prev_state)
        locs = dist_info[..., self.n_kernels:]
        locs = einops.rearrange(locs, "b i (j k) -> b i j k", j = self.n_kernels)
        weights = dist_info[..., :self.n_kernels] / self.temperature
        weights = torch.log_softmax(weights, dim=-1)
        #print(self.dist.log_density(state, locs, weights))
        return self.dist.log_density(state, locs, weights)


class dynamic_model4(pydpf.Module):
    def __init__(self, dx, layers, hidden_dim, generator):
        super().__init__()
        self.dx = dx
        self.nets = FCNN(dx, 2*dx, hidden_dim, activation_function="tanh", output_function="id", n_hidden_layers=layers - 1, device=generator.device)
        self.dist = StudentsT(dx, generator)

    def sample(self, prev_state, **data):
        mean_sd = self.net(prev_state)
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        return (sample * mean_sd[..., self.dx:] + mean_sd[..., :self.dx] + prev_state)

    def log_density(self, prev_state, state, **data):
        mean_sd = self.net(prev_state)
        # print(mean_sd[0])
        sd = torch.abs(mean_sd[..., self.dx:])
        sd = torch.clamp(sd, 0.01)
        standardised = (state - mean_sd[..., :self.dx] - prev_state) / torch.abs(sd)
        log_sd = torch.sum(torch.log(torch.abs(sd.squeeze()) + 1e-8), dim=-1)
        return self.dist.log_density(torch.clamp(standardised, -5, 5)) - log_sd


class dummy_dynamic(pydpf.Module):
    def __init__(self, dx, generator):
        super().__init__()
        self.dist = StudentsT(dx, 1., generator, loc=0, scale=3)

    def sample(self, prev_state, **data):
        return self.dist.sample((prev_state.size(0), prev_state.size(1)))

    def log_density(self, prev_state, state, **data):
        return self.dist.log_density(state)


class dynamic_model_reg(pydpf.Module):
    def __init__(self, dx, generator, n_layers, depth_per_layer, hidden_dim, n_regimes):
        super().__init__()
        self.dx = dx
        self.gen = generator
        self.n_regimes = n_regimes
        self.log_switching_matrix = torch.full((n_regimes, n_regimes), torch.log(torch.tensor(0.2/n_regimes, device = generator.device)), device = generator.device)
        self.log_switching_matrix[range(n_regimes), range(n_regimes)] = torch.log(torch.tensor(0.8/n_regimes, device = generator.device))
        real_nvps = []
        for i in range(n_layers):
            real_nvps += [RealNVP_cond(dx, hidden_dim, FCNN, dx, self.gen, True, depth_per_layer)]
        #self.prior_dist = pydpf.MultivariateGaussian(mean, cov, diagonal_cov=True, generator=self.gen)
        self.prior_dist = StudentsT(dx,3., generator)
        self.flow = NormalizingFlowModel_cond(self.prior_dist, real_nvps, self.gen.device)

    @cached_property
    def switching_matrix(self):
        return torch.nn.functional.softmax(self.log_switching_matrix, dim=-1)

    def sample(self, prev_state, **data):
        return self.flow.sample((prev_state.size(0), prev_state.size(1)), prev_state)

    def log_density(self, state, prev_state, **data):
        return self.flow.log_density(state, prev_state)


class proposal_model(pydpf.Module):
    def __init__(self, dx, generator, use_vix=False):
        super().__init__()
        self.dx = dx
        self.gen = generator
        layers = []
        print(use_vix)
        in_dims = [(3 if use_vix else 1), 2*dx, 4*dx, 4*dx, 4*dx]
        for di in range(len(in_dims) - 1):
            layers += [{"type": "conv", "in_channels": in_dims[di], "out_channels": in_dims[di+1], "kernel_size": 5, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "relu"},
                       {"type": "linear", "in_features": in_dims[di + 1], "out_features": in_dims[di + 1], "bias": True, "device": device, "activation": "relu"}]
        layers += [{"type": "conv", "in_channels": in_dims[-1], "out_channels": 2*dx, "kernel_size": 5, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "id"}]
        self.conv = ConvEncoder(layers, self.gen.device)
        self.dist = pydpf.StandardGaussian(dx, generator)

    def forward(self, n_particles, observation, series_metadata, **data):
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        vix = einops.repeat(series_metadata, "b i -> t b i", t=observation.size(0))
        pos_encoding = einops.repeat(torch.arange(observation.size(0), device = series_metadata.device, dtype=series_metadata.dtype), "t -> t b 1", b=observation.size(1))
        input = torch.cat([observation, vix, pos_encoding/120], dim=-1)
        mean_sd = self.conv(input)
        sd = torch.nn.functional.tanh(mean_sd[:, :, None, self.dx:])
        state = sample * torch.exp(sd) + mean_sd[:, :, None, :self.dx]
        #print(state[0, 0])
        return state, self.dist.log_density(sample) - torch.sum(sd, dim=-1)

class proposal_model2(pydpf.Module):
    def __init__(self, dx, generator):
        super().__init__()
        device = generator.device
        self.dist = pydpf.StandardGaussian(dx, generator)
        self.dx = dx
        self.first_linear = torch.nn.Linear(1, 2*dx, device=device)
        self.lstm = torch.nn.LSTM(1, 2*dx, 1, batch_first=False, bidirectional=False, device=device)

    def forward(self, n_particles, observation, series_metadata, **data):
        initial_hidden_state = self.first_linear(series_metadata)
        zero_tensor = torch.zeros((1, observation.size(1), 2*self.dx), device=observation.device)
        net_out = self.lstm(observation, (initial_hidden_state.unsqueeze(0), zero_tensor))[0]
        mean_state = net_out[..., self.dx:].unsqueeze(-2)
        sd_state = net_out[..., :self.dx].unsqueeze(-2)
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))

        state = sample * sd_state + mean_state
        return state, self.dist.log_density(sample) - torch.sum(torch.log(torch.abs(sd_state)), dim=-1)