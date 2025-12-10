import pydpf
import torch
from pydpf import cached_property

import models.linear_gaussian.true_model as tm
from parallel_kalman import ParallelKalmanFilter, ParallelKalmanSmoother
from models.generic_nets.conv import ConvEncoder
from models.generic_nets.FCNN import FCNN
import einops

class GaussianDynamic(pydpf.Module):
    def __new__(cls, dx:int, generator):
        device = generator.device
        dynamic_matrix = torch.nn.Parameter(torch.randn((dx, dx), device=device, generator=generator)/5, requires_grad=True)
        dynamic_offset = torch.nn.Parameter(torch.randn(dx, device=device, generator=generator)/5)
        dynamic_cov = torch.nn.Parameter(torch.eye(dx, device=device))
        return pydpf.LinearGaussian(weight=dynamic_matrix, bias=dynamic_offset, cholesky_covariance=dynamic_cov, generator=generator, constrain_spectral_radius=0.98, diagonal_cov=True)

class GaussianObservation(pydpf.Module):
    def __new__(cls, dx:int, dy:int, generator):
        device = generator.device
        observation_matrix = torch.nn.Parameter(torch.randn((dy, dx), device=generator.device, generator=generator) / 5, requires_grad=True)
        observation_offset = torch.nn.Parameter(torch.randn(dx, device=device, generator=generator) / 5)
        observation_cov = torch.nn.Parameter(torch.eye(dy, device=device))
        return pydpf.LinearGaussian(weight=observation_matrix, bias=observation_offset, cholesky_covariance=observation_cov, generator=generator, diagonal_cov=True)

class GaussianPrior(pydpf.Module):
    def __new__(cls, dx:int, generator):
        device = generator.device
        prior_mean = torch.nn.Parameter(torch.randn(dx, device=device, generator=generator) / 5)
        prior_cov =  torch.nn.Parameter(torch.eye(dx, device=device))
        return pydpf.MultivariateGaussian(prior_mean, prior_cov, generator=generator)

class KalmanProposal(pydpf.Module):
    def __init__(self, dx:int, dy:int, generator, true_model = False, use_smoother = True):
        super().__init__()
        device = generator.device
        if true_model:
            prior = tm.GaussianPrior(dx, generator)
            obs = tm.GaussianObservation(dx, dy, generator)
            dyn = tm.GaussianDynamic(dx, generator)
        else:
            prior = GaussianPrior(dx, generator)
            obs = GaussianObservation(dx, dy, generator)
            dyn = GaussianDynamic(dx, generator)
        if use_smoother:
            self.kalman = ParallelKalmanSmoother(prior_model=prior, dynamic_model=dyn, observation_model=obs)
            self.kalman = pydpf.KalmanFilter(prior, dyn, obs)
        else:
            self.kalman = ParallelKalmanFilter(prior_model=prior, dynamic_model=dyn, observation_model=obs)
        self.dist = pydpf.MultivariateGaussian(torch.zeros(dx, device=device), torch.eye(dx, device = device), diagonal_cov=True, generator=generator)
        self.log_2pi = torch.log(torch.tensor(2 * torch.pi, device = device)) * dx

    def forward(self, n_particles, observation, **data):
        time_extent = observation.size(0) - 1
        means, covs, _ = self.kalman(time_extent, observation)
        particles_standard = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        cholesky = torch.linalg.cholesky(covs)
        log_det = torch.sum(torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)), dim=-1, keepdim=True)
        particles = torch.einsum("t b n d, t b d e -> t b n e", particles_standard, cholesky) + means.unsqueeze(-2)
        return particles, -0.5 * ( torch.sum(particles_standard**2, dim=-1) + self.log_2pi ) - log_det


class ConvProposal(pydpf.Module):
    def __init__(self, dx, dy, time_extent, generator):
        super().__init__()
        self.dx = dx
        self.gen = generator
        if time_extent < 12:
            self.net = FCNN(dy*(time_extent+1), dx*(time_extent+1), 128, "tanh", "id", 7, generator.device)
        else:
            self.net = None
        layers = []
        in_dims = [dy, 16, 16, 16]
        for di in range(len(in_dims) - 1):
            layers += [{"type": "conv", "in_channels": in_dims[di], "out_channels": in_dims[di+1], "kernel_size": 7, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "relu"}]#,
                       #{"type": "linear", "in_features": in_dims[di + 1], "out_features": in_dims[di + 1], "bias": True, "device": generator.device, "activation": "relu"}]
        layers += [{"type": "conv", "in_channels": in_dims[-1], "out_channels": dx, "kernel_size": 9, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "id"}]
        self.conv = ConvEncoder(layers, self.gen.device)
        self.dist = pydpf.StandardGaussian(dx, generator)
        self.log_sd = torch.nn.Parameter(torch.log(torch.ones(dx, device=generator.device)/3.), requires_grad=True)

    @cached_property
    def sd(self):
        return self.log_sd.exp()

    @cached_property
    def log_det(self):
        return torch.sum(self.log_sd, dim=0, keepdim=True)

    def forward(self, n_particles, observation, **data):
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        if self.net is None:
            mean_sd = self.conv(observation)
        else:
            mean_sd = self.net(einops.rearrange(observation, "t b d -> b (t d)"))
            mean_sd = einops.rearrange(mean_sd, "b (t d) -> t b d", t = observation.size(0))
        state = sample * self.sd + mean_sd[:, :, None, :]
        #state = sample / 3 + mean_sd[:, :, None, :self]
        #print(state[0, 0])
        return state, self.dist.log_density(sample) - self.log_det

class proposal_model(pydpf.Module):
    def __init__(self, dx, dy, time_extent, generator):
        super().__init__()
        self.dx = dx
        self.gen = generator
        layers = []
        in_dims = [dy, 16, 16, 16]
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