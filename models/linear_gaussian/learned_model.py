import pydpf
import torch
import models.linear_gaussian.true_model as tm

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
    def __init__(self, dx:int, dy:int, generator, true_model = False):
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
        self.kalman = pydpf.KalmanFilter(prior_model=prior, dynamic_model=dyn, observation_model=obs)
        self.dist = pydpf.MultivariateGaussian(torch.zeros(dx, device=device), torch.eye(dx, device = device), diagonal_cov=True, generator=generator)
        self.log_2pi = torch.log(torch.tensor(2 * torch.pi, device = device))

    def forward(self, n_particles, observation, **data):
        time_extent = observation.size(0) - 1
        means, covs, _ = self.kalman(time_extent, observation)
        particles_standard = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        cholesky = torch.linalg.cholesky(covs)
        log_det = torch.sum(torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)), dim=-1, keepdim=True)
        particles = torch.einsum("t b n d, t b d e -> t b n e", particles_standard, cholesky) + means.unsqueeze(-2)
        return particles, -0.5 * ( torch.sum(particles_standard**2, dim=-1) + self.log_2pi ) - log_det

