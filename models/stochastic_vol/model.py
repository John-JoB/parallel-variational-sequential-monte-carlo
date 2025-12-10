import pydpf
import torch
from models.generic_nets.FCNN import FCNN
import einops
from models.generic_nets.conv import ConvEncoder
from models.linear_gaussian.learned_model import GaussianPrior, GaussianObservation, GaussianDynamic
from parallel_kalman import ParallelKalmanSmoother
from math import log

class StochasticVolatility_Prior(pydpf.Module):
    @pydpf.cached_property
    def sd(self):
        return torch.sqrt(self.sigma.squeeze()**2/(1-self.alpha.squeeze()**2))

    @pydpf.cached_property
    def sigma(self):
        if not self.learned_sigma:
            return self.sigma_
        log_sigma = torch.nn.functional.sigmoid(self.sigma_)
        return torch.exp(-4 * log_sigma) * 5

    @pydpf.cached_property
    def alpha(self):
        if not self.learned_alpha:
            return self.alpha_
        log_alpha = torch.nn.functional.sigmoid(self.alpha_)
        return torch.exp(-4 * log_alpha)

    def __init__(self, sigma, alpha, generator):
        super().__init__()
        if isinstance(sigma, torch.Tensor):
            self.sigma_ = sigma
            self.learned_sigma = True
        else:
            self.sigma_ = torch.tensor(sigma, device=generator.device)
            self.learned_sigma = False
        if isinstance(alpha, torch.Tensor):
            self.alpha_ = alpha
            self.learned_alpha = True
        else:
            self.alpha_ = torch.tensor(alpha, device=generator.device)
            self.learned_alpha = False
        i1 = torch.ones((1, 1), device=generator.device)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(1, device=generator.device), cholesky_covariance=i1, generator=generator)

    def sample(self, batch_size: int, n_particles: int, **data):
        return self.dist.sample(sample_size=(batch_size, n_particles)) * self.sd

    def log_density(self, state, **data):
        return self.dist.log_density(sample=state/self.sd) - torch.log(self.sd)

class StochasticVolatility_Dynamic(pydpf.Module):
    def __init__(self, sigma, alpha, generator):
        super().__init__()
        if isinstance(sigma, torch.Tensor):
            self.sigma_ = sigma
            self.learned_sigma = True
        else:
            self.sigma_ = torch.tensor(sigma, device=generator.device)
            self.learned_sigma = False
        if isinstance(alpha, torch.Tensor):
            self.alpha_ = alpha
            self.learned_alpha = True
        else:
            self.alpha_ = torch.tensor(alpha, device=generator.device)
            self.learned_alpha = False

        self.dist = pydpf.StandardGaussian(1, generator)

    @pydpf.cached_property
    def sigma(self):
        if not self.learned_sigma:
            return self.sigma_
        log_sigma = torch.nn.functional.sigmoid(self.sigma_)
        return torch.exp(-4*log_sigma) * 5

    @pydpf.cached_property
    def log_sigma(self):
        return torch.log(self.sigma)

    @pydpf.cached_property
    def alpha(self):
        if not self.learned_alpha:
            return self.alpha_
        log_alpha = torch.nn.functional.sigmoid(self.alpha_)
        return torch.exp(-4*log_alpha)

    def sample(self, prev_state, **data):
        sample = self.dist.sample(sample_size=(prev_state.size(0), prev_state.size(1)))
        sample = sample * self.sigma + prev_state * self.alpha
        return sample

    def log_density(self, state, prev_state, **data):
        standard_sample = (state - prev_state * self.alpha)/self.sigma
        return self.dist.log_density(standard_sample) - self.log_sigma


class StochasticVolatility_Observation(pydpf.Module):

    @pydpf.cached_property
    def beta(self):
        if not self.learned_beta:
            return self.beta_
        log_beta = torch.nn.functional.sigmoid(self.beta_)
        return torch.exp(-log_beta*3/2)

    def __init__(self, beta, generator):
        super().__init__()
        if isinstance(beta, torch.Tensor):
            self.beta_ = beta
            self.learned_beta = True
        else:
            self.beta_ = torch.tensor(beta, device=generator.device)
            self.learned_beta = False
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(1, device=generator.device), cholesky_covariance=torch.ones((1, 1), device=generator.device), generator=generator)

    def sample(self, state, **data):
        sample = self.dist.sample((state.size(0), state.size(1)))
        return sample * torch.exp(state/2) * self.beta

    def score(self, observation, state, **data):
        sd = torch.exp(state/2) * self.beta
        #With this simple SV model there's not a convenient way to disallow very small volatilities, clip them to avoid numerical errors.
        sd = torch.clip(sd, 1e-6)
        return self.dist.log_density(observation.unsqueeze(1)/sd) - torch.log(sd).squeeze()

class ConvProposal(pydpf.Module):
    def __init__(self, dx, dy, hidden_dim, time_extent, generator):
        super().__init__()
        self.dx = dx
        self.gen = generator
        if time_extent < 12:
            self.net = FCNN(dy*(time_extent+1), dx*(time_extent+1), 128, "tanh", "id", 7, generator.device)
        else:
            self.net = None
        layers = []
        in_dims = [dy, hidden_dim, hidden_dim]
        for di in range(len(in_dims) - 1):
            layers += [{"type": "conv", "in_channels": in_dims[di], "out_channels": in_dims[di+1], "kernel_size": 9, "kernel_offset":0, "left_input_size": 9, "right_input_size": 9, "activation": "leaky_relu"},
                       {"type": "dropout", "p": 0.3},
                       ]
        layers += [{"type": "linear", "in_features": in_dims[-1], "out_features": 2*dx, "bias": True, "device": generator.device, "activation": "id"}]
        self.conv = ConvEncoder(layers, self.gen.device)
        self.dist = pydpf.StandardGaussian(dx, generator)
        #self.log_sd = torch.nn.Parameter(torch.log(torch.ones(dx, device=generator.device)/3.), requires_grad=True)

    @pydpf.cached_property
    def sd(self):
        return self.log_sd.exp()

    @pydpf.cached_property
    def log_det(self):
        return torch.sum(self.log_sd, dim=0, keepdim=True)

    def forward(self, n_particles, observation, **data):
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        if self.net is None:
            mean_sd = self.conv(observation)
        else:
            mean_sd = self.net(einops.rearrange(observation, "t b d -> b (t d)"))
            mean_sd = einops.rearrange(mean_sd, "b (t d) -> t b d", t = observation.size(0))
        sd = torch.exp(torch.tanh(mean_sd[:, :, None, self.dx:]))
        state = sample * sd + mean_sd[:, :, None, :self.dx]
        #print(sd)
        #state = sample / 3 + mean_sd[:, :, None, :self]
        #print(state[0, 0])
        return state, self.dist.log_density(sample) - torch.sum(torch.tanh(mean_sd[:, :, None, self.dx:]), dim=-1)

class KalmanProposal(pydpf.Module):
    def __init__(self, dx:int, dy:int, generator):
        super().__init__()
        device = generator.device
        prior = GaussianPrior(dx, generator)
        obs = GaussianObservation(dx, dy, generator)
        dyn = GaussianDynamic(dx, generator)
        self.kalman = ParallelKalmanSmoother(prior_model=prior, dynamic_model=dyn, observation_model=obs)
        self.kalman = pydpf.KalmanFilter(prior, dyn, obs)
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

class LSTM_proposal(pydpf.Module):
    def __init__(self, dx, dy, hidden_dim, time_extent, generator):
        super().__init__()
        self.dx = dx
        self.gen = generator
        self.lstm = torch.nn.LSTM(
            input_size=dy,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=False,
            device=generator.device
        )
        self.proj = torch.nn.Linear(2*hidden_dim, dx*2, device=generator.device)
        self.dist = pydpf.StandardGaussian(dx, generator)
        #self.log_sd = torch.nn.Parameter(torch.log(torch.ones(dx, device=generator.device)/3.), requires_grad=True)

    @pydpf.cached_property
    def sd(self):
        return self.log_sd.exp()

    @pydpf.cached_property
    def log_det(self):
        return torch.sum(self.log_sd, dim=0, keepdim=True)

    def forward(self, n_particles, observation, **data):
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        lstm, _ = self.lstm(observation)
        mean_sd = self.proj(lstm)

        #state = sample * self.sd + mean_sd[:, :, None, :]
        #state = sample / 3 + mean_sd[:, :, None, :self]
        #print(state[0, 0])
        sd = torch.exp(torch.tanh(mean_sd[:, :, None, self.dx:]))
        state = sample * sd + mean_sd[:, :, None, :self.dx]
        return state, self.dist.log_density(sample) - torch.sum(torch.tanh(mean_sd[:, :, None, self.dx:]), dim=-1)
        return state, self.dist.log_density(sample) - self.log_det