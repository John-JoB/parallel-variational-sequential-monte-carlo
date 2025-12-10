import pydpf
import torch
from math import ceil

from experiments.linear_gaussian.test_kalman_filters import time_extent


def L1ReconLoss(observation, recon_obs):
    return torch.sum(torch.abs(observation - recon_obs), dim=(0, 2))

class TCVAE(pydpf.Module):
    def __init__(self, encoder, decoder, prior, dx, gen, beta):
        super().__init__()
        self.dx = dx
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.dist = pydpf.StandardGaussian(dx, gen)
        self.gauss_factor = -0.5 * (1 + torch.log(2. * torch.tensor(torch.pi, device=gen.device)))
        self.beta = torch.tensor(beta, device=gen.device)


    @property
    def SSM(self):
        return self

    class apply_beta(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor, beta):
            ctx.save_for_backward(beta)
            return tensor

        @staticmethod
        def backward(ctx, grad):
            beta, = ctx.saved_tensors
            return grad * beta, None

    def forward(self, time_extent, observation, **data):
        observation = observation[:time_extent+1]
        mu, log_var = self.encoder(observation)
        std = torch.exp(0.5*log_var)
        latent_state = self.dist.sample((observation.size(0), observation.size(1)))
        latent_state = latent_state * std + mu
        recon = self.decoder(latent_state)
        l1_loss = L1ReconLoss(observation, recon)
        prior_log_density = self.prior.log_density(latent_state)
        kld_loss = self.gauss_factor - 0.5*torch.sum(log_var, dim=(0, 2)) - prior_log_density
        kld_loss = self.apply_beta.apply(kld_loss, self.beta)
        return l1_loss, kld_loss

    def generate(self, n_samples, time_extent, **data):
        n_series = ceil((time_extent+1)/self.prior.length)
        latent_state = []
        for _ in range(n_series):
            latent_state.append(self.prior.sample((n_samples,)))
        latent_state = torch.cat(latent_state, dim=0)
        latent_state = latent_state[:time_extent + 1]
        recon_obs = self.decoder(latent_state)
        return recon_obs