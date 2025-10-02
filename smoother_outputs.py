import pydpf
from pydpf import Module
import torch
from torch import Tensor
import einops
from warnings import warn
from math import log

class dSMC_ELBO(Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    def forward(self, weight, **empty):
        if self.debug:
            return torch.logsumexp(weight, dim=-1) - log(weight.size(-1))
        return torch.mean(torch.logsumexp(weight[0], dim=-1) - log(weight.size(-1)), dim=-1)


class VAE_ELBO(Module):
    def __init__(self):
        super().__init__()

    def forward(self, kernel, initial_likelihood, **empty):
        n_particles = kernel.size(1)
        return torch.mean(torch.sum(kernel, dim=(-1,-2))/(n_particles**2) + torch.sum(initial_likelihood, dim=(-2))/n_particles, dim=-1)


class MarginalSmoothingMean(Module):
    def __init__(self, function = lambda state, **data:state):
        super().__init__()
        self.function = function

    def forward(self, weight, **data):
        norm_weight, _ = pydpf.normalise(weight)
        return torch.sum(self.function(**data)*torch.exp(weight).unsqueeze(-1), dim=(-2))

class MSE(Module):
    def __init__(self, function = lambda state, **data: state):
        super().__init__()
        self.marginal_expec = MarginalSmoothingMean(function)

    def forward(self, weight, ground_truth, **data):
        est = self.marginal_expec(weight=weight, ground_truth=ground_truth **data)
        return torch.mean(torch.sum((est - ground_truth)**2, dim=-1), dim =-1)
