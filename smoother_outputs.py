import pydpf
from pydpf import Module
import torch
from torch import Tensor

class dSMC_ELBO(Module):
    need_weight = False
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    def forward(self, elbo, **empty):
        return elbo


class dSMC_ELBO2(Module):
    need_weight = False
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    def forward(self, elbo, **empty):
        return elbo

class State(Module):
    need_weight = False
    def __init__(self, **data):
        super().__init__()

    def forward(self, state, **data):
        return state

class Weight(Module):
    need_weight = True
    def __init__(self, **data):
        super().__init__()

    def forward(self, weight, **data):
        return weight

class VAE_ELBO(Module):
    need_weight = False
    def __init__(self):
        super().__init__()

    def forward(self, kernel, initial_likelihood, **empty):
        if kernel.dim() == 4:
            return torch.mean(torch.sum(torch.mean(kernel, dim=-1), dim=0) + initial_likelihood, dim=-1)
        else:
            return torch.mean(torch.sum(kernel, dim=0) + initial_likelihood, dim=-1)


class MarginalSmoothingMean(Module):
    need_weight = True
    def __init__(self, function = lambda state, **data:state):
        super().__init__()
        self.function = function

    def forward(self, weight, **data):
        return torch.sum(self.function(**data)*torch.exp(weight).unsqueeze(-1), dim=(-2))
        #return torch.mean(self.function(**data), dim=(-2))

class MSE(Module):
    need_weight = True
    def __init__(self, function = lambda state, **data: state):
        super().__init__()
        self.marginal_expec = MarginalSmoothingMean(function)

    def forward(self, weight, ground_truth, **data):
        est = self.marginal_expec(weight=weight, ground_truth=ground_truth, **data)
        return torch.sum((est - ground_truth)**2, dim=-1)

class KernelLogLikelihood(Module):
    """Get the negative log data likelihood per-timestep under a kernel density estimator.
        This function applies a kernel density estimator over the particles and calculates the log likelihood of the ground truth given the KDE.

        Parameters
        ----------
        kernel: KernelMixture
            The kernel density estimator.
        """
    need_weight = True
    def __init__(self, kernel: pydpf.KernelMixture):

        super().__init__()
        self.KDE = kernel

    def forward(self, *, state: Tensor, weight: Tensor, ground_truth, **kwargs):
        """Get the negative log data likelihood factor under the a KDE and given a time-step"""
        return self.KDE.log_density(ground_truth, state, weight)