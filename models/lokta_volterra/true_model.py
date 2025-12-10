import einops
import pydpf
import torch

class TrueDynamicModel(pydpf.Module):
    def __init__(self, alpha = 6., beta = 2., gamma = 6., zeta = 4., sigma = 0.15, step_size = 3/256, device = torch.device('cpu')):
        super().__init__()

        self.alpha = torch.tensor(alpha, device=device)
        self.beta = torch.tensor(beta, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.zeta = torch.tensor(zeta, device=device)
        self.sigma = torch.tensor(sigma, device=device)
        self.sigma_squared = self.sigma**2
        self.step_size = torch.tensor(step_size, device=device)
        self.root_step_size = torch.sqrt(self.step_size)

    def drift(self, prev_state):
        output = torch.empty_like(prev_state)
        output[..., 0] = (-prev_state[..., 1] * self.beta + self.alpha) * prev_state[..., 0]
        output[..., 1] = (prev_state[..., 0] * self.zeta - self.gamma) * prev_state[..., 1]
        return output

    def sample(self, prev_state, **data):
        drift = self.drift(prev_state)
        dw = torch.normal(0, self.root_step_size, drift.shape, device=prev_state.device)
        return prev_state + drift * self.step_size + prev_state * (self.sigma * dw + 0.5 * self.sigma_squared * (dw**2 - self.step_size))



class TrueObservationModel(pydpf.Module):
    def __init__(self):
        super().__init__()

    def poisson_mean(self, state):
        factor = torch.empty_like(state)
        factor[..., 0] = 5 * state[..., 0]
        factor[..., 1] = state[..., 1] * state[..., 0]
        return 5 / (1 + torch.exp(4 - factor))

    def sample(self, state, **data):
        mean = self.poisson_mean(state)
        return torch.poisson(mean)

    def score(self, state, observation, **data):
        mean = self.poisson_mean(state)
        k = observation.unsqueeze(1)
        factorial_term = torch.lgamma(k + 1)
        return (k * torch.log(mean) - mean - factorial_term).sum(dim=-1)

class TruePriorModel(pydpf.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def sample(self, batch_size, n_particles, **data):
        return einops.repeat(torch.tensor([2., 5.], device=self.device), "d -> b p d", b =batch_size, p = n_particles)

    def log_density(self, state, **data):
        return torch.zeros((state.size(0), state.size(1)), device=self.device, dtype=state.dtype)