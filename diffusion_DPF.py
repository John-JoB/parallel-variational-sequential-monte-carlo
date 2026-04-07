import torch
import pydpf
from math import log

class DiffusionResamplerModified(pydpf.Module):
    # SDE version, modified from the implementation given in the repository attached to the original paper.
    #Modified slightly to be more stable.

    def __init__(self, alpha, T, n_steps, generator):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.n_steps = n_steps
        self.ts = torch.linspace(0., T, n_steps + 1, device=generator.device)
        self.dist = pydpf.StandardGaussian(1, generator=generator)


    def log_pdf(self, x, mu, sigma):
        norm_x = (x - mu)/sigma
        out = self.dist.log_density(norm_x.unsqueeze(-1))
        return out - torch.log(sigma)

    def forward(self, state, weight, **data):
        a = self.alpha
        n = weight.shape[1]
        ws = torch.exp(weight)
        mu = torch.sum(ws[..., None] * state, dim=1, keepdim=True)
        # stat_vars = jnp.einsum('i,i...->...', ws, (samples - mu) ** 2)
        stat_vars = torch.sum(ws[..., None] * ((state - mu) ** 2), dim=1, keepdim=True)
        if torch.all(stat_vars < 1e-6):
            return state.clone(), weight.clone()
        b2 = -a

        def fwd_coeffs(t):
            semigroup = torch.exp(a * t)
            sig2t = pydpf.multiple_unsqueeze((1 - torch.exp(2 * a * t)), 3)
            return semigroup, sig2t

        def logpdf_trans(x, mts, sig2ts):
            """(...,), (n, ...), (n, ...) -> (n, )"""
            # normalised_state = (state - mts)/sig2ts
            return torch.sum(self.log_pdf(x, mts, sig2ts ** 0.5), dim=-1)

        def s(x, sg, sig2ts):
            """Ensemble score
            (..., ), () -> (..., )
            """
            mts = state * sg
            log_alps = weight + logpdf_trans(x, mts, sig2ts)
            log_alps, _ = pydpf.normalise(log_alps)
            return torch.sum(torch.exp(log_alps)[..., None].unsqueeze(2) * (-(x.unsqueeze(1) - mts.unsqueeze(2)) / sig2ts.unsqueeze(2)), dim=1)

        def f(x, sg, sig2ts):
            return  2 * b2 * s(x, sg, sig2ts)

        def drift(x, sg, sig2ts):
            return -a * x + f(x, sg, sig2ts) - 0.5 * x

        sgs, sig2ts = fwd_coeffs(self.T - self.ts)
        x_o = self.dist.sample((mu.size(0), n, mu.size(2))).squeeze(-1)
        rng = self.dist.sample((self.n_steps, *x.shape)).squeeze(-1)
        dts = self.ts[1:] - self.ts[:-1]
        for i in range(self.n_steps):
            x_o = x_o + drift(x_o, sgs[i], sig2ts[i]) * dts[i] + rng[i] * (dts[i] * b2 * 2) ** 0.5
        x_o = x_o * stat_vars + mu
        return x_o, torch.full_like(weight, -log(n))

class DiffusionResampler(pydpf.Module):
    #SDE version, true to the pseudo-code in the paper.

    def __init__(self, alpha, T, n_steps, generator):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.n_steps = n_steps
        self.ts = torch.linspace(0., T, n_steps + 1, device=generator.device)
        self.dist = pydpf.StandardGaussian(1, generator=generator)


    def log_pdf(self, x, mu, sigma):
        norm_x = (x - mu)/sigma
        out = self.dist.log_density(norm_x.unsqueeze(-1))
        return out - torch.log(sigma)

    def forward(self, state, weight, **data):
        a = self.alpha
        n = weight.shape[1]
        ws = torch.exp(weight)
        mu = torch.sum(ws[..., None] * state, dim=1, keepdim=True)
        stat_vars = torch.sum(ws[..., None] * ((state - mu) ** 2), dim=1, keepdim=True)
        if torch.all(stat_vars < 1e-6):
            return state.clone(), weight.clone()
        b2 = -stat_vars * a

        def fwd_coeffs(t):
            semigroup = torch.exp(a * t)
            sig2t =  pydpf.multiple_unsqueeze((1 - torch.exp(2 * a * t)), 3)*stat_vars.unsqueeze(0)
            return semigroup, sig2t


        def logpdf_trans(x, mts, sig2ts):
            """(...,), (n, ...), (n, ...) -> (n, )"""
            #normalised_state = (state - mts)/sig2ts
            return torch.sum(self.log_pdf(x, mts, sig2ts ** 0.5), dim=-1)


        def s(x, sg, sig2ts):
            """Ensemble score
            (..., ), () -> (..., )
            """
            mts = state * sg + mu * (1 - sg)
            log_alps = weight + logpdf_trans(x, mts, sig2ts)
            log_alps, _ = pydpf.normalise(log_alps)
            return torch.sum(torch.exp(log_alps)[..., None].unsqueeze(2) * (-(x.unsqueeze(1) - mts.unsqueeze(2)) / sig2ts.unsqueeze(2)), dim=1)

        def drift(x, sg, sig2ts):
            return 2 * b2 * s(x, sg, sig2ts)

        sgs, sig2ts = fwd_coeffs(self.T - self.ts)
        x_o = mu + (stat_vars ** 0.5) * self.dist.sample((mu.size(0), n, mu.size(2))).squeeze(-1)
        rng = self.dist.sample((self.n_steps, *x_o.shape)).squeeze(-1)
        dts = self.ts[1:] - self.ts[:-1]
        for i in range(self.n_steps):
            x_o = x_o + drift(x_o, sgs[i], sig2ts[i]) * dts[i] + rng[i] * (dts[i] * b2 * 2) ** 0.5
        return x_o, torch.full_like(weight, -log(n))


class DiffusionDPF(pydpf.ParticleFilter):
    def __init__(self, SSM, resampling_generator, alpha=-1, T=2, n_steps=16, use_modified=False):
        if use_modified:
            res = DiffusionResamplerModified(alpha, T, n_steps, resampling_generator)
        else:
            res = DiffusionResampler(alpha, T, n_steps, resampling_generator)
        super().__init__(res, SSM)


