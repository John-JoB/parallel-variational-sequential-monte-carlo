import torch
import pydpf
from math import ceil

from experiments.SPX.main import device
from models.generic_nets.FCNN import FCNN

class RealNVP_cond(pydpf.Module):

    def __init__(self, dim, hidden_dim=8, base_network=FCNN, condition_on_dim=0, generator = torch.default_generator, zero_i = False, depth = 0):
        super().__init__()
        self.dim = dim
        self.condition_on_dim = condition_on_dim
        self.t1 = base_network(dim // 2 + self.condition_on_dim, ceil(dim / 2), hidden_dim, "tanh", "id", depth, generator.device)
        self.t2 = base_network(ceil(dim / 2) + self.condition_on_dim, dim // 2, hidden_dim,  "tanh", "id", depth, generator.device)
        self.s1 = base_network(dim // 2 + self.condition_on_dim, ceil(dim / 2), hidden_dim, "tanh", "tanh", depth, generator.device)
        self.s2 = base_network(ceil(dim / 2) + self.condition_on_dim, dim // 2, hidden_dim,  "tanh", "tanh", depth, generator.device)
        self.generator = generator
        #if zero_i:
         #   self.zero_initialization()

    """
    def zero_initialization(self, std=0.01):
        for layer in self.t1.net:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                layer.bias.data.fill_(0)
        for layer in self.t2.net:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                layer.bias.data.fill_(0)
    """

    def forward(self, x, condition_on):
        lower, upper = x[..., :self.dim // 2], x[..., self.dim // 2:]
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        s1_transformed = self.s1(lower_extended)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        s2_transformed = self.s2(upper_extended)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=-1)
        log_det = s2_transformed.sum(dim=-1) + s1_transformed.sum(dim=-1)
        return z, log_det

    def inverse(self, z, condition_on):
        lower, upper = z[..., :self.dim // 2], z[..., self.dim // 2:]

        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        s2_transformed = self.s2(upper_extended)
        lower = (lower - t2_transformed)/torch.exp(s2_transformed)
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        s1_transformed = self.s1(lower_extended)
        upper = (upper - t1_transformed)/torch.exp(s1_transformed)
        x = torch.cat([lower, upper], dim=-1)
        log_det = -(s1_transformed.sum(dim=-1) + s2_transformed.sum(dim=-1))
        return x, log_det




class NormalizingFlowModel_cond(pydpf.Module):

    def __init__(self, prior, flows, device='cuda:0'):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = torch.nn.ModuleList(flows).to(self.device)

    def forward(self, x, condition_on):
        b, m, d = x.shape
        log_det = torch.zeros((b, m)).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition_on)
            log_det += ld
        return x, log_det

    def inverse(self, z, condition_on):
        b, m, d = z.shape
        log_det = torch.zeros((b, m)).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, condition_on)
            log_det += ld
        x = z
        return x, log_det

    def log_density(self, x, condition_on):
        z, log_det = self.forward(x, condition_on)
        prior_prob = self.prior.log_density(z)
        return prior_prob + log_det

    def sample(self, sample_size, condition_on):
        z = self.prior.sample(sample_size)
        return self.inverse(z, condition_on)[0]


class very_simple_cond(pydpf.Module):
    def __init__(self, dim, hidden_dim=8, base_network=FCNN, condition_on_dim=0, generator = torch.default_generator, depth = 0):
        super().__init__()
        self.dim = dim
        self.condition_on_dim = condition_on_dim
        self.exp_net = base_network(self.condition_on_dim, dim , hidden_dim, "tanh", "sigmoid", depth, generator.device)
        self.translate_net = base_network(self.condition_on_dim, dim, hidden_dim,  "tanh", "sigmoid", depth, generator.device)
        self.prod_net = base_network(self.condition_on_dim, dim , hidden_dim, "tanh", "tanh", depth, generator.device)
        self.exp_factor = torch.nn.Parameter(torch.tensor(2., device=generator.device))
        self.prod_factor = torch.nn.Parameter(torch.tensor(1., device=generator.device))

    def forward(self, x, condition_on):
        exp = self.exp_net(condition_on)
        translate = self.translate_net(condition_on)
        prod = self.prod_net(condition_on)
        z_not_translate = torch.exp(x * self.exp_factor * torch.log(exp))  * prod * self.prod_factor
        z = z_not_translate - translate
        return z, (torch.log(torch.abs(z_not_translate*self.exp_factor * torch.log(exp)))).squeeze(-1)

    def inverse(self, z, condition_on):
        exp = self.exp_net(condition_on)
        translate = self.translate_net(condition_on)
        prod = self.prod_net(condition_on)
        z_not_translate = z + translate
        x = (torch.log(z_not_translate)  - torch.log(prod * self.prod_factor)) / (self.exp_factor * exp)
        return x, (-torch.log(z_not_translate) - torch.log(self.exp_factor * torch.log(exp))).squeeze(-1)