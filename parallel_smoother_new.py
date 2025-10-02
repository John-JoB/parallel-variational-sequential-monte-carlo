import pydpf
from pydpf import Module
import torch
from torch import Tensor
import einops
from warnings import warn
from math import floor
import torch.backends as tb
from math import log


class ParallelSmoother(Module):
    def __init__(self, proposal, SSM):
        super().__init__()
        self.proposal = proposal
        self.SSM = SSM

    einsum_letters = " a b c d e f g h m n o p q"

    @staticmethod
    def _get_batched_dict(**data):
        batched_data = {}
        letters = " a c d e f g h i j k l m n o p q r s u v w x y z"
        for key, value in data.items():
            if value is None:
                continue
            if key == "series_metadata":
                batched_data[key] = value
                continue

            extra_dims_as_letters = letters[:(value.dim() - 2)*2]
            batched_data[key] = einops.rearrange(value, f"t b{extra_dims_as_letters} -> (t b){extra_dims_as_letters}")
        return batched_data

    class logsumredexp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, left, centre, right):
            max_left = torch.amax(left, -1, keepdim=True)
            max_centre = torch.amax(centre, (-1, -2 ), keepdim=True)
            max_right = torch.amax(right, -2, keepdim=True)
            exp_right = torch.exp(right - max_right)
            exp_centre = torch.exp(centre - max_centre)
            exp_left = torch.exp(left - max_left)
            letters = ParallelSmoother.einsum_letters[:2 * (left.dim() - 2)]
            exp_output = torch.einsum(f"{letters} i j, {letters} j k, {letters} k l -> {letters} i l", exp_left, exp_centre, exp_right)
            ctx.save_for_backward(exp_left, exp_centre, exp_right, exp_output)
            return torch.log(exp_output) + max_left + max_right + max_centre - log(centre.size(-1))*2

        @staticmethod
        def backward(ctx, do):
            exp_left, exp_centre, exp_right, exp_output = ctx.saved_tensors
            letters = ParallelSmoother.einsum_letters[:2 * (exp_left.dim() - 2)]
            grad_scaled = do / exp_output
            #Do a lot of repeated computation to save having a giant tensor in memory.
            #These are all really different reductions over a single {letters}+4d tensor
            grad_left = torch.einsum(f"{letters} i j, {letters} j k, {letters} k l, {letters} i l -> {letters} i j", exp_left, exp_centre, exp_right, grad_scaled )
            grad_centre = torch.einsum(f"{letters} i j, {letters} j k, {letters} k l, {letters} i l -> {letters} j k", exp_left, exp_centre, exp_right, grad_scaled)
            grad_right = torch.einsum(f"{letters} i j, {letters} j k, {letters} k l, {letters} i l -> {letters} k l", exp_left, exp_centre, exp_right, grad_scaled)
            return grad_left, grad_centre, grad_right


    def combine(self, left_ls, right_ls, kernels):
        if left_ls.size(0) != right_ls.size(0):
            output = ParallelSmoother.logsumredexp.apply(left_ls[:-1], kernels, right_ls)
            output = torch.concat([output, left_ls[-1:]], 0)
            return output
        return ParallelSmoother.logsumredexp.apply(left_ls, kernels, right_ls)

    def tree_recurse(self, ls, kernels):
        kernels_1 = kernels[::2]
        left_1 = ls[::2]
        right_1 = ls[1::2]
        combine_1 = self.combine(left_1, right_1, kernels_1)
        kernels_2 = kernels[1::2]
        left_2 = ls[1::2]
        right_2 = ls[2::2]
        combine_2 = torch.concat([ls[0:1], self.combine(left_2, right_2, kernels_2)], dim = 0)
        if combine_1.size(0) != combine_2.size(0):

            new_kernels = [kernels_1, torch.concat([kernels_2, torch.zeros_like(kernels_2[0:1])], dim=0,)]
            new_ls = [combine_2, torch.concat([combine_1, torch.zeros_like(combine_2[0:1])], dim = 0)]
        else:
            new_kernels = [kernels_1, kernels_2]
            new_ls = [combine_2, combine_1]
        return einops.rearrange(new_ls, "p t s b n m -> t (p s) b n m"), einops.rearrange(new_kernels, "p t s b n m -> t (p s) b n m")

    def forward(self, n_particles: int,
                time_extent: int,
                aggregation_function: Module | dict,
                observation: Tensor,
                *,
                gradient_regulariser: torch.autograd.Function | None = None,
                ground_truth: Tensor | None = None,
                control: Tensor | None = None,
                time: Tensor | None = None,
                series_metadata: Tensor | None = None) -> Tensor|dict:

        observation = observation[:time_extent+1]
        if ground_truth is not None:
            ground_truth = ground_truth[:time_extent+1]
        if control is not None:
            control = control[:time_extent+1]
        if time is not None:
            time = time[:time_extent+1]
        state, prop_density = self.proposal(n_particles, observation=observation, control=control, time=time, series_metadata=series_metadata)
        batched_data = ParallelSmoother._get_batched_dict(ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
        state_repeat = einops.repeat(state[1:], 't b n d -> (t b) (n m) d', m = n_particles)
        prev_state_repeat = einops.repeat(state[:-1], 't b n d -> (t b) (m n) d', m = n_particles)
        obs_score = self.SSM.observation_model.score(**batched_data)
        del(batched_data["state"])
        prior_density = self.SSM.prior_model.log_density(state=state[0], observation=observation[0])
        dynamic_density = self.SSM.dynamic_model.log_density(state=state_repeat, prev_state=prev_state_repeat, **batched_data)
        obs_score = einops.rearrange(obs_score, "(t b) n -> t b 1 n", t = time_extent + 1)
        prop_density = einops.rearrange(prop_density, "t b n -> t b 1 n", t = time_extent + 1)
        dynamic_density = einops.rearrange(dynamic_density, "(t b) (m n) -> t b m n", t = time_extent, m = n_particles)
        kernels = obs_score[1:] + dynamic_density - prop_density[1:]
        time_zero_l = obs_score[0].squeeze() + prior_density - prop_density[0].squeeze()
        if time_extent % 2 == 0:
            ls = torch.concat([time_zero_l[None, :, :, None].expand(-1, -1, -1, kernels.size(-1)),  kernels[1::2]], dim=0)
            remaining_kernels = kernels[::2]
        else:
            ls = kernels[::2]
            remaining_kernels = kernels[1::2]
            ls[0] += time_zero_l.unsqueeze(-1)
        ls = ls.unsqueeze(1)
        new_kernels = remaining_kernels.unsqueeze(1)
        while True:
            ls, new_kernels = self.tree_recurse(ls, new_kernels)
            if ls.size(0) == 2:
                break
        useful_ls = ls[:, :floor((time_extent-1) /2)]
        left_facing_ls = torch.logsumexp(useful_ls[0], dim=-2, keepdim=False).unsqueeze(-1)
        right_facing_ls = torch.logsumexp(useful_ls[1], dim=-1, keepdim=False).unsqueeze(-2)
        combined_weights = left_facing_ls + right_facing_ls + remaining_kernels
        weights = einops.rearrange([torch.logsumexp(combined_weights, dim=-1), torch.logsumexp(combined_weights, dim=-2)], "s t b n -> (t s) b n")
        outer_weights = ParallelSmoother.logsumredexp.apply(useful_ls[0, 0], remaining_kernels[0], useful_ls[1, 0]) + 2*log(weights.size(-1))
        weights = torch.concat([torch.logsumexp(outer_weights, dim=-1).unsqueeze(0), weights, torch.logsumexp(outer_weights, dim=-2).unsqueeze(0)], dim=0) - 3*log(weights.size(-1))
        if isinstance(aggregation_function, dict):
            output = {}
            for name, function in aggregation_function.items():
                output[name] = function(weight=weights, kernel = kernels, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
            return output
        return aggregation_function(weight=weights, kernel = kernels, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)


class FCNN(Module):
    def __init__(self, *layers, device):
        super().__init__()
        instantiated = []
        for i in range(len(layers)-1):
            instantiated.append(torch.nn.Linear(layers[i], layers[i+1], device=device))
            if i < len(layers)-2:
                instantiated.append(torch.nn.Tanh())
        self.seq = torch.nn.Sequential(*instantiated)

    def forward(self, x):
        return self.seq(x)


class ConvolutionalProposal(Module):
    def __init__(self, observation_dim, state_dim, kernel_size, generator):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.trailing_time_steps = kernel_size - 1
        self.mean_layer_1 = torch.nn.Conv1d(in_channels=observation_dim, out_channels=state_dim, kernel_size=kernel_size, device=generator.device)
        self.mean_layer_2 = torch.nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=kernel_size, device=generator.device)
        self.mean_network = torch.nn.Sequential(self.mean_layer_1, torch.nn.ReLU(), self.mean_layer_2)
        self.mean_start_network = FCNN(observation_dim*(self.trailing_time_steps*2-1), state_dim*self.trailing_time_steps, state_dim*self.trailing_time_steps, device=generator.device)
        self.mean_end_network = FCNN(observation_dim*(self.trailing_time_steps*2-1), state_dim*self.trailing_time_steps, state_dim*self.trailing_time_steps, device=generator.device)

        self.cov_layer_1 = torch.nn.Conv1d(in_channels=observation_dim, out_channels=state_dim, kernel_size=kernel_size, device=generator.device)
        self.cov_layer_2 = torch.nn.Conv1d(in_channels=state_dim, out_channels=state_dim, kernel_size=kernel_size, device=generator.device)
        self.cov_network = torch.nn.Sequential(self.cov_layer_1, torch.nn.ReLU(), self.cov_layer_2)
        self.cov_start_network = FCNN(observation_dim * (self.trailing_time_steps*2-1), state_dim * self.trailing_time_steps, state_dim * self.trailing_time_steps, device=generator.device)
        self.cov_end_network = FCNN(observation_dim * (self.trailing_time_steps*2-1), state_dim * self.trailing_time_steps, state_dim * self.trailing_time_steps, device=generator.device)
        self.dist = pydpf.MultivariateGaussian(torch.zeros(state_dim, device=generator.device), torch.eye(state_dim, device=generator.device), False, generator=generator)

    def forward(self, n_particles, observation):
        observation = torch.movedim(observation, 0, -1)
        means_main = self.mean_network(observation)
        means_start = self.mean_start_network(observation[:,:,:self.trailing_time_steps*2-1].flatten(start_dim=1)).reshape(-1, means_main.size(1), self.trailing_time_steps)
        means_end = self.mean_end_network(observation[:,:,-(self.trailing_time_steps*2-1):].flatten(start_dim=1)).reshape(-1, means_main.size(1), self.trailing_time_steps)
        cov_main = self.cov_network(observation)
        cov_start = self.cov_start_network(observation[:,:,:self.trailing_time_steps*2-1].flatten(start_dim=1)).reshape(-1, means_main.size(1), self.trailing_time_steps)
        cov_end = self.cov_end_network(observation[:,:,-(self.trailing_time_steps*2-1):].flatten(start_dim=1)).reshape(-1, means_main.size(1), self.trailing_time_steps)
        mean = torch.cat((means_start, means_main, means_end), dim=-1).movedim(-1,0).contiguous()
        cov =  torch.abs(torch.cat((cov_start, cov_main, cov_end), dim=-1).movedim(-1,0)).contiguous()
        sample = self.dist.sample((observation.size(-1), observation.size(0), n_particles))
        dets = torch.sum(torch.log(cov), dim=-1)
        return mean.unsqueeze(-2) + sample*cov.unsqueeze(-2), self.dist.log_density(sample) - dets.unsqueeze(-1)


class SimpleProposal(pydpf.Module):
    def __init__(self, observation_dim, state_dim, generator):
        super().__init__()
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.mean_layer = FCNN(self.observation_dim, self.state_dim, self.state_dim, device=generator.device)
        self.cov_layer = FCNN(self.observation_dim, self.state_dim**2, self.state_dim**2, device=generator.device)
        self.dist = pydpf.MultivariateGaussian(torch.zeros(state_dim, device=generator.device), torch.eye(state_dim, device=generator.device), False, generator=generator)

    def forward(self, n_particles, observation: Tensor, **data):
        means = self.mean_layer(observation)
        temp = self.cov_layer(observation)
        temp = temp.reshape(temp.size(0), temp.size(1), self.state_dim, self.state_dim)
        temp = torch.tril(temp)
        temp[:,:, torch.arange(self.state_dim), torch.arange(self.state_dim)] = torch.abs(temp[:,:, torch.arange(self.state_dim), torch.arange(self.state_dim)])
        cov = temp
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        #print(temp[0, 0])
        dets = torch.linalg.slogdet(temp)[1]
       # print(torch.mean(-(self.dist.log_density(sample)), dim=-1))
        return means.unsqueeze(-2) + torch.einsum('tbpi,tbij->tbpj', sample, cov), self.dist.log_density(sample) - dets.unsqueeze(-1)

class Basicprop(pydpf.Module):
    def __init__(self, observation_dim, state_dim, generator):
        super().__init__()
        self.observation_dim = observation_dim
        self.dist = pydpf.MultivariateGaussian(torch.zeros(state_dim, device=generator.device), torch.nn.Parameter(torch.eye(state_dim, device=generator.device)*0.1, requires_grad=True), True, generator=generator)

    def forward(self, n_particles, observation: Tensor, **data):
        sample = self.dist.sample((observation.size(0), observation.size(1), n_particles))
        return observation.unsqueeze(-2) + sample, self.dist.log_density(sample)