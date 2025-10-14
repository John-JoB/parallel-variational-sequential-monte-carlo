import pydpf
from pydpf import Module
import torch
from torch import Tensor
import einops
from math import log, ceil
import opt_einsum as oe

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


    @staticmethod
    def _get_time_zero_data(**data):
        batched_data = {}
        for key, value in data.items():
            if value is None:
                continue
            if key == "series_metadata":
                batched_data[key] = value
                continue
            batched_data[key] = value[0]
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
            exp_output = oe.contract(f"{letters} i j, {letters} j k, {letters} k l -> {letters} i l", exp_left, exp_centre, exp_right, backend="torch")
            ctx.save_for_backward(exp_left, exp_centre, exp_right, exp_output)
            return (torch.log(exp_output) + max_left + max_right + max_centre - log(centre.size(-1))*2)

        @staticmethod
        def backward(ctx, do):
            exp_left, exp_centre, exp_right, exp_output = ctx.saved_tensors
            letters = ParallelSmoother.einsum_letters[:2 * (exp_left.dim() - 2)]
            grad_scaled = do / exp_output
            #Do a lot of repeated computation to save having a giant tensor in memory.
            grad_left = oe.contract(f"{letters} j k, {letters} k l, {letters} i l -> {letters} i j", exp_centre, exp_right, grad_scaled, backend="torch") * exp_left
            grad_right = oe.contract(f"{letters} i j, {letters} j k, {letters} i l -> {letters} k l", exp_left, exp_centre, grad_scaled, backend="torch") * exp_right
            grad_centre = oe.contract(f"{letters} i j, {letters} k l, {letters} i l -> {letters} j k", exp_left, exp_right, grad_scaled, backend="torch") * exp_centre
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

    @staticmethod
    def print_grad(grad):
        print("Gradient for state:")
        print(torch.mean(grad, dim=1))
        print("---")

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
        with torch.profiler.record_function("Proposal model"):
            state, prop_density = self.proposal(n_particles, observation=observation, control=control, time=time, series_metadata=series_metadata)
        with torch.profiler.record_function("Reshaping"):
            batched_data = ParallelSmoother._get_batched_dict(ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
            state_repeat = einops.repeat(state[1:], 't b n d -> (t b) (n m) d', m = n_particles)
            prev_state_repeat = einops.repeat(state[:-1], 't b n d -> (t b) (m n) d', m = n_particles)
        with torch.profiler.record_function("Observation model"):
            obs_score = self.SSM.observation_model.score(**batched_data)
        with torch.profiler.record_function("Reshaping"):
            del (batched_data["state"])
            t_zero_data = ParallelSmoother._get_time_zero_data(state = state, observation=observation, control=control, time=time, series_metadata=series_metadata)
        with torch.profiler.record_function("Prior Model"):
            prior_density = self.SSM.prior_model.log_density(**t_zero_data)
        with torch.profiler.record_function("Dynamic Model"):
            dynamic_density = self.SSM.dynamic_model.log_density(state=state_repeat, prev_state=prev_state_repeat, **batched_data)
        with torch.profiler.record_function("Reshaping"):
            obs_score = einops.rearrange(obs_score, "(t b) n -> t b 1 n", t = time_extent + 1)
            prop_density = einops.rearrange(prop_density, "t b n -> t b 1 n", t = time_extent + 1)
            dynamic_density = einops.rearrange(dynamic_density, "(t b) (m n) -> t b m n", t = time_extent, m = n_particles)

        with torch.profiler.record_function("Kernel creation"):
            kernels = obs_score[1:] + dynamic_density - prop_density[1:]
            #kernels.register_hook(self.print_grad)
            time_zero_l = obs_score[0].squeeze() + prior_density - prop_density[0].squeeze()
            even_t_e = False
            if time_extent % 2 == 0:
                ls = torch.concat([time_zero_l[None, :, :, None].expand(-1, -1, -1, kernels.size(-1)),  kernels[1::2]], dim=0)
                remaining_kernels = kernels[::2]
                even_t_e = True
            else:
                ls = kernels[::2]
                remaining_kernels = kernels[1::2]
                ls[0] += time_zero_l.unsqueeze(-1)
            ls = ls.unsqueeze(1)
            new_kernels = remaining_kernels.unsqueeze(1)
        with torch.profiler.record_function("Prefix sum"):
            while True:
                ls, new_kernels = self.tree_recurse(ls, new_kernels)
                if ls.size(0) == 2:
                    break
        with torch.profiler.record_function("Compiling final weights"):
            useful_ls = ls[:, :ceil((time_extent-1) /2)]
            left_facing_ls = torch.logsumexp(useful_ls[0], dim=-2, keepdim=False).unsqueeze(-1)
            right_facing_ls = torch.logsumexp(useful_ls[1], dim=-1, keepdim=False).unsqueeze(-2)
            combined_weights = left_facing_ls + right_facing_ls + remaining_kernels
            weights = einops.rearrange([torch.logsumexp(combined_weights, dim=-1), torch.logsumexp(combined_weights, dim=-2)], "s t b n -> (t s) b n")
            outer_weights = ParallelSmoother.logsumredexp.apply(useful_ls[0, 0], remaining_kernels[0], useful_ls[1, 0]) + 2*log(weights.size(-1))
            if even_t_e:
                weights = torch.concat([weights, torch.logsumexp(outer_weights, dim=-2).unsqueeze(0)], dim=0) - 3*log(weights.size(-1))
            else:
                weights = torch.concat([torch.logsumexp(outer_weights, dim=-1).unsqueeze(0), weights, torch.logsumexp(outer_weights, dim=-2).unsqueeze(0)], dim=0) - 3 * log(weights.size(-1))
            weights, test = pydpf.normalise(weights)
        with torch.profiler.record_function("Calculating outputs"):
            if isinstance(aggregation_function, dict):
                output = {}
                for name, function in aggregation_function.items():
                    output[name] = function(weight=weights, kernel = kernels, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
                return output
            return aggregation_function(weight=weights, kernel = kernels, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)