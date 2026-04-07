import pydpf
from pydpf import Module
import torch
from torch import Tensor
import einops
from math import log, ceil
import opt_einsum as oe
from parallel_scan import parallel_associative_reduce



class ParallelSmoother(Module):
    def __init__(self, proposal, SSM, clip_likelihoods_for_stability = False, control_net = None):
        super().__init__()
        self.proposal = proposal
        self.SSM = SSM
        self.beta_observation = 1.
        self.beta_dynamic = 1.
        self.beta_prior = 1.
        self.beta_proposal = 1.
        self.mode = "model"
        self.clip_likelihoods = clip_likelihoods_for_stability
        self.control_net = control_net
        self.use_control_var = True
        if control_net is None:
            self.use_control_var = False

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
        """@staticmethod
        def forward(ctx, left, centre, right):
            max_left = torch.amax(left, -1, keepdim=True)
            max_centre = torch.amax(centre, (-1, -2 ), keepdim=True)
            max_right = torch.amax(right, -2, keepdim=True)
            exp_right = torch.exp(torch.clamp(right - max_right, -800, 80))
            exp_centre = torch.exp(torch.clamp(centre - max_centre, -800, 80))
            exp_left = torch.exp(torch.clamp(left - max_left, -800, 80))
            exp_output = exp_left @ exp_centre @ exp_right
            ctx.save_for_backward(exp_left, exp_centre, exp_right, exp_output)
            return torch.log(torch.clamp(exp_output, -800, 80)) + max_left + max_right + max_centre"""

        @staticmethod
        def forward(ctx, left, centre, right):
            max_left = torch.amax(left, -1, keepdim=True)
            max_centre = torch.amax(centre, (-1, -2), keepdim=True)
            max_right = torch.amax(right, -2, keepdim=True)
            exp_right = torch.exp(right - max_right)
            exp_centre = torch.exp(centre - max_centre)
            exp_left = torch.exp(left - max_left)
            exp_output = exp_left @ exp_centre @ exp_right
            ctx.save_for_backward(exp_left, exp_centre, exp_right, exp_output)
            return torch.log(exp_output) + max_left + max_right + max_centre

        @staticmethod
        def backward(ctx, do):
            exp_left, exp_centre, exp_right, exp_output = ctx.saved_tensors
            letters = ParallelSmoother.einsum_letters[:2 * (exp_left.dim() - 2)]
            grad_scaled = do / (exp_output + 1e-8)
            #Do a lot of repeated computation to save having a giant tensor in memory.
            grad_left = oe.contract(f"{letters} j k, {letters} k l, {letters} i l -> {letters} i j", exp_centre, exp_right, grad_scaled, backend="torch") * exp_left
            grad_right = oe.contract(f"{letters} i j, {letters} j k, {letters} i l -> {letters} k l", exp_left, exp_centre, grad_scaled, backend="torch") * exp_right
            grad_centre = oe.contract(f"{letters} i j, {letters} k l, {letters} i l -> {letters} j k", exp_left, exp_right, grad_scaled, backend="torch") * exp_centre
            return grad_left, grad_centre, grad_right


    class logmatmulexp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, left, right):
            max_left = torch.amax(left, -1, keepdim=True)
            max_right = torch.amax(right, -2, keepdim=True)
            exp_right = torch.exp(torch.clamp(right - max_right, -800, 80))
            exp_left = torch.exp(torch.clamp(left - max_left, -800, 80))
            exp_output = exp_left @ exp_right
            ctx.save_for_backward(exp_left, exp_right, exp_output)
            return torch.log(torch.clamp(exp_output, -800, 80)) + max_left + max_right - log(left.size(-1))

        """@staticmethod
        def forward(ctx, left, right):
            max_left = torch.amax(left, -1, keepdim=True)
            max_right = torch.amax(right, -2, keepdim=True)
            exp_right = torch.exp(right - max_right)
            exp_left = torch.exp(left - max_left)
            exp_output = exp_left @ exp_right
            ctx.save_for_backward(exp_left, exp_right, exp_output)
            return torch.log(exp_output) + max_left + max_right - log(left.size(-1))"""

        @staticmethod
        def backward(ctx, do):
            exp_left, exp_right, exp_output = ctx.saved_tensors
            grad_scaled = do / (exp_output + 1e-8)
            grad_left =  (grad_scaled @ exp_right.transpose(-1, -2)) * exp_left
            grad_right = (grad_scaled.transpose(-1, -2) @ exp_left) * exp_right
            return grad_left, grad_right

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

    class clip_grad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t

        @staticmethod
        def backward(ctx, grad):
            return torch.clip(grad, -10, 10)

    @staticmethod
    def clamp_grad(grad):
        return torch.clip(grad, -1, 1)

    class prop_grad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, log_prop, baseline):
            ctx.save_for_backward(baseline)
            return log_prop

        @staticmethod
        def backward(ctx, grad):
            baseline, = ctx.saved_tensors
            controlled_grad = grad - baseline
            return controlled_grad, None

    class apply_beta(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor, beta):
            ctx.save_for_backward(beta)
            return tensor

        @staticmethod
        def backward(ctx, grad):
            beta, = ctx.saved_tensors
            return grad * beta, None



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

        need_weight = False
        if isinstance(aggregation_function, pydpf.Module) and aggregation_function.need_weight:
            need_weight = True
        elif isinstance(aggregation_function, dict):
            for v in aggregation_function.values():
                if v.need_weight:
                    need_weight = True

        one_tensor = torch.tensor(1., device = observation.device)
        obs_weighting = torch.tensor(self.beta_observation, device=observation.device)  if self.training else one_tensor
        dyn_weighting = torch.tensor(self.beta_dynamic, device=observation.device) if self.training else one_tensor
        prop_weighting = torch.tensor(self.beta_proposal, device=observation.device) if self.training else one_tensor
        prior_weighting = torch.tensor(self.beta_prior, device=observation.device) if self.training else one_tensor

        if not self.training:
            self.mode = "test"
        elif self.mode == "test":
            self.mode = "prop"
        elif self.mode == "prop":
            self.mode = "model"
        else:
            self.mode = "prop"

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
            state_repeat = einops.repeat(state[1:], 't b n d -> (t b) (m n) d', m = n_particles)
            prev_state_repeat = einops.repeat(state[:-1], 't b n d -> (t b) (n m) d', m = n_particles)
        with torch.profiler.record_function("Observation model"):
            obs_score = self.SSM.observation_model.score(**batched_data)
            #print("obs:", obs_score.mean())
            #obs_score = torch.zeros_like(obs_score)
        with torch.profiler.record_function("Reshaping"):
            del (batched_data["state"])
            t_zero_data = ParallelSmoother._get_time_zero_data(state = state, observation=observation, control=control, time=time, series_metadata=series_metadata)
        with torch.profiler.record_function("Prior Model"):
            prior_density = self.SSM.prior_model.log_density(**t_zero_data)
            #print("prior:", prior_density.mean())
            #prior_density = torch.zeros_like(prior_density)
        with torch.profiler.record_function("Dynamic Model"):
            dynamic_density = self.SSM.dynamic_model.log_density(state=state_repeat, prev_state=prev_state_repeat, **batched_data)
            #print("dyn:", dynamic_density.mean())
            #dynamic_density = torch.zeros_like(dynamic_density)
        if self.use_control_var and self.training:
            estimated_R = self.control_net(torch.cat([state_repeat, prev_state_repeat], dim=-1)).squeeze(-1)
            estimated_R = einops.rearrange(estimated_R, "(t b) (m n) -> t b m n", t = time_extent, m = n_particles)
            ler = estimated_R[-1:].mean(dim=-1)
            estimated_R = torch.cat([estimated_R.mean(dim = -1), ler],dim=0)
            prop_density = self.prop_grad.apply(prop_density, estimated_R.detach())
            dummy_R = torch.zeros_like(prop_density, requires_grad=True)
            prop_density = prop_density + dummy_R
        with torch.profiler.record_function("Reshaping"):
            obs_score = einops.rearrange(obs_score, "(t b) n -> t b 1 n", t = time_extent + 1)
            obs_score = self.apply_beta.apply(obs_score, obs_weighting)
            prop_density = einops.rearrange(prop_density, "t b n -> t b 1 n", t = time_extent + 1)
            prop_density = self.apply_beta.apply(prop_density, prop_weighting)
            dynamic_density = einops.rearrange(dynamic_density, "(t b) (m n) -> t b m n", t = time_extent, m = n_particles)
            dynamic_density = self.apply_beta.apply(dynamic_density, dyn_weighting)
        with torch.profiler.record_function("Kernel creation"):
            kernels = obs_score[1:] + dynamic_density - prop_density[1:]
            #if kernels.requires_grad:
            #    kernels = self.clip_grad.apply(kernels)
            prior_density = self.apply_beta.apply(prior_density, prior_weighting)
            time_zero_l = obs_score[0].squeeze() + prior_density - prop_density[0].squeeze()
        #print('start')
        #print(obs_score[0])
        #print(-prop_density[0])
        #print(kernels[0])
        if self.clip_likelihoods:
            kernels = torch.clip(kernels, -1e2 / 2, float('inf'))
            time_zero_l = torch.clip(time_zero_l, -1e2 / 2, float('inf'))

        if not need_weight:

            reduced = parallel_associative_reduce(ParallelSmoother.logmatmulexp.apply, None, False, kernels)
            reduced = reduced + time_zero_l.unsqueeze(-1)
            elbo = torch.logsumexp(reduced, dim=(-1, -2)) - 2 *log(kernels.size(-1))
            #print('Hmm')
            #print(obs_score[20, 0])
            #print(-prop_density[20, 0])
            #print(kernels[20, 0])
            #print(kernels.size())
            #print(dynamic_density[20, 0, 0])
            if isinstance(aggregation_function, dict):
                output = {}
                for name, function in aggregation_function.items():
                    output[name] = function(kernel = kernels, elbo = elbo, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
                return output
            return aggregation_function( kernel = kernels, elbo = elbo, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)

        #print(torch.min(time_zero_l))
        #print(torch.min(kernels))
        even_t_e = False
        if time_extent % 2 == 0:
            ls = torch.concat([time_zero_l[None, :, None, :].expand(-1, -1, kernels.size(-1), -1),  kernels[1::2]], dim=0)
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
            outer_weights = ParallelSmoother.logsumredexp.apply(useful_ls[0, 0], remaining_kernels[0], useful_ls[1, 0])
            if even_t_e:
                weights = torch.concat([weights, torch.logsumexp(outer_weights, dim=-2).unsqueeze(0)], dim=0) - (time_extent+2)*log(weights.size(-1))
            else:
                weights = torch.concat([torch.logsumexp(outer_weights, dim=-1).unsqueeze(0), weights, torch.logsumexp(outer_weights, dim=-2).unsqueeze(0)], dim=0) - (time_extent+1) * log(weights.size(-1))
            weights, test = pydpf.normalise(weights)

        with torch.profiler.record_function("Calculating outputs"):
            if isinstance(aggregation_function, dict):
                output = {}
                for name, function in aggregation_function.items():
                    output[name] = function(weight=weights, kernel = kernels, elbo = test[0].squeeze(), initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
            else:
                output = aggregation_function(weight=weights, kernel = kernels, elbo = test[0].squeeze(), initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
            if self.use_control_var and self.training:
                return output, estimated_R, dummy_R
            return output
