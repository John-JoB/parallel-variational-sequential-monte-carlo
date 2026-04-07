from pydpf import Module
import torch
from torch import Tensor
import einops
from parallel_smoother_new import ParallelSmoother



class DMM(Module):
    def __init__(self, proposal, SSM, clip_likelihoods_for_stability = False):
        super().__init__()
        self.proposal = proposal
        self.SSM = SSM
        self.beta_observation = 1.
        self.beta_dynamic = 1.
        self.beta_prior = 1.
        self.beta_proposal = 1.
        self.mode = "model"
        self.clip_likelihoods = clip_likelihoods_for_stability

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

        one_tensor = torch.tensor(1., device=observation.device)
        obs_weighting = torch.tensor(self.beta_observation, device=observation.device) if self.training else one_tensor
        dyn_weighting = torch.tensor(self.beta_dynamic, device=observation.device) if self.training else one_tensor
        prop_weighting = torch.tensor(self.beta_proposal, device=observation.device) if self.training else one_tensor
        prior_weighting = torch.tensor(self.beta_prior, device=observation.device) if self.training else one_tensor

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
        with torch.profiler.record_function("Observation model"):
            obs_score = self.SSM.observation_model.score(**batched_data)
        with torch.profiler.record_function("Reshaping"):
            t_zero_data = ParallelSmoother._get_time_zero_data(state = state, observation=observation, control=control, time=time, series_metadata=series_metadata)
        with torch.profiler.record_function("Prior Model"):
            prior_density = self.SSM.prior_model.log_density(**t_zero_data)
        del(batched_data["state"])
        prev_state = einops.rearrange(state[:-1], f"t b n d -> (t b) n d")
        state = einops.rearrange(state[1:], "t b n d -> (t b) n d")
        with torch.profiler.record_function("Dynamic Model"):
            dynamic_density = self.SSM.dynamic_model.log_density(state=state, prev_state=prev_state, **batched_data)
        with torch.profiler.record_function("Reshaping"):
            obs_score = einops.rearrange(obs_score, "(t b) n -> t b n", t = time_extent + 1)
            dynamic_density = einops.rearrange(dynamic_density, "(t b) m -> t b m", t = time_extent, m = n_particles)

        prior_density = self.apply_beta.apply(prior_density, prior_weighting)
        dynamic_density = self.apply_beta.apply(dynamic_density, dyn_weighting)
        obs_score = self.apply_beta.apply(obs_score, obs_weighting)
        prop_density = self.apply_beta.apply(prop_density, prop_weighting)

        with torch.profiler.record_function("Kernel creation"):
            kernels = obs_score[1:] + dynamic_density - prop_density[1:]
            time_zero_l = obs_score[0].squeeze() + prior_density - prop_density[0].squeeze()
        if self.clip_likelihoods:
            kernels = torch.clip(kernels, -1e2 / 2, float('inf'))
            time_zero_l = torch.clip(time_zero_l, -1e2 / 2, float('inf'))
        elbo = torch.mean(time_zero_l + torch.sum(kernels, dim=0), dim =-1)
        if isinstance(aggregation_function, dict):
            output = {}
            for name, function in aggregation_function.items():
                output[name] = function(kernel = kernels, elbo = elbo, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)
            return output
        return aggregation_function( kernel = kernels, elbo = elbo, initial_likelihood = time_zero_l, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=state)

