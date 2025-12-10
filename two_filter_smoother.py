import pydpf
import torch
import einops

class TwoFilter(pydpf.Module):
    def __init__(self, SSM, information_SSM, generator):
        super().__init__()
        self.forward_SSM = SSM
        self.information_SSM = information_SSM
        self.forward_filter = pydpf.DPF(SSM, generator)
        self.information_filter = pydpf.DPF(information_SSM, generator)

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

            extra_dims_as_letters = letters[:(value.dim() - 2) * 2]
            batched_data[key] = einops.rearrange(value, f"t b{extra_dims_as_letters} -> (t b){extra_dims_as_letters}")
        return batched_data

    def forward(self,
                n_particles: int,
                time_extent: int,
                aggregation_function,
                observation,
                *,
                gradient_regulariser = None,
                ground_truth = None,
                control = None,
                time = None,
                series_metadata = None):
        observation = observation[:time_extent + 1]
        backwards_ob = torch.flip(observation, dims=(0,))
        if ground_truth is None:
            backwards_gt = None
        else:
            ground_truth = ground_truth[:time_extent + 1]
            backwards_gt = torch.flip(ground_truth, dims=(0,))
        if control is None:
            backwards_control = None
        else:
            control = control[:time_extent + 1]
            backwards_control = torch.flip(control, dims=(0,))
        if time is None:
            backwards_time = None
        else:
            time = time[:time_extent + 1]
            backwards_time = torch.flip(time, dims=(0,))

        forward_particles = self.forward_filter(n_particles,
                                                time_extent,
                                                {"state": pydpf.State(), "weight": pydpf.Weight(), "likelihood": pydpf.LogLikelihoodFactors()},
                                                observation,
                                                gradient_regulariser = gradient_regulariser,
                                                ground_truth = ground_truth,
                                                control = control,
                                                time = time,
                                                series_metadata = series_metadata)

        backward_particles = self.information_filter(n_particles,
                                                    time_extent-1,
                                                    {"state": pydpf.State(), "weight": pydpf.Weight()},
                                                    backwards_ob,
                                                    gradient_regulariser = gradient_regulariser,
                                                    ground_truth = backwards_gt,
                                                    control = backwards_control,
                                                    time = backwards_time,
                                                    series_metadata = series_metadata)
        forward_state = forward_particles["state"]
        forward_weight = forward_particles["weight"]
        backward_state = torch.flip(backward_particles["state"], dims =(0,))
        backward_weight = torch.flip(backward_particles["weight"], dims =(0,))

        state_repeat = einops.repeat(backward_state, 't b n d -> (t b) (m n) d', m=n_particles)
        prev_state_repeat = einops.repeat(forward_state[:-1], 't b n d -> (t b) (n m) d', m=n_particles)
        kernels = self.forward_SSM.dynamic_model.log_density(state = state_repeat, prev_state = prev_state_repeat)
        kernels = einops.rearrange(kernels, "(t b) (n m) -> t b n m", m=n_particles, t=time_extent)
        auxiliary_density = self.information_SSM.prior_model.auxiliary.log_density(backward_state)
        combined_weight = kernels + backward_weight.unsqueeze(-2) + forward_weight[:-1].unsqueeze(-1) - auxiliary_density.unsqueeze(-2)
        weight = torch.logsumexp(combined_weight, dim=-1)
        log_likelihood = torch.sum(forward_particles["likelihood"].squeeze(-1), dim=0)
        weight = torch.cat((weight, forward_weight[-1:]), dim=0)
        #weight = forward_weight
        weight, _ = pydpf.normalise(weight)
        if isinstance(aggregation_function, dict):
            output = {}
            for name, function in aggregation_function.items():
                output[name] = function(weight=weight, kernel=kernels, elbo=log_likelihood, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=forward_state)
            return output
        return aggregation_function(weight=weight, kernel=kernels, elbo=log_likelihood, ground_truth=ground_truth, control=control, time=time, series_metadata=series_metadata, observation=observation, state=forward_state)


