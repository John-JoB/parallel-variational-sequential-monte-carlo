import pydpf
import torch

class GaussianDynamic(pydpf.Module):
    def __new__(cls, dx:int, generator):
        device = generator.device
        dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        dynamic_offset = torch.zeros(dx, device=device)
        return pydpf.LinearGaussian(weight=torch.nn.Parameter(dynamic_matrix, requires_grad=False),
                                    bias=torch.nn.Parameter(dynamic_offset, requires_grad=False),
                                    cholesky_covariance=torch.nn.Parameter(torch.eye(dx, device=device), requires_grad=False),
                                    generator=generator)

class GaussianObservation(pydpf.Module):
    def __new__(cls, dx:int, dy:int, generator):
        device = generator.device
        observation_matrix = torch.zeros((dy, dx), device=device)
        for i in range(dy):
            observation_matrix[i, i] = 1
        observation_offset = torch.nn.Parameter(torch.zeros(dy, device=device), requires_grad=False)
        return pydpf.LinearGaussian(weight=torch.nn.Parameter(observation_matrix, requires_grad=False),
                                    bias=observation_offset,
                                    cholesky_covariance=torch.nn.Parameter(torch.eye(dy, device=device), requires_grad=False),
                                    generator=generator)

class GaussianPrior(pydpf.Module):
    def __new__(cls, dx:int, generator):
        device = generator.device
        return pydpf.MultivariateGaussian(torch.nn.Parameter(torch.zeros(dx, device=device), requires_grad=False),
                                          torch.nn.Parameter(torch.eye(dx, device=device), requires_grad=False),
                                          generator=generator)

class GaussianOptimalProposal(pydpf.Module):
    def __init__(self, dx:int, dy:int, generator):
        super().__init__()
        device = generator.device
        covariance = torch.eye(dx, device=device)
        self.dx = dx
        self.dy = dy
        for i in range(dy):
            covariance[i,i] = .5
        self.dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device), cholesky_covariance=torch.sqrt(covariance), generator=generator)

    def sample(self, observation, prev_state, **data):
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1)
        mean[:,:,:self.dy] = (mean[:,:,:self.dy] + observation.unsqueeze(1))/2
        return mean + sample

    def log_density(self, observation, prev_state, state, **data):
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1)
        mean[:, :, :self.dy] = (mean[:, :, :self.dy] + observation.unsqueeze(1))/2
        sample = state - mean
        return self.dist.log_density(sample)


class InformationAuxiliary(pydpf.Module):
    @staticmethod
    def propagate_dist(mean, cov, condition_matrix, offset, conditional_cov):
        new_mean = condition_matrix @ mean + offset
        new_cov = condition_matrix @ cov @ condition_matrix.transpose(-1,-2) + conditional_cov
        return new_mean, new_cov

    def __init__(self, SSM_prior, SSM_dynamic, max_time_extent):
        super().__init__()
        self.SSM_prior = SSM_prior
        self.SSM_dynamic = SSM_dynamic
        self.max_time_extent = max_time_extent
        self.dist = pydpf.StandardGaussian(SSM_prior.mean.size(0), generator=self.SSM_prior.generator)

    @pydpf.cached_property
    def dist_info(self):
        c_mean = self.SSM_prior.mean
        c_cov = self.SSM_prior.cholesky_covariance @ self.SSM_prior.cholesky_covariance.transpose(-1,-2)
        condition_matrix = self.SSM_dynamic.mean_fun.weight
        offset = self.SSM_dynamic.mean_fun.bias
        conditional_cov = self.SSM_dynamic.dist.cholesky_covariance @ self.SSM_dynamic.dist.cholesky_covariance.transpose(-1,-2)
        means = [c_mean]
        covs = [c_cov]
        for t in range(self.max_time_extent):
            c_mean, c_cov = self.propagate_dist(c_mean.unsqueeze(-1), c_cov, condition_matrix, offset.unsqueeze(-1), conditional_cov)
            c_mean = c_mean.squeeze(-1)
            means.append(c_mean)
            covs.append(c_cov)
        return torch.stack(means, dim=0), torch.stack(covs, dim=0)

    @pydpf.cached_property
    def dist_mean(self):
        return self.dist_info[0]

    @pydpf.cached_property
    def dist_cov(self):
        return self.dist_info[1]

    @pydpf.cached_property
    def inv_chole_cov(self):
        chole_covs = torch.linalg.cholesky(self.dist_cov)
        I = torch.eye(chole_covs.size(-1), device=chole_covs.device, dtype=chole_covs.dtype)
        return torch.linalg.solve_triangular(chole_covs, I, upper=False)

    @pydpf.cached_property
    def half_log_det_cov(self):
        return torch.sum(torch.log(torch.diagonal(self.inv_chole_cov, dim1=-2, dim2=-1)), dim=-1)/2

    def log_density(self, state, **data):
        normalised_state = self.inv_chole_cov[:state.size(0)].unsqueeze(-3).unsqueeze(-3) @ (state - self.dist_mean[:state.size(0)].unsqueeze(-2).unsqueeze(-2)).unsqueeze(-1)
        return self.dist.log_density(normalised_state.squeeze(-1)) - self.half_log_det_cov[:state.size(0)].unsqueeze(-1).unsqueeze(-1)



class InformationPrior(pydpf.Module):
    def __init__(self, auxiliary, time_extent):
        super().__init__()
        self.auxiliary = auxiliary
        self.time_extent = time_extent
        self.dist = pydpf.StandardGaussian(auxiliary.SSM_prior.mean.size(0), generator=auxiliary.SSM_prior.generator)

    @pydpf.cached_property
    def dist_chole_cov(self):
        #return (self.auxiliary.dist_cov[self.time_extent])
        return torch.linalg.cholesky(self.auxiliary.dist_cov[self.time_extent])

    @pydpf.cached_property
    def dist_mean(self):
        return self.auxiliary.dist_mean[self.time_extent]

    def sample(self, batch_size, n_particles, **data):
        mean = self.dist_mean
        sample = self.dist.sample((batch_size, n_particles))
        return mean + (self.dist_chole_cov @ sample.unsqueeze(-1)).squeeze(-1)


class InformationDynamic(pydpf.Module):
    def __init__(self, auxiliary, time_extent):
        super().__init__()
        self.auxiliary = auxiliary
        self.time_extent = time_extent
        self.dist = pydpf.StandardGaussian(auxiliary.SSM_prior.mean.size(0), generator=auxiliary.SSM_prior.generator)

    @pydpf.cached_property
    def dist_info(self):
        prior_mean = self.auxiliary.dist_mean
        prior_cov = self.auxiliary.dist_cov
        condition_matrix = self.auxiliary.SSM_dynamic.mean_fun.weight
        t_condition_matrix = condition_matrix.transpose(-1, -2)
        offset = self.auxiliary.SSM_dynamic.mean_fun.bias
        conditional_cov = self.auxiliary.SSM_dynamic.dist.cholesky_covariance @ self.auxiliary.SSM_dynamic.dist.cholesky_covariance.transpose(-1, -2)

        innovation_cov = conditional_cov + condition_matrix @ prior_cov @ t_condition_matrix
        kalman_gain = torch.linalg.solve(innovation_cov, prior_cov @ t_condition_matrix, left=False)
        backwards_offset = prior_mean - (kalman_gain @ (condition_matrix @ prior_mean.unsqueeze(-1) + offset.unsqueeze(-1))).squeeze(-1)
        backwards_cov = (torch.eye(backwards_offset.size(-1), device=backwards_offset.device, dtype=backwards_offset.dtype) - kalman_gain @ condition_matrix) @ prior_cov
        return kalman_gain, backwards_offset, backwards_cov

    @pydpf.cached_property
    def dist_offset(self):
        return self.dist_info[1]

    @pydpf.cached_property
    def dist_chole_cov(self):
        return torch.linalg.cholesky(self.dist_info[2])

    @pydpf.cached_property
    def dist_matrix(self):
        return self.dist_info[0]

    def sample(self, t, prev_state, **data):
        mean = (self.dist_matrix[self.time_extent - t + 1] @ prev_state.unsqueeze(-1)).squeeze(-1) + self.dist_offset[self.time_extent - t + 1]
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        return mean + (self.dist_chole_cov[self.time_extent - t + 1] @ sample.unsqueeze(-1)).squeeze(-1)
