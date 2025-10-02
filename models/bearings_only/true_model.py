import torch
import pydpf
from pydpf import Module
from einops import rearrange, repeat


def _wrap_angle(angle):
    return (angle - torch.pi) % (2 * torch.pi) - torch.pi

class true_prior(Module):
    def __init__(self, generator):
        super().__init__()
        self.device = generator.device
        self.generator = generator
        pos_maxes = pydpf.multiple_unsqueeze(torch.tensor([10, 10, torch.pi], device=self.device),2, 0 )
        pos_mins = pydpf.multiple_unsqueeze(torch.tensor([-10, -10, -torch.pi], device=self.device), 2, 0)
        target_maxes = pydpf.multiple_unsqueeze(torch.tensor([10, 10], device=self.device),2, 0 )
        target_mins = pydpf.multiple_unsqueeze(torch.tensor([10, 10], device=self.device), 2, 0)
        self.total_maxes = torch.cat([pos_maxes, target_maxes], dim=-1)
        self.total_mins = torch.cat([pos_mins, target_mins], dim=-1)
        self.speed_probs = torch.tensor([0.5, 0.5], device=self.device)
        self.speed_options = torch.tensor([1., 2.], device=self.device)

    def sample(self, n_particles, batch_size, **data):
        rand = torch.rand((batch_size, n_particles, 5), device=self.device, generator=self.generator)
        pos_target_state = rand * (self.total_maxes - self.total_mins) + self.total_mins
        speed_index = torch.multinomial(self.speed_probs, batch_size * n_particles, replacement=True, generator=self.generator)
        speed_index = rearrange(speed_index, '(b n) -> b n 1', b = batch_size)
        speed = self.speed_options[speed_index]
        return torch.cat([pos_target_state, speed, torch.zeros_like(speed)], dim=-1)


class true_dynamics(Module):
    def __init__(self, generator):
        super().__init__()
        self.device = generator.device
        self.generator = generator
        self.target_maxes = pydpf.multiple_unsqueeze(torch.tensor([10, 10], device=self.device),2, 0 )
        self.target_mins = pydpf.multiple_unsqueeze(torch.tensor([-10, -10], device=self.device), 2, 0)
        self.speed_probs = torch.tensor([0.5, 0.5], device=self.device)
        self.speed_options = torch.tensor([1., 2.], device=self.device)
        self.max_steps = 1000
        self.observation_gap = 3
        self.dt = 0.25
        self.max_turn = torch.tensor(30 * torch.pi / 180)

    def new_target(self, n_particles, batch_size, ):
        speed_index = torch.multinomial(self.speed_probs, batch_size * n_particles, replacement=True, generator=self.generator)
        speed_index = rearrange(speed_index, '(b n) -> b n 1', b=batch_size)
        speed = self.speed_options[speed_index]
        rand = torch.rand((batch_size, n_particles, 2), device=self.device, generator=self.generator)
        new_target = rand * (self.target_maxes - self.target_mins) + self.target_mins
        return torch.cat([new_target, speed, torch.zeros_like(speed)], dim=-1)


    def update_angle(self, angle, target, loc):
        diff = target - loc
        target_angle = torch.atan2(diff[...,1], diff[...,0]).unsqueeze(-2)
        angle_diff = _wrap_angle(target_angle - angle)
        d_angle = torch.minimum(angle_diff, self.max_turn)
        new_angle = _wrap_angle(angle + d_angle)
        return new_angle

    def update_loc(self, angle, loc, speed):
        x_diff = self.dt * speed * torch.cos(angle)
        y_diff = self.dt * speed * torch.sin(angle)
        return loc + torch.cat([x_diff, y_diff], dim=-1)


    def do_dynamics_step(self, prev_state):
        loc = prev_state[..., :2]
        angle = prev_state[..., 2:3]
        target = prev_state[..., 3:5]
        target_state = prev_state[..., 3:]
        count = prev_state[..., 6:]
        new_target_state = self.new_target(prev_state.size(1), prev_state.size(0))
        at_target = torch.linalg.vector_norm(loc - target) < 0.5
        over_max_steps = count > self.max_steps
        target_state = torch.where(torch.logical_or(at_target, over_max_steps), new_target_state, target_state)
        target = target_state[..., 0:2]
        speed = target_state[..., 2:3]
        target_state[..., 3] = target_state[..., 3] + 1
        new_angle = self.update_angle(angle, target, loc)
        new_loc = self.update_loc(new_angle, loc, speed)
        return torch.cat([new_loc, new_angle, target_state], dim=-1)


    def sample(self, prev_state):
        c_state = prev_state
        for i in range(self.observation_gap):
            c_state = self.do_dynamics_step(c_state)
        return c_state

class true_observation(Module):
    def __init__(self, generator):
        super().__init__()
        self.device = generator.device
        self.generator = generator
        self.loc = torch.tensor([[[-5, 0]]], device=self.device)
        self.alpha = 0.85
        self.vonMises = pydpf.VonMises(torch.zeros((1,), device=self.device), torch.full((1,), 100, device=self.device), generator=self.generator)


    def sample(self, state, **data):
        diff = state[..., 0:2] - self.loc
        bearing = torch.atan2(diff[..., 1], diff[..., 0]).unsqueeze(-1)
        sample = self.vonMises.sample((bearing.size(0), bearing.size(1)))
        von_Mises_shifted = _wrap_angle(bearing + sample)
        uniform_angle = torch.rand(sample.size(), device=self.device, generator=self.generator)*(2*torch.pi) - torch.pi
        use_von_Mises = torch.rand(sample.size(), device=self.device, generator=self.generator) < self.alpha
        return torch.where(use_von_Mises, von_Mises_shifted, uniform_angle)

    def score(self, state, obseravtion, **data):
        return 0

