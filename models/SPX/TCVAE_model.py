import einops
import pydpf
import torch

from models.generic_nets.Normalizing_flow import RealNVP, NormalizingFlowModel

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(
            m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(m.bias)
        except:
            pass
    elif isinstance(m, torch.nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
    elif isinstance(m, torch.nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)

        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(m.bias)
        except:
            pass


class TCVAE_Encoder(pydpf.Module):
    def __init__(self, dx, hidden_dim, n_layers, device):
        super().__init__()
        self.linear_pre = torch.nn.Linear(1, hidden_dim)
        self.linear_pre.apply(init_weights)
        self.lstm = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=False)
        self.lstm.apply(init_weights)
        self.linear_post_1 = torch.nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_post_1.apply(init_weights)
        self.linear_post_2 = torch.nn.Linear(2*hidden_dim + 1, hidden_dim)
        self.linear_post_2.apply(init_weights)
        self.mean_net = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh(), torch.nn.Linear(hidden_dim, dx))
        self.mean_net.apply(init_weights)
        self.log_var_net = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh(), torch.nn.Linear(hidden_dim, dx))
        self.log_var_net.apply(init_weights)
        self.to(device)


    def forward(self, observation):
        pre_enc = torch.nn.functional.relu(self.linear_pre(observation))
        enc, _ = self.lstm(pre_enc)
        enc = torch.nn.functional.tanh(enc)
        res_enc_1 = torch.nn.functional.tanh(self.linear_post_1(torch.cat([enc, observation], dim=-1)))
        res_enc_2 = torch.nn.functional.tanh(self.linear_post_2(torch.cat([res_enc_1, enc, observation], dim=-1)))
        mean = self.mean_net(res_enc_1)
        logvar = self.log_var_net(res_enc_2)
        return mean, logvar

class TCVAE_Decoder(pydpf.Module):
    def __init__(self, dx, hidden_dim, n_layers, device):
        super().__init__()
        self.linear_pre = torch.nn.Linear(dx, hidden_dim)
        self.linear_pre.apply(init_weights)
        self.lstm = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=False)
        self.lstm.apply(init_weights)
        self.linear_post_1 = torch.nn.Linear(hidden_dim + dx, hidden_dim)
        self.linear_post_1.apply(init_weights)
        self.linear_post_2 = torch.nn.Linear(hidden_dim, 1)
        self.linear_post_2.apply(init_weights)
        self.to(device)

    def forward(self, state, **data):
        pre_dec = torch.nn.functional.tanh(self.linear_pre(state))
        dec, _ = self.lstm(pre_dec)
        res_dec_1 = self.linear_post_1(torch.nn.functional.tanh(torch.cat([dec, state], dim=-1)))
        res_dec_2 = self.linear_post_2(torch.nn.functional.tanh(res_dec_1))
        return res_dec_2


class TCVAE_Prior(pydpf.Module):
    def __init__(self, dx, hidden_dim, n_layers, generator, time_extent):
        super().__init__()
        self.dx = dx
        self.length = time_extent + 1
        priors_prior = pydpf.StandardGaussian(dx*self.length, generator=generator)
        flow_layers = [RealNVP(self.length * dx, hidden_dim, device=generator.device, depth = 2, activation="leaky_relu") for _ in range(n_layers)]
        self.flow = NormalizingFlowModel(priors_prior, flow_layers, device=generator.device)

    def sample(self, n_samples, **data):
        state = self.flow.sample(n_samples)
        return einops.rearrange(state, "b (t d) -> t b d", d = self.dx)

    def log_density(self, state, **data):
        rearr_state = einops.rearrange(state, "t b d -> b (t d)")
        return self.flow.log_density(rearr_state)



