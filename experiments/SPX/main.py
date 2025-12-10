from ast import Param

import pydpf
from experiments.common.parameter_set import ParameterSet
import models.SPX.model as model
import torch
import matplotlib
from experiments.common.optimisers import OptimList

from parallel_smoother_new import ParallelSmoother
from smoother_outputs import dSMC_ELBO, VAE_ELBO
from experiments.common.training import Trainer, TrainingStage, VanillaPydpfRun
from experiments.SPX.simulate_paths import plot_paths
import einops

#matplotlib.use('TkAgg')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class weird_prop(pydpf.Module):
    def __init__(self, proposal, dynamic):
        super().__init__()
        self.proposal = proposal
        self.dynamic = dynamic

    def forward(self, n_particles, observation, series_metadata, **kwargs):
        prev_state, prop_log_density = self.proposal(n_particles, observation, series_metadata, **kwargs)
        #return prev_state, prop_log_density
        flat_prev_state = einops.rearrange(prev_state[:-1], 't b n d -> (t b) n d')
        flat_state = self.dynamic.sample(prev_state=flat_prev_state)
        log_density = torch.empty((observation.size(0), observation.size(1), n_particles), device=observation.device)
        log_density[0] = prop_log_density[0]
        state = torch.empty_like(prev_state)
        state[0] = prev_state[0]
        #print(flat_state.shape)
        state[1:] = einops.rearrange(flat_state, '(t b) n d -> t b n d', t = observation.size(0)-1)
        flat_density = self.dynamic.log_density(prev_state = flat_prev_state, state = flat_state)
        log_density[1:] = einops.rearrange(flat_density, '(t b) n -> t b n', t = observation.size(0)-1) + prop_log_density[:-1]
        #log_density[1:] = prop_log_density[:-1]
        return state, log_density



def make_model(dx, use_vix = False):
    gen = torch.Generator(device=device).manual_seed(0)
    prior_model = model.prior_model(dx, gen, use_vix=use_vix, depth=3, hidden_dim=8)
    dynamic_model = model.dynamic_model(dx, gen, 3, 3, 32)
    #dynamic_model = model.very_simple_dynamic(dx, gen)
    #dynamic_model = model.dynamic_model2(dx, 8, 64, gen)
    #dynamic_model = model.dynamic_model3(dx, gen, 7, 128, 24)
    #observation_model = model.observation_model(dx, 7, 32, gen)
    observation_model = model.observation_model3(dx, gen, 7, 64, 16)
    #observation_model = model.observation_model3(dx, gen, 1, 4, 16)
    proposal_model = model.proposal_model(dx, gen, True)
    return prior_model, dynamic_model, observation_model, proposal_model

def make_ssm(prior_model, dynamic_model, observation_model):
    return pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)



def make_pvmc(dx, use_vix = False):
    prior_model, dynamic_model, observation_model, proposal_model = make_model(dx, use_vix=use_vix)
    SSM = make_ssm(prior_model, dynamic_model, observation_model)
    return ParallelSmoother(proposal_model, SSM)

def make_dummy_pvmc(dx, pvmc):
    ssm = make_ssm(model.dummy_prior(dx, torch.Generator(device = device).manual_seed(0)), model.dummy_dynamic(dx, torch.Generator(device = device).manual_seed(0)), pvmc.SSM.observation_model)
    return ParallelSmoother(pvmc.proposal, ssm)

def make_dummy_pvmc2(dx, pvmc):
    ssm = make_ssm(model.dummy_prior(dx, torch.Generator(device = device).manual_seed(0)), model.dummy_dynamic(dx, torch.Generator(device = device).manual_seed(0)), pvmc.SSM.observation_model)
    weird_p = weird_prop(pvmc.proposal, pvmc.SSM.dynamic_model)
    return ParallelSmoother(weird_p, ssm)

def make_filter(dx):
    prior_model, dynamic_model, observation_model, proposal_model = make_model(dx)
    SSM = make_ssm(prior_model, dynamic_model, observation_model)
    return pydpf.MarginalStopGradientDPF(SSM, torch.Generator(device = device).manual_seed(0))

def make_run_info(dataset, SSM):
    train_info = {"n_particles": 32,
                  "batch_size": 32,
                  "collate_fn": dataset.collate,
                  "shuffle": True,
                  "time_extent": 119,
                  "output_function": {"ELBO": dSMC_ELBO(), "reconstruction": GMRecon(SSM),}} #"proprecon": GMDoubleRecon(SSM)}}
    info = {"train": train_info,
            "loss": f"(- time_average.ELBO / {len(dataset.observation)})",
            "print_each_epoch": {"Train ELBO": "train.mean.time_average.ELBO", "Train reconstruction": "train.mean.time_average.reconstruction"},# "Prop reconstruction": "train.mean.time_average.proprecon"},
            "epochs": 20,
            "device": device,
            "target": f"-train.mean.time_average.ELBO"
            }

    return info

def make_filter_run_info(dataset, SSM):
    train_info = {"n_particles": 50,
                  "batch_size": 64,
                  "collate_fn": dataset.collate,
                  "time_extent": 119,
                  "output_function": {"ELBO": pydpf.LogLikelihoodFactors()}}
    info = {"train": train_info,
            "loss": f"(- time_average.ELBO)",
            "print_each_epoch": {"Train ELBO": "train.mean.time_average.ELBO"},
            "epochs": 1,
            "device": device,
            "target": f"-train.mean.time_average.ELBO"
            }

    return info

class SimpleRecon(pydpf.Module):
    need_weight = False

    def __init__(self, SSM):
        super().__init__()
        self.model = SSM.observation_model

    def forward(self, state, observation, **data):
        prop_obs = self.model.sample(state = torch.flatten(state, 0, 1))
        prop_obs = torch.reshape(prop_obs, (state.shape[0], state.shape[1], -1, 1))
        mean_obs = torch.mean(prop_obs, dim=-2)

        return torch.sum((mean_obs - observation)**2, dim=-1)

class GMRecon(pydpf.Module):
    need_weight = False
    def __init__(self, SSM):
        super().__init__()
        self.model = SSM.observation_model
        self.net = self.model.net

    def forward(self, state, observation, **data):
        dist_info = self.net(state)
        locs = dist_info[..., :dist_info.size(-1) // 2]
        weights = dist_info[..., dist_info.size(-1) // 2:]
        weights, _ = pydpf.normalise(weights)
        mean_obs = torch.sum(torch.exp(weights) * locs, dim=-1)
        return (torch.mean(mean_obs, dim=-1) - observation.squeeze(-1))**2

class GMDoubleRecon(pydpf.Module):
    need_weight = False
    def __init__(self, SSM):
        super().__init__()
        self.obs = SSM.observation_model
        self.dyn = SSM.dynamic_model
        self.recon = GMRecon(SSM)

    def forward(self, state, observation, **data):
        reshape_state = einops.rearrange(state, "t b n j -> (t b) n j")
        reshape_obs = einops.rearrange(torch.roll(observation, shifts=-1, dims=0), "t b j -> (t b) j")
        next_state = self.dyn.sample(prev_state=reshape_state)
        return -einops.rearrange(torch.mean(self.obs.score(state = next_state, observation = reshape_obs), dim = -1), "(t b) -> t b", t = state.size(0))[:-1]


class run_each_epoch():
    def __init__(self, model):
        self.model = model

    def __call__(self):
        self.model.beta_observation = (self.model.beta_observation - 1) * 0.9 + 1
        self.model.beta_prior = 20.

def make_trainer_routine(model, dataset, dummy_model = None):
    params = ParameterSet(model)
    optim = torch.optim.Adam([{"params": params - ParameterSet(model.proposal), "lr":1e-3}], lr=1e-3)
    #optim = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    runner = VanillaPydpfRun(model)
    if dummy_model is not None:
        dummy_optim = torch.optim.Adam(params, lr=1e-3)
        dummy_optim2 = torch.optim.Adam(params - ParameterSet(model.proposal), lr=1e-5)
        dummy_runner = VanillaPydpfRun(dummy_model)
        dummy_stage = TrainingStage(dummy_runner, dataset, None, None, dummy_optim, ["observation", "series_metadata"], run_on_epoch=run_each_epoch(dummy_model))
        dummy_model_2 = make_dummy_pvmc2(8, model)
        dummy_runner_2 = VanillaPydpfRun(dummy_model_2)
        dummy_stage_2 = TrainingStage(dummy_runner_2, dataset, None, None, dummy_optim2, ["observation", "series_metadata"])
   #obs_params = ParameterSet(model.SSM.observation_model)
    #prop_params = ParameterSet(model.proposal)

    stage = TrainingStage(runner, dataset, None, None, optim, ["observation", "series_metadata"])
    trainer = Trainer(model, stages=[dummy_stage, stage])
    return trainer

def make_dataset():
    dataset = pydpf.StateSpaceDataset("./experiments/SPX/data/data.csv", device=device, series_metadata_path="./experiments/SPX/data/metadata.csv")
    return dataset


def to_log_ret(observation, **data):
    log_ret = torch.log(observation)
    return torch.cat((log_ret[0:1], log_ret[1:] - log_ret[:-1]), dim=0)

def from_log_ret(observation, **data):
    log_ret = torch.cumsum(observation, dim=0)
    return torch.exp(log_ret)

if __name__ == "__main__":
    dataset = make_dataset()
    dataset.apply(to_log_ret)
    #dataset.apply(lambda observation, **data: torch.log(observation))
    obs = dataset.observation
    mean_obs = torch.mean(obs, dim=(0,1)).squeeze()
    sd_obs = torch.sqrt(torch.var(obs, dim=(0,1)).squeeze())
    dataset.apply(lambda observation, **data: (mean_obs - observation)/sd_obs)
    obs = dataset.observation
    pvmc = make_pvmc(4, True)
    dummy_pvmc = make_dummy_pvmc(4, pvmc)
    dummy_pvmc.beta_observation = 10.
    dummy_info = make_run_info(dataset, dummy_pvmc.SSM)
    dummy_info["epochs"] = 10
    run_info = [dummy_info, make_run_info(dataset, pvmc.SSM)]
    #run_info = [make_filter_run_info(dataset, pvmc.SSM)]
    trainer_routine = make_trainer_routine(pvmc, dataset, dummy_pvmc)
    trainer_routine.fit("run", run_info, False)
    ssm = pvmc.SSM
    for n, p in ssm.named_parameters():
        print(n)
        print(p.mean())
    plot_paths(ssm, pvmc.proposal, 360, mean_obs, sd_obs, dataset)

