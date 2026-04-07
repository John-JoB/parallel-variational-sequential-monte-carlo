import pydpf
from experiments.common.parameter_set import ParameterSet
import models.SPX.model as model
import torch
from models.SPX.TCVAE_model import TCVAE_Prior, TCVAE_Decoder, TCVAE_Encoder
from time_causal_VAE import TCVAE
from experiments.SPX.tcvae_run import TCVAERun
from dmm import DMM
from parallel_smoother_new import ParallelSmoother
from smoother_outputs import dSMC_ELBO, VAE_ELBO
from experiments.common.training import Trainer, TrainingStage, VanillaPydpfRun
from experiments.SPX.simulate_paths import plot_paths
import einops
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#options: experiments = ["PVMC, VAE, TCVAE, Soft, DMM"]
experiments = ["PVMC, VAE, TCVAE, Soft, DMM"]





def make_model(dx, use_dmm_prop=False):
    gen = torch.Generator(device=device).manual_seed(0)
    prior_model = model.prior_model(dx, gen)
    dynamic_model = model.dynamic_model(dx, gen, 3, 3, 32)
    observation_model = model.observation_model(dx, gen, 7, 64, 16)
    if use_dmm_prop:
        proposal_model = model.dmm_proposal_model(dx, gen)
    else:
        proposal_model = model.proposal_model(dx, gen)
    return prior_model, dynamic_model, observation_model, proposal_model

def make_ssm(prior_model, dynamic_model, observation_model):
    return pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)


def make_dmm(dx):
    prior_model, dynamic_model, observation_model, proposal_model = make_model(dx, True)
    return DMM(proposal_model, make_ssm(prior_model, dynamic_model, observation_model))

def make_tcvae(dx):
    gen = torch.Generator(device=device).manual_seed(0)
    prior = TCVAE_Prior(dx, 250, 3, gen, 119)
    encoder = TCVAE_Encoder(dx, 16, 2, device)
    decoder = TCVAE_Decoder(dx, 16, 2, device)
    return TCVAE(encoder, decoder, prior, dx, gen, 0.04)


def make_pvmc(dx):
    prior_model, dynamic_model, observation_model, proposal_model = make_model(dx)
    SSM = make_ssm(prior_model, dynamic_model, observation_model)
    return ParallelSmoother(proposal_model, SSM)

def make_soft_dpf(dx):
    prior_model, dynamic_model, observation_model, proposal_model = make_model(dx)
    SSM = make_ssm(prior_model, dynamic_model, observation_model)
    #return pydpf.SoftDPF(SSM, resampling_generator= SSM.observation_model.gen)
    return pydpf.MarginalStopGradientDPF(SSM, resampling_generator=SSM.observation_model.gen)


def make_filter(dx):
    prior_model, dynamic_model, observation_model, proposal_model = make_model(dx)
    SSM = make_ssm(prior_model, dynamic_model, observation_model)
    return pydpf.MarginalStopGradientDPF(SSM, torch.Generator(device = device).manual_seed(0))

def make_run_info(dataset, SSM, vae = False):
    if vae:
        loss = VAE_ELBO()
    else:
        loss = dSMC_ELBO()
    train_info = {"n_particles": 16,
                  "batch_size": 32,
                  "collate_fn": dataset.collate,
                  "shuffle": True,
                  "time_extent": 119,
                  "output_function": {"ELBO": loss, "reconstruction": GMRecon(SSM),}}
    info = {"train": train_info,
            "loss": f"(- time_average.ELBO / {len(dataset.observation)})",
            "print_each_epoch": {"Train ELBO": "train.mean.time_average.ELBO", "Train reconstruction": "train.mean.time_average.reconstruction"},# "Prop reconstruction": "train.mean.time_average.proprecon"},
            "epochs": 100,
            "device": device,
            "target": f"-train.mean.time_average.ELBO"
            }

    return info

def make_filter_run_info(dataset, SSM):
    train_info = {"n_particles": 16,
                  "batch_size": 32,
                  "collate_fn": dataset.collate,
                  "shuffle": True,
                  "time_extent": 119,
                  "output_function": {"ELBO": pydpf.LogLikelihoodFactors()}}
    info = {"train": train_info,
            "loss": f"-time_average.ELBO",
            "print_each_epoch": {"Train ELBO": f"train.mean.time_average.ELBO.item() * {len(dataset.observation)}",},
            "epochs": 100,
            "device": device,
            "target": f"-train.mean.time_average.ELBO"
            }

    return info

def make_tcvae_run_info(dataset):
    train_info = {"batch_size": 32,
                  "collate_fn": dataset.collate,
                  "shuffle": True,
                  "time_extent": 119}
    info = {"train": train_info,
            "loss": f"time_average.l1_loss + time_average.kld",
            "print_each_epoch": {"Train l1_loss": f"train.mean.time_average.l1_loss", "Train kld": f"train.mean.time_average.kld"},
            "epochs": 100,
            "device": device,
            "target": f"train.mean.time_average.l1_loss + train.mean.time_average.kld"
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
        try:
            dist_info = self.net(state)
            locs = dist_info[..., :dist_info.shape[-1]//2]
            weights = dist_info[..., dist_info.shape[-1]//2:]/self.model.temperature
            weights = torch.softmax(weights, dim=-1)
            mean_obs = torch.sum(weights * locs, dim=-1)
            return (torch.mean(mean_obs, dim=-1) - observation.squeeze(-1))**2
        except:
            return torch.full((120, 32), torch.nan, device = state.device)



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

class check_each_epoch():
    def __init__(self, pvmc, mean, sd, dataset, experiment):
        self.model = pvmc
        self.count = 0
        self.mean = mean
        self.sd = sd
        self.dataset = dataset
        self.experiment = experiment

    def __call__(self):
        if self.count % 5 == 0:
            plot_paths(self.model.SSM, 360, self.mean, self.sd, self.dataset, f"{self.experiment}_3", self.count+1)
        self.model.beta_dynamic = 0.05 + 0.01*self.count
        self.count += 1

def make_trainer_routine(alg, dataset, dummy_model = None, mean= 0, sd=0, experiment="pvmc"):
    params = ParameterSet(alg)
    optim = torch.optim.Adam(params, lr=1e-4)
    if experiment == "TCVAE":
        runner = TCVAERun(alg)
    else:
        runner = VanillaPydpfRun(alg)

    stage = TrainingStage(runner, dataset, None, None, optim, ["observation", "series_metadata"], run_on_epoch=check_each_epoch(alg, mean, sd, dataset, experiment))
    trainer = Trainer(alg, stages=[stage])
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


def make_alg(exper):
    match exper:
        case "PVMC":
            return make_pvmc(4)
        case "DMM":
            return make_dmm(4)
        case "VAE":
            return make_pvmc(4)
        case "Soft":
            return make_soft_dpf(4)
        case "TCVAE":
            return make_tcvae(4)

def make_info(exper, dataset, SSM):
    match exper:
        case "PVMC":
            return make_run_info(dataset, SSM)
        case "DMM":
            return make_run_info(dataset, SSM, True)
        case "VAE":
            return make_run_info(dataset, SSM, True)
        case "Soft":
            return make_filter_run_info(dataset, SSM)
        case "TCVAE":
            return make_tcvae_run_info(dataset)

if __name__ == "__main__":
    for experiment in experiments:
        dataset = make_dataset()
        dataset.apply(to_log_ret)
        obs = dataset.observation
        mean_obs = torch.mean(obs, dim=(0,1)).squeeze()
        sd_obs = torch.sqrt(torch.var(obs, dim=(0,1)).squeeze())
        dataset.apply(lambda observation, **data: (observation - mean_obs)/sd_obs)
        obs = dataset.observation
        alg = make_alg(experiment)
        if not (experiment == "Soft" or experiment == "TCVAE"):
            alg.beta_dynamic = 0.05
        if experiment == "TCVAE":
            ssm = alg
            run_info = [make_info(experiment, dataset, None)]
        else:
            ssm = alg.SSM
            ps = []
            for p in alg.proposal.parameters():
                ps.append(p.flatten())
            ps = torch.cat(ps, dim=0)
            print(ps.shape)
            run_info = [make_info(experiment, dataset, ssm)]
        trainer_routine = make_trainer_routine(alg, dataset, dummy_model=None, mean=mean_obs, sd=sd_obs, experiment=experiment)
        trainer_routine.fit("run", run_info, False)
        alg.update()
        plot_paths(ssm, 360, mean_obs, sd_obs, dataset, f"{experiment}_3", 100)

