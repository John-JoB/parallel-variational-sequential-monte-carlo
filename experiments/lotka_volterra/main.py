import torch
from models.lokta_volterra import learned_model as lm
from models.lokta_volterra import true_model as tm
from models.lokta_volterra import extra_components_for_mdps as ec
import pydpf
from parallel_smoother_new import ParallelSmoother
from pathlib import Path
from experiments.common.parameter_set import ParameterSet
from experiments.common.training import TrainingStage, VanillaPydpfRun, Trainer
from experiments.common.testing import Test_Runner
from smoother_outputs import MSE, dSMC_ELBO, VAE_ELBO
from mdps import MDPS
import numpy as np
from time import time
from math import floor
import pandas as pd
from diffusion_DPF import DiffusionDPF
import traceback

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder = Path("./experiments/lotka_volterra/data/")
results_folder = Path("./experiments/lotka_volterra/results/")
table_results = results_folder / "table_results.csv"
raw_results = results_folder / "raw_results.csv"
experiments = ["PVMC", "Soft", "Stop-grad", "MDPS", "DSSM"]
repeats = 20
epochs = 100


def sliced_wasserstein_2(state_x, state_y, weight_x, weight_y, n_projections=512):
    B, N, d = state_x.shape
    device = state_x.device

    theta = torch.randn(n_projections, d, device=device)
    theta = theta / torch.linalg.norm(theta, dim=-1, keepdim=True)
    x_proj = torch.einsum("bnd,kd->bnk", state_x, theta)
    y_proj = torch.einsum("bnd,kd->bnk", state_y, theta)
    x_sort, idx_x = torch.sort(x_proj, dim=1)
    y_sort, idx_y = torch.sort(y_proj, dim=1)
    a_sort = torch.gather(weight_x.unsqueeze(-1).expand(-1, -1, n_projections),1, idx_x)
    b_sort = torch.gather(weight_y.unsqueeze(-1).expand(-1, -1, n_projections),1, idx_y)
    Fx = torch.cumsum(a_sort, dim=1)
    Fy = torch.cumsum(b_sort, dim=1)
    w2_per_proj = torch.sum((Fx - Fy).abs() * (x_sort - y_sort) ** 2, dim=1)
    return w2_per_proj.mean(dim=-1)


def make_SSM(generator):
    prior_model = tm.TruePriorModel(device=device)
    observation_model = tm.TrueObservationModel()
    dynamic_model = lm.LearnedDynamicModel(depth = 6, hidden_dim=32, device=device, generator=generator)
    return pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)


def make_true_SSM(generator):
    prior_model = tm.TruePriorModel(device=device)
    observation_model = tm.TrueObservationModel()
    dynamic_model = tm.TrueDynamicModel(device=device)
    return pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)

#Our model
def make_pvmc(SSM, generator):
    proposal_model = lm.proposal_model(device=device, generator=generator)
    return ParallelSmoother(proposal_model, SSM, clip_likelihoods_for_stability=True), 1 #, control_net= FCNN(4, 1, 32, "swish", "id", 6, device)), 1



#PF
def make_pf(SSM, generator):
    return pydpf.DPF(SSM, resampling_generator=generator)

#Baselines

#Stop-gradient DPF
def make_sg_dpf(SSM, generator):
    return pydpf.MarginalStopGradientDPF(SSM, resampling_generator=generator), 0

#Soft DPF
def make_soft_dpf(SSM, generator):
    return pydpf.SoftDPF(SSM, resampling_generator=generator), 0

#Diffusion DPF
def make_diff_dpf(SSM, generator):
    return DiffusionDPF(SSM, resampling_generator=generator, use_modified=False), 0

#Diffusion DPF
def make_diff_dpf_mod(SSM, generator):
    return DiffusionDPF(SSM, resampling_generator=generator, use_modified=True), 0

#DSSM
def make_DSSM(SSM, generator):
    proposal_model = lm.proposal_model(device=device, generator=generator)
    return ParallelSmoother(proposal_model, SSM, clip_likelihoods_for_stability=True), 3

#MDPS
def make_kernel(generator, starting_bandwith_pos):
    device = generator.device
    gaussian_kernel = pydpf.MultivariateGaussian(torch.zeros(2, device=device), torch.nn.Parameter(torch.eye(2, device=device) * starting_bandwith_pos, requires_grad=True), diagonal_cov=True, generator=generator)
    syst_resampler = pydpf.SystematicResampler(generator)
    kernel_mixture = pydpf.KernelMixture(gaussian_kernel, generator, syst_resampler)
    return kernel_mixture


def make_mdps(SSM, generator):
    backward_prior = ec.backward_prior(generator)
    backward_dynamic = lm.LearnedDynamicModel(depth = 6, hidden_dim=32, device=device, generator=generator)
    backward_ssm = pydpf.FilteringModel(prior_model=backward_prior, dynamic_model=backward_dynamic, observation_model=SSM.observation_model)
    combination_model = ec.CombinationWeights(device=device)
    forward_kernel = make_kernel(generator, 0.5)
    backward_kernel = make_kernel(generator, 0.5)
    forward_res_kernel = make_kernel(generator, 0.1)
    backward_res_kernel = make_kernel(generator, 0.1)
    return MDPS(SSM, backward_ssm, combination_model, forward_kernel, backward_kernel, forward_res_kernel, backward_res_kernel), 2



def get_data(folder):
    train = pydpf.StateSpaceDataset(folder / f"train.csv", state_prefix="state")
    validation = pydpf.StateSpaceDataset(folder / f"validation.csv", state_prefix="state")
    test = pydpf.StateSpaceDataset(folder / f"test.csv", state_prefix="state")
    return train, validation, test


class FilterMSE(pydpf.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state, ground_truth, weight, **data):
        return torch.sum((torch.sum(torch.exp(weight).unsqueeze(-1) * state, dim= 1) - ground_truth)**2, dim=-1)

def make_smoother_tvt_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent):
    train_info = {"n_particles": n_particles[0],
                  "batch_size": batch_size[0],
                  "collate_fn": train_set.collate,
                  "time_extent": time_extent[0],
                  "output_function": {"ELBO": dSMC_ELBO(), "MSE": MSE()}}
    validation_info = {"n_particles": n_particles[1],
                  "batch_size": batch_size[1],
                  "collate_fn": validation_set.collate,
                  "time_extent": time_extent[1],
                  "output_function": {"ELBO": dSMC_ELBO(), "MSE": MSE()}}
    test_info = {"n_particles": n_particles[2],
                  "batch_size": batch_size[2],
                  "collate_fn": test_set.collate,
                  "time_extent": time_extent[2],
                 "output_function": {"ELBO": dSMC_ELBO(), "MSE": MSE()}}
    return train_info, validation_info, test_info

def make_dssm_tvt_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent):
    train_info = {"n_particles": n_particles[0],
                  "batch_size": batch_size[0],
                  "collate_fn": train_set.collate,
                  "time_extent": time_extent[0],
                  "output_function": {"ELBO": VAE_ELBO(), "MSE": MSE()}}
    validation_info = {"n_particles": n_particles[1],
                  "batch_size": batch_size[1],
                  "collate_fn": validation_set.collate,
                  "time_extent": time_extent[1],
                  "output_function": {"ELBO": VAE_ELBO(), "MSE": MSE()}}
    test_info = {"n_particles": n_particles[2],
                  "batch_size": batch_size[2],
                  "collate_fn": test_set.collate,
                  "time_extent": time_extent[2],
                 "output_function": {"ELBO": VAE_ELBO(), "MSE": MSE()}}
    return train_info, validation_info, test_info

def make_mdps_tvt_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent, generator):
    output_kernel = make_kernel(generator, 0.5)
    output_knll = pydpf.NegLogDataLikelihood_Loss(output_kernel)
    train_info = {"n_particles": n_particles[0],
                  "batch_size": batch_size[0],
                  "collate_fn": train_set.collate,
                  "time_extent": time_extent[0],
                  "output_function": {"ELBO": dSMC_ELBO(), "MSE": MSE()}}
    validation_info = {"n_particles": n_particles[1],
                  "batch_size": batch_size[1],
                  "collate_fn": validation_set.collate,
                  "time_extent": time_extent[1],
                  "output_function": {"ELBO": dSMC_ELBO(), "MSE": MSE()}}
    test_info = {"n_particles": n_particles[2],
                  "batch_size": batch_size[2],
                  "collate_fn": test_set.collate,
                  "time_extent": time_extent[2],
                 "output_function": {"ELBO": dSMC_ELBO(), "MSE": MSE()}}
    return train_info, validation_info, test_info

def make_filter_tvt_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent):
    train_info = {"n_particles": n_particles[0],
                  "batch_size": batch_size[0],
                  "collate_fn": train_set.collate,
                  "time_extent": time_extent[0],
                  "output_function": {"ELBO": pydpf.LogLikelihoodFactors(), "MSE": FilterMSE()}}
    validation_info = {"n_particles": n_particles[1],
                  "batch_size": batch_size[1],
                  "collate_fn": validation_set.collate,
                  "time_extent": time_extent[1],
                  "output_function": {"ELBO": pydpf.LogLikelihoodFactors(), "MSE": FilterMSE()}}
    test_info = {"n_particles": n_particles[2],
                  "batch_size": batch_size[2],
                  "collate_fn": test_set.collate,
                  "time_extent": time_extent[2],
                 "output_function": {"ELBO": pydpf.LogLikelihoodFactors(), "MSE": FilterMSE()}}
    return train_info, validation_info, test_info

def make_smoother_info(train_info, validation_info, test_info, epochs, device):
    info =  {"train": train_info,
        "validation": validation_info,
        "test": test_info,
        "loss": f"- time_average.ELBO/257 + time_average.MSE",
        "print_each_epoch": {"Train ELBO": "train.mean.time_average.ELBO", "Train MSE": "train.mean.time_average.MSE", "Validation ELBO": "validation.mean.time_average.ELBO", "Validation MSE": "validation.mean.time_average.MSE"},
        "epochs": epochs,
        "device": device,
        "target": f"validation.mean.time_average.ELBO",
        "print": {"Test ELBO": "test.mean.time_average.ELBO", "Test MSE": "test.mean.time_average.MSE"},
        "return": {"MSE": "test.mean.time_average.MSE"},
        "use_control_variate": False
        }
    return info

def make_mdps_info(train_info, validation_info, test_info, epochs, device):
    info =  {"train": train_info,
        "validation": validation_info,
        "test": test_info,
        "loss": f"-time_average.ELBO/257 + time_average.MSE",
        "print_each_epoch": {"Train ELBO": "train.mean.time_average.ELBO", "Train MSE": "train.mean.time_average.MSE", "Validation ELBO": "validation.mean.time_average.ELBO", "Validation MSE": "validation.mean.time_average.MSE"},
        "epochs": epochs,
        "device": device,
        "target": f"validation.mean.time_average.MSE",
        "print": {"Test ELBO": "test.mean.time_average.ELBO", "Test MSE": "test.mean.time_average.MSE"},
        "return": {"MSE": "test.mean.time_average.MSE"}
        }
    return info


def make_filter_info(train_info, validation_info, test_info, epochs, device):
    info =  {"train": train_info,
        "validation": validation_info,
        "test": test_info,
        "loss": f"- time_average.ELBO + time_average.MSE",
        "print_each_epoch": {"Train ELBO": "train.mean.time_average.ELBO", "Train MSE": "train.mean.time_average.MSE", "Validation ELBO": "validation.mean.time_average.ELBO", "Validation MSE": "validation.mean.time_average.MSE"},
        "epochs": epochs,
        "device": device,
        "target": f"validation.mean.time_average.MSE",
        "print": {"Test ELBO": "test.mean.time_average.ELBO", "Test MSE": "test.mean.time_average.MSE"},
        "return": {"MSE": "test.mean.time_average.MSE"}
        }
    return info

def make_test_info(dataset):
    run_info = {"return": {"mean" : "mean", "likelihood" : "likelihood", "MSE" : "MSE", "time" : "time", "state": "State", "weight": "Weight"},
                "n_particles": 1000,
                "shuffle": False,
                "batch_size": 64,
                "collate_fn": dataset.collate,
                "time_extent": 256,
                "device": "cuda:0",
                "output_function": {"likelihood" :  pydpf.LogLikelihoodFactors(),  "MSE": FilterMSE(), "State": pydpf.State(), "Weight": pydpf.Weight()}}
    return run_info

def make_trainer_routine(model, train_dataset, validation_dataset, test_dataset):
    params = ParameterSet(model)
    optim = torch.optim.Adam(params)
    runner = VanillaPydpfRun(model)
    stage = TrainingStage(runner, train_dataset, validation_dataset, test_dataset, optim, ["ground_truth", "observation"])
    trainer = Trainer(model, stages=[stage])
    return trainer

def make_test(dpf, dataset):
    run_func = VanillaPydpfRun(dpf)
    return Test_Runner(dpf, run_func=run_func, dataset=dataset, data_order=["ground_truth", "observation"])

def to_interval(data_list):
    data_arr = np.array(data_list)
    return f"{data_arr.mean().item()} +- {data_arr.std(ddof=1).item()}"

def to_minutes_and_seconds(time):
    mins = floor(time/60)
    secs = time - mins*60
    return f"{mins}:{round(secs)}"


def to_minutes_interval(time_list):
    time_arr = np.array(time_list)
    mean = time_arr.mean().item()
    sd = time_arr.std(ddof=1).item()
    return f"{to_minutes_and_seconds(mean)} +- {to_minutes_and_seconds(sd)}"

def increase_to_size(data_list, size):
    while len(data_list) < size:
        data_list.append(np.nan)
    return data_list

def train_and_test_alg(create_model, experiment):
    train_set, validation_set, test_set = get_data(folder)
    n_particles = [32] * 3
    batch_size = [16] * 3
    time_extent = [256] * 3
    generator = torch.Generator(device=device).manual_seed(0)
    SSM = make_SSM(generator)

    _, is_smoother = create_model(SSM, generator)

    if is_smoother == 1:
        train_info, validation_info, test_info  = make_smoother_tvt_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent)
        info = make_smoother_info(train_info, validation_info, test_info, epochs, device)
    elif is_smoother == 2:
        train_info, validation_info, test_info = make_mdps_tvt_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent, generator)
        info = make_mdps_info(train_info, validation_info, test_info, epochs, device)
    elif is_smoother == 3:
        train_info, validation_info, test_info = make_dssm_tvt_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent)
        info = make_smoother_info(train_info, validation_info, test_info, epochs, device)
    else:
        train_info, validation_info, test_info = make_filter_tvt_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent)
        info = make_filter_info(train_info, validation_info, test_info, epochs, device)

    pf_test_info = make_test_info(test_set)

    n_fails_self = 0
    n_fails_pf = 0
    self_MSE_list = []
    pf_MSE_list = []
    time_list = []
    W2_list = []
    for i in range(repeats):
        generator = torch.Generator(device=device).manual_seed(i * 10)
        SSM = make_SSM(generator)
        ps = []
        for n, p in SSM.named_parameters():
            ps.append(p.flatten())
        ps = torch.cat(ps)
        a = ps.shape[0]
        print(ps.shape)
        true_SSM = make_true_SSM(generator)
        model, _ = create_model(SSM, generator)

        pf = make_pf(SSM, generator)
        ps = []
        for n, p in model.named_parameters():
            ps.append(p.flatten())
        ps = torch.cat(ps)
        print(ps.shape[0] - 2*a)
        true_pf = make_pf(true_SSM, generator)
        trainer = make_trainer_routine(model, train_set, validation_set, test_set)
        start_time = time()
        try:
            self_test_mse = trainer.fit("run", [info])["MSE"].item()
        except Exception as e:
            print(traceback.format_exc())
            self_test_mse = np.nan

        if np.isnan(self_test_mse) or np.isinf(self_test_mse):
            n_fails_self += 1
            n_fails_pf += 1
            continue
        time_list.append(time() - start_time)
        pf.update()
        test = make_test(pf, test_set)
        test_true = make_test(true_pf, test_set)

        try:
            pf_res = test.test(f"Repeat {i + 1}", pf_test_info)
            pf_mse = pf_res["MSE"].mean().item()
            true_res = test_true.test(f"Repeat {i + 1}", pf_test_info)
            print(f"True MSE {true_res['MSE'].mean().item()}")
            pf_state = pf_res["state"][-1]
            pf_weight = np.exp(pf_res["weight"][-1])
            true_state = true_res["state"][-1]
            true_weight = np.exp(true_res["weight"][-1])

            W2 = sliced_wasserstein_2(torch.tensor(pf_state, device=device),
                                      torch.tensor(true_state, device=device),
                                      torch.tensor(pf_weight, device=device),
                                      torch.tensor(true_weight, device=device),
                                     512).mean().item()
        except Exception as e:
            print(e)
            pf_mse = np.nan
            W2 = np.nan
        self_MSE_list.append(self_test_mse)
        if np.isnan(pf_mse) or np.isinf(pf_mse):
            n_fails_pf += 1
            continue
        print(pf_mse)
        print(W2)
        pf_MSE_list.append(pf_mse)
        W2_list.append(W2)
    table_results_df = pd.read_csv(table_results, index_col=0)
    raw_results_df = pd.read_csv(raw_results, index_col=0)
    table_results_experiment = [to_interval(self_MSE_list), to_interval(pf_MSE_list), to_minutes_interval(time_list), to_interval(W2_list)]
    table_results_df.loc[experiment] = table_results_experiment
    print(table_results_df)
    table_results_df.to_csv(table_results)
    raw_results_df.loc[f"self_{experiment}"] = increase_to_size(self_MSE_list, repeats)
    raw_results_df.loc[f"pf_{experiment}"] = increase_to_size(pf_MSE_list, repeats)
    raw_results_df.loc[f"time_{experiment}"] = increase_to_size(time_list, repeats)
    raw_results_df.loc[f"W2_{experiment}"] = increase_to_size(W2_list, repeats)
    raw_results_df.to_csv(raw_results)

def add_experiment(experiment_name):
    match experiment_name:
        case "PVMC":
            return make_pvmc
        case "Soft":
            return make_soft_dpf
        case "Stop-grad":
            return make_sg_dpf
        case "MDPS":
            return make_mdps
        case "Diffusion":
            return make_diff_dpf
        case "Diffusion-Mod":
            return make_diff_dpf_mod
        case "DSSM":
            return make_DSSM





if __name__ == "__main__":
    if not table_results.exists():
        temp_df = pd.DataFrame(index=pd.Index(experiments, name="Method"), columns=["Self MSE", "PF MSE", "Time", "W2"], dtype=str)
        temp_df.to_csv(table_results)
    if not raw_results.exists():
        index = [f"self_{e}" for e in experiments] + [f"pf_{e}" for e in experiments] + [f"time_{e}" for e in experiments]
        temp_df = pd.DataFrame(index=pd.Index(index, name="Method"), columns=[f"run_{i}" for i in range(repeats)])
        temp_df.to_csv(raw_results)
    else:
        temp_df = pd.read_csv(raw_results)
        if len(temp_df.columns) - 1 != repeats:
            raise ValueError("The raw results df was run with a different number of repeats than the current run")

    for experiment in experiments:
        make_model = add_experiment(experiment)
        train_and_test_alg(make_model, experiment)




