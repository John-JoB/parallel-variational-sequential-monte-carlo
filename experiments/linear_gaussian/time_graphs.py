from time import time

import pandas as pd
import pydpf
import torch
from pydpf import FilteringModel, LogLikelihoodFactors, MarginalStopGradientDPF

from diffusion_DPF import DiffusionDPF
from experiments.linear_gaussian.kalman_stage import Kalman_mean, Kalman_log_likelihood_factors, Kalman_MSE
from models.linear_gaussian import learned_model, true_model
from experiments.common.testing import Test_Runner
from parallel_smoother_new import ParallelSmoother
from smoother_outputs import MSE, MarginalSmoothingMean, dSMC_ELBO
from pathlib import Path
from experiments.common.training import TrainingStage, Trainer, VanillaPydpfRun
from two_filter_smoother import TwoFilter
from dSMC import dSMC
from mdps import MDPS
from models.lokta_volterra.extra_components_for_mdps import CombinationWeights
import numpy as np

experiments = ["dSMC", "PVMC", "Soft", "TFS", "Diffusion", "Stop-Grad"]
time_extent = 500
noise_sd = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def make_true_SSM(dx, dy, generator):
    prior_model = true_model.GaussianPrior(dx, generator)
    dynamic_model = true_model.GaussianDynamic(dx, generator)
    obs_model = true_model.GaussianObservation(dx, dy, generator)
    return prior_model, dynamic_model, obs_model

def make_reverse_SSM(prior_model, dynamic_model):
    auxiliary_model = true_model.InformationAuxiliary(prior_model, dynamic_model, time_extent)
    prior_model = true_model.InformationPrior(auxiliary_model, time_extent)
    dynamic_model = true_model.InformationDynamic(auxiliary_model, time_extent)
    return prior_model, dynamic_model

def make_pvmc(SSM, generator):
    return ParallelSmoother(learned_model.proposal_model(5, 5, time_extent, generator), SSM)

def make_tfs(SSM, generator):
    ipm, idm = make_reverse_SSM(SSM.prior_model.dist, SSM.dynamic_model.dist)
    issm = FilteringModel(prior_model=ipm, dynamic_model=idm, observation_model=SSM.observation_model)
    return TwoFilter(SSM, issm, generator)

def make_dsmc(SSM, generator, prop):
    return dSMC(prop, SSM, generator)

def make_soft_dpf(SSM, generator):
    return pydpf.SoftDPF(SSM, resampling_generator=generator)

def make_diffusion_dpf(SSM, generator):
    return DiffusionDPF(SSM, generator)

def make_stop_grad_dpf(SSM, generator):
    return MarginalStopGradientDPF(SSM, generator)

def make_kernel(generator, starting_bandwith_pos):
    device = generator.device
    gaussian_kernel = pydpf.MultivariateGaussian(torch.zeros(5, device=device), torch.nn.Parameter(torch.eye(5, device=device) * starting_bandwith_pos, requires_grad=True), diagonal_cov=True, generator=generator)
    syst_resampler = pydpf.SystematicResampler(generator)
    kernel_mixture = pydpf.KernelMixture(gaussian_kernel, generator, syst_resampler)
    return kernel_mixture


def make_mdps(SSM, generator):
    backward_prior = learned_model.GaussianPrior(5, generator)
    backward_dynamic = learned_model.GaussianDynamic(5, generator)
    backward_ssm = pydpf.FilteringModel(prior_model=backward_prior, dynamic_model=backward_dynamic, observation_model=SSM.observation_model)
    combination_model = CombinationWeights(device=device,dim=5)
    forward_kernel = make_kernel(generator, 0.5)
    backward_kernel = make_kernel(generator, 0.5)
    forward_res_kernel = make_kernel(generator, 0.1)
    backward_res_kernel = make_kernel(generator, 0.1)
    return MDPS(SSM, backward_ssm, combination_model, forward_kernel, backward_kernel, forward_res_kernel, backward_res_kernel)

def get_data(folder, dx, dy):
    train = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-train.csv", state_prefix="state", device=torch.device("cuda:0"))
    validation = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-validation.csv", state_prefix="state", device=torch.device("cuda:0"))
    test = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-test.csv", state_prefix="state", device=torch.device("cuda:0"))
    return train, validation, test

def get_only_test_data(folder, dx, dy):
    test = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-test.csv", state_prefix="state", device=torch.device("cuda:0"))
    return test



def make_info(dataset, te):
    run_info = {"return": {"mean" : "kalman_mean", "likelihood" : "likelihood", "MSE" : "MSE", "time" : "time"},
                "output_function": {"kalman_mean": Kalman_mean(), "likelihood": Kalman_log_likelihood_factors(), "MSE": Kalman_MSE()},
                "shuffle": False,
                "batch_size": 64,
                "device": "cuda:0",
                "time_extent": te,
                "collate_fn": dataset.collate}
    return run_info

def make_pvmc_info(dataset, te):
    run_info = {"return": {"mean" : "mean", "likelihood" : "likelihood", "MSE" : "MSE", "time" : "time"},
                "n_particles": 32,
                "shuffle": False,
                "batch_size": 16,
                "collate_fn": dataset.collate,
                "time_extent": te,
                "device": "cuda:0",
                "output_function": {"mean": MarginalSmoothingMean(), "likelihood" :  dSMC_ELBO(),  "MSE": MSE()}}
    return run_info


def make_trainer_info(train_set, validation_set, test_set):
    train_info = {"n_particles": 32,
                  "batch_size": 32,
                  "collate_fn": train_set.collate,
                  "time_extent": 500,
                  "output_function": {"ELBO": dSMC_ELBO()}}
    validation_info = {"n_particles": 32,
                  "batch_size": 128,
                  "collate_fn": validation_set.collate,
                  "time_extent": 50,
                  "output_function": {"likelihood": dSMC_ELBO()}}
    test_info = {"n_particles": 32,
                  "batch_size": 128,
                  "collate_fn": test_set.collate,
                  "time_extent": 50,
                  "output_function": {"likelihood": dSMC_ELBO()}}
    info = {"train": train_info,
            "validation": validation_info,
            "test": test_info,
            "loss": f"- time_average.ELBO / {time_extent + 1}",
            "print_each_epoch": {"Validation ELBO": "validation.mean.time_average.likelihood", "Train ELBO": "train.mean.time_average.ELBO"},
            "epochs": 10,
            "device": device,
            "target": f"-validation.mean.time_average.likelihood",
            "print": {"Test ELBO": "test.mean.time_average.likelihood"}
            }
    return info

def make_trainer_routine(model, train_set, validation_set, test_set):
    runner = VanillaPydpfRun(model)
    optim = torch.optim.AdamW(model.parameters())
    stage = TrainingStage(runner, train_set, validation_set, test_set, optim, ["ground_truth", "observation"])
    trainer = Trainer(model, stages=[stage])
    return trainer

def get_trainer(experiment, model, train_set, validation_set, test_set):
    if not experiment == "PVMC" and not experiment == "MDPS":
        return None
    runner = VanillaPydpfRun(model)
    optim = torch.optim.AdamW(model.parameters())
    stage = TrainingStage(runner, train_set, validation_set, test_set, optim, ["ground_truth", "observation"])
    trainer = Trainer(model, stages=[stage])
    return trainer

def get_run(model, dataset):
    func = VanillaPydpfRun(model)
    return Test_Runner(model, run_func=func, dataset=dataset, data_order=["ground_truth", "observation"])


def get_model(experiment, SSM, generator, prop):
    match experiment:
        case "PVMC":
            return make_pvmc(SSM, generator)
        case "MDPS":
            return make_mdps(SSM, generator)
        case "TFS":
            return make_tfs(SSM, generator)
        case "Soft":
            return make_soft_dpf(SSM, generator)
        case "Diffusion":
            return  make_diffusion_dpf(SSM, generator)
        case "dSMC":
            return make_dsmc(SSM, generator, prop)
        case "Stop-Grad":
            return make_stop_grad_dpf(SSM, generator)

def get_output_function(experiment):
    l = LogLikelihoodFactors()
    DSMC_ELBO = lambda **outputs: torch.sum(l(**outputs), dim=0)
    match experiment:
        case "PVMC":
            return {"ELBO": dSMC_ELBO(), "EST": MarginalSmoothingMean()}
        case "MDPS":
            return {"ELBO": dSMC_ELBO(), "EST": MarginalSmoothingMean()}
        case "TFS":
            return {"ELBO": dSMC_ELBO(), "EST": MarginalSmoothingMean()}
        case "dSMC":
            return {"ELBO": dSMC_ELBO(), "EST": MarginalSmoothingMean()}
        case "Soft":
            return DSMC_ELBO
        case "Diffusion":
            return DSMC_ELBO
        case "Stop-Grad":
            return DSMC_ELBO

def simple_test_loop(experiment, model, dataset, time_extent, repeats):
    loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate, batch_size=1, shuffle=False)
    output_fun = get_output_function(experiment)
    for p in model.SSM.parameters():
        p.requires_grad = True
    times = []
    backward_time = []
    grads = []
    for state, observation in loader:
        observation = observation[:, 0:1].expand(-1, repeats*16, -1)
        for r in range(repeats):
            model.update()
            obs = observation[:, r*16:(r+1)*16]
            time_zero = time()
            output = model(observation=obs, aggregation_function=output_fun, time_extent=time_extent, n_particles=32)
            if isinstance(output, dict):
                output = output["ELBO"]
            times.append(time() - time_zero)
            loss = -torch.mean(output/(time_extent+1))
            time_zero = time()
            loss.backward()
            backward_time.append(time() - time_zero)
            grad_inside = []
            for p in model.SSM.parameters():
               grad_inside.append(p.grad.flatten())
            grad_inside = torch.cat(grad_inside)
            grads.append(grad_inside)
        break
    grads = torch.stack(grads)
    cov = torch.cov(grads.T)
    print(f"Experiment: {experiment}, {time_extent}")
    cov_radius = torch.max(torch.abs(torch.linalg.eigvals(cov))).item()
    print(f"Time {np.mean(np.array(times))}")
    print(f"B time {np.mean(np.array(backward_time))}")
    print(f"Cov radius: {cov_radius}")
    return np.mean(np.array(times)), np.mean(np.array(backward_time)), cov_radius




if __name__ == '__main__':
    pm, dm, om = make_true_SSM(5, 5, torch.Generator(device))
    t_dataset, v_dataset, dataset = get_data(Path("./experiments/linear_gaussian/data/"), 5, 5)
    for p in pm.parameters():
        p += torch.randn_like(p)*noise_sd
    for p in om.parameters():
        p += torch.randn_like(p)*noise_sd
    for p in dm.parameters():
        p += torch.randn_like(p)*noise_sd
    SSM = pydpf.FilteringModel(dynamic_model=dm, prior_model=pm, observation_model=om)
    models = {}
    for experiment in experiments:
        if experiment != "dSMC":
            models[experiment] = get_model(experiment, SSM, torch.Generator(device).manual_seed(0), None)
    if "dSMC" in experiments:
        models["dSMC"] = get_model("dSMC", SSM, torch.Generator(device).manual_seed(0), models["PVMC"].proposal)
    for experiment in experiments:
        train = get_trainer(experiment, models[experiment], t_dataset, v_dataset, dataset)
        if train is not None:
            info = make_trainer_info(t_dataset, v_dataset, dataset)
            train.fit("_", [info])
    times = np.arange(50) * 10 + 10
    results_df = pd.DataFrame(columns=["Method", "Time Extent", "Forward", "Backward", "Variance"])
    for experiment in experiments:
        for time_e in times:
            f, b, v = simple_test_loop(experiment, models[experiment], dataset, time_e, 20)
            results_df.loc[len(results_df)] = {"Method": experiment, "Time Extent": time_e, "Forward": f, "Backward": b, "Variance": v}
    results_df.to_csv(Path("./experiments/linear_gaussian/results_timing.csv"))