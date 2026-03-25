import argparse
from time import time

import pandas as pd
import pydpf
import torch
from pydpf import FilteringModel

from experiments.linear_gaussian.kalman_stage import KalmanRun, Kalman_mean, Kalman_log_likelihood_factors, Kalman_MSE, Kalman_covariance
from parallel_kalman import ParallelKalmanSmoother, ParallelKalmanFilter
from models.linear_gaussian import learned_model, true_model
from experiments.common.testing import Test_Runner
from parallel_smoother_new import ParallelSmoother
from proposal_to_output import ProposalRunner
from truncated_pvmc import Truncated
from smoother_outputs import MSE, MarginalSmoothingMean, dSMC_ELBO, State, Weight
from pathlib import Path
from experiments.common.training import TrainingStage, Trainer, VanillaPydpfRun
from two_filter_smoother import TwoFilter
from dSMC import dSMC

import numpy as np

time_extent = 500

def position_error(pvmc_state, smoother_state):
    return np.mean(np.sum((pvmc_state - smoother_state)**2, axis = -1))

def sliced_w2_empirical_gaussian_vec(
    x, w, m, Sigma, n_proj=100, eps=1e-12
):
    B, N, d = x.shape
    x = torch.tensor(x, device = "cuda:0")
    w = torch.tensor(w, device = "cuda:0")
    m = torch.tensor(m, device = "cuda:0")
    Sigma = torch.tensor(Sigma, device = "cuda:0")
    device = x.device

    # normalize weights
    w = w / (w.sum(dim=1, keepdim=True) + eps)

    # sample projections (shared across batch for variance reduction)
    theta = torch.randn(n_proj, d, device=device)
    theta = theta / (theta.norm(dim=1, keepdim=True) + eps)
    # (P, d)

    # project particles
    # (B, P, N)
    proj_x = torch.einsum('pd,bnd->bpn', theta, x)

    # sort along particle axis
    vals, idx = torch.sort(proj_x, dim=2)  # (B, P, N)

    # reorder weights accordingly
    w_expanded = w.unsqueeze(1).expand(B, n_proj, N)
    w_sorted = torch.gather(w_expanded, 2, idx)

    # cumulative weights
    S = torch.cumsum(w_sorted, dim=2)
    S_prev = torch.cat(
        [torch.zeros(B, n_proj, 1, device=device), S[:, :, :-1]],
        dim=2
    )

    # midpoint rule
    u_mid = 0.5 * (S + S_prev)
    u_mid = u_mid.clamp(eps, 1 - eps)

    # projected Gaussian parameters
    # mean: (B, P)
    m_theta = torch.einsum('pd,bd->bp', theta, m)

    # variance: θᵀ Σ θ
    # (B, P)
    Sigma_theta = torch.einsum('pd,bdk,pk->bp', theta, Sigma, theta)
    sigma_theta = torch.sqrt(Sigma_theta + eps)

    # quantiles
    normal = torch.distributions.Normal(0.0, 1.0)
    z = normal.icdf(u_mid)  # (B, P, N)

    y = m_theta.unsqueeze(2) + sigma_theta.unsqueeze(2) * z

    # 1D Wasserstein per projection
    w2 = torch.sum(w_sorted * (vals - y) ** 2, dim=2)  # (B, P)

    # average over projections
    return w2.mean().item()  # (B,)

def likelihood_error(pvmc_l, smoother_l):
    return np.mean((1 - np.exp(pvmc_l - smoother_l))**2)


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


def make_kalman_proposal(dx, dy, generator):
    return

def make_pvmc(dx, dy, generator):
    pm, dm, om = make_true_SSM(dx, dy, generator)
    ssm = FilteringModel(prior_model=pm, dynamic_model=dm, observation_model=om)
    return ParallelSmoother(learned_model.KalmanProposal(dx, dy, generator, True, False), ssm)

def make_learned_pvmc(dx, dy, generator):
    pm, dm, om = make_true_SSM(dx, dy, generator)
    ssm = FilteringModel(prior_model=pm, dynamic_model=dm, observation_model=om)
    return ParallelSmoother(learned_model.proposal_model(dx, dy, time_extent, generator), ssm)

def make_tfs(dx, dy, generator):
    pm, dm, om = make_true_SSM(dx, dy, generator)
    ssm = FilteringModel(prior_model=pm, dynamic_model=dm, observation_model=om)
    ipm, idm = make_reverse_SSM(pm, dm)
    issm = FilteringModel(prior_model=ipm, dynamic_model=idm, observation_model=om)
    return TwoFilter(ssm, issm, generator)

def get_data(folder, dx, dy):
    train = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-train.csv", state_prefix="state", device=torch.device("cuda:0"))
    validation = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-validation.csv", state_prefix="state", device=torch.device("cuda:0"))
    test = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-test.csv", state_prefix="state", device=torch.device("cuda:0"))
    return train, validation, test

def get_only_test_data(folder, dx, dy):
    test = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-test.csv", state_prefix="state", device=torch.device("cuda:0"))
    return test

def make_filters(prior_model, dynamic_model, observation_model):
    kalman_filter = pydpf.KalmanFilter(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)
    p_kalman_filter = ParallelKalmanFilter(prior_model, dynamic_model, observation_model)
    p_kalman_smoother = ParallelKalmanSmoother(prior_model, dynamic_model, observation_model)
    return kalman_filter, p_kalman_filter, p_kalman_smoother

def make_dsmc(dx, dy, generator):
    pm, dm, om = make_true_SSM(dx, dy, generator)
    ssm = FilteringModel(prior_model=pm, dynamic_model=dm, observation_model=om)
    return dSMC(learned_model.KalmanProposal(dx, dy, generator, True, True), ssm, generator)

def make_pvmc_test_run(pvmc, dataset):
    pvmc_run = VanillaPydpfRun(pvmc)
    return Test_Runner(pvmc, run_func=pvmc_run, dataset=dataset, data_order=["ground_truth", "observation"])

def make_test_runs(kalman_filter, p_kalman_filter, p_kalman_smoother, pvmc, tfs, dataset):
    kalman_run = KalmanRun(kalman_filter)
    p_kalman_run = KalmanRun(p_kalman_filter)
    p_smoother_run = KalmanRun(p_kalman_smoother)
    tfs_run = VanillaPydpfRun(tfs)
    kalman_test = Test_Runner(kalman_filter, run_func=kalman_run, dataset=dataset, data_order=["ground_truth", "observation"])
    p_kalman_test = Test_Runner(p_kalman_filter, run_func=p_kalman_run, dataset=dataset, data_order=["ground_truth", "observation"])
    p_smoother_test = Test_Runner(p_kalman_smoother, run_func=p_smoother_run, dataset=dataset, data_order=["ground_truth", "observation"])
    pvmc_test = make_pvmc_test_run(pvmc, dataset)
    tfs_test = Test_Runner(tfs, run_func=tfs_run, dataset=dataset, data_order=["ground_truth", "observation"])
    return kalman_test, p_kalman_test, p_smoother_test, pvmc_test, tfs_test

def make_info(dataset):
    run_info = {"return": {"mean" : "kalman_mean", "likelihood" : "likelihood", "MSE" : "MSE", "time" : "time", "cov": "Covariance"},
                "output_function": {"kalman_mean": Kalman_mean(), "likelihood": Kalman_log_likelihood_factors(), "MSE": Kalman_MSE(), "Covariance": Kalman_covariance()},
                "shuffle": False,
                "batch_size": 64,
                "device": "cuda:0",
                "time_extent": time_extent,
                "collate_fn": dataset.collate}
    return run_info

def make_pvmc_info(dataset):
    run_info = {"return": {"mean" : "mean", "likelihood" : "likelihood", "MSE" : "MSE", "time" : "time", "State": "State", "Weight": "Weight"},
                "n_particles": 64,
                "shuffle": False,
                "batch_size": 16,
                "collate_fn": dataset.collate,
                "time_extent": time_extent,
                "device": "cuda:0",
                "output_function": {"mean": MarginalSmoothingMean(), "likelihood" :  dSMC_ELBO(),  "MSE": MSE(), "State": State(), "Weight": Weight()}}
    return run_info

def make_tfs_info(dataset):
    run_info = {"return": {"mean": "mean", "likelihood": "likelihood", "MSE": "MSE", "time" : "time", "State": "State", "Weight": "Weight"},
                "n_particles": 64,
                "shuffle": False,
                "batch_size": 64,
                "collate_fn": dataset.collate,
                "time_extent": time_extent,
                "device": "cuda:0",
                "output_function": {"mean": MarginalSmoothingMean(), "likelihood": dSMC_ELBO(), "MSE": MSE(), "State": State(), "Weight": Weight()}}
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
                  "time_extent": time_extent,
                  "output_function": {"likelihood": dSMC_ELBO()}}
    test_info = {"n_particles": 32,
                  "batch_size": 128,
                  "collate_fn": test_set.collate,
                  "time_extent": time_extent,
                  "output_function": {"likelihood": dSMC_ELBO()}}
    info = {"train": train_info,
            "validation": validation_info,
            "test": test_info,
            "loss": f"- time_average.ELBO / {time_extent + 1}",
            "print_each_epoch": {"Validation ELBO": "validation.mean.time_average.likelihood", "Train ELBO": "train.mean.time_average.ELBO"},
            "epochs": 100,
            "device": device,
            "target": f"-validation.mean.time_average.likelihood",
            "print": {"Test ELBO": "test.mean.time_average.likelihood"}
            }
    return info

def make_trainer_routine(model, train_set, validation_set, test_set):
    runner = VanillaPydpfRun(model)
    optim = torch.optim.Adam(model.parameters())
    stage = TrainingStage(runner, train_set, validation_set, test_set, optim, ["ground_truth", "observation"])
    trainer = Trainer(model, stages=[stage])
    return trainer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pm, dm, om = make_true_SSM(5, 5, torch.Generator(device))
    t_dataset, v_dataset, dataset = get_data(Path("./experiments/linear_gaussian/data/"), 5, 5)
    kalman, p_filter, p_smoother = make_filters(pm, dm, om)
    pvmc = make_pvmc(5, 5, torch.Generator(device))
    tfs = make_tfs(5, 5, torch.Generator(device))
    dsmc_m = make_dsmc(5,  5, torch.Generator(device))
    learned_pvmc = make_learned_pvmc(5, 5, torch.Generator(device))
    pvmc_train = make_trainer_routine(learned_pvmc, t_dataset, v_dataset, dataset)
    train_info = make_trainer_info(t_dataset, v_dataset, dataset)
    pvmc_train.fit("run", [train_info], False)
    kalman_t, p_filter_t, p_smoother_t, pvmc_t, tfs_t = make_test_runs(kalman, p_filter, p_smoother, pvmc, tfs, dataset)
    learned_pvmc_t = make_pvmc_test_run(learned_pvmc, dataset)
    dSMC_t = make_pvmc_test_run(dsmc_m, dataset)

    info = make_info(dataset)
    pvmc_info = make_pvmc_info(dataset)
    tfs_info = make_tfs_info(dataset)
    dsmc_info = make_tfs_info(dataset)

    del t_dataset, v_dataset
    experiments = {"dSMC": lambda: dSMC_t.test("dSMC", dsmc_info),
                    "Kalman Filter": lambda: p_filter_t.test("p_filter", info),
                   "RTS Smoother": lambda: p_smoother_t.test("p_smoother", info),
                   "PVMC Oracle": lambda: pvmc_t.test("pvmc_oracle", pvmc_info),
                   "PVMC": lambda: learned_pvmc_t.test("pvmc", pvmc_info),
                   "TFS": lambda: tfs_t.test("tfs", tfs_info),
                   }

    results_df = pd.DataFrame(index=pd.Index(experiments.keys(), name="Method"), columns = ["e_x", "e_l", "time", "W2"])
    results_dict = {}
    for method in experiments.keys():
        results_dict[method] = np.zeros(4)

    n_repeats = 10
    for i in range(n_repeats):
        #dataset = get_only_test_data(Path("./experiments/linear_gaussian/data/"), 5, 5)
        #dataset.apply(lambda observation, **d: observation[:, i:i+1].expand(*observation.shape), "observation")
        #dataset.apply(lambda state, **d: state[:, i:i+1].expand(*state.shape), "state")


        smoother_result = experiments["RTS Smoother"]()
        smoother_mean = smoother_result["mean"]
        smoother_cov = smoother_result["cov"]
        smoother_log_likelihood = smoother_result["likelihood"]
        smoother_log_likelihood = np.sum(smoother_log_likelihood, axis=0)
        smoother_time = smoother_result["time"]
        results_dict["RTS Smoother"][2] += smoother_time
        for method, run_func in experiments.items():
            if method == "RTS Smoother" or method == "Kalman Filter":
                continue
            res = experiments[method]()
            pos_error = position_error(res["mean"], smoother_mean)

            if method == "Kalman Filter":
                res["likelihood"] = np.sum(res["likelihood"], axis = 0)
                res["W2"] = 0.
            else:
                last_state = res["State"][250]
                last_weight = res["Weight"][250]
                res["W2"] = sliced_w2_empirical_gaussian_vec(last_state, last_weight, smoother_mean[250], smoother_cov[250], 1000)
            print((res["likelihood"] - smoother_log_likelihood)[:10] )
            lik_error = likelihood_error(res["likelihood"], smoother_log_likelihood)
            print(f"MSE: {pos_error}")
            print(f"Likelihood: {lik_error}")
            print(f"Time: {res["time"]}")
            print(f"W2: {res["W2"]}")
            result_arr = np.array([pos_error, lik_error, res["time"], res["W2"]])
            results_dict[method] += result_arr

    for method, res in results_dict.items():
        results_df.loc[method] = res/n_repeats
    results_df.to_csv(Path(f"./experiments/linear_gaussian/results-{time_extent}.csv"))

    #pvmc_t.draw_particles(pvmc_info, 0, 1, xlim=[-5,5], ylim=[-5,5], plotted_gt=p_smoother_mean[:, 0])




