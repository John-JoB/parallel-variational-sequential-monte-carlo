import torch

from experiments.common.optimisers import OptimList
from experiments.common.parameter_set import ParameterSet
from models.stochastic_vol import model
import pydpf
from parallel_smoother_new import ParallelSmoother
from smoother_outputs import dSMC_ELBO
from experiments.common.training import VanillaPydpfRun, TrainingStage, Trainer
from pathlib import Path

true_alpha = 0.91
true_beta = 0.5
true_sigma = 1.
time_extent = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(10)
folder = Path("experiments/stochastic_vol/")
data_folder = folder / "data/"


def make_true_ssm():
    prior = model.StochasticVolatility_Prior(true_sigma, true_alpha, generator)
    dynamic = model.StochasticVolatility_Dynamic(true_sigma, true_alpha, generator)
    observation = model.StochasticVolatility_Observation(true_beta, generator)
    return pydpf.FilteringModel(dynamic_model=dynamic, prior_model=prior, observation_model=observation)

def inverse_sigmoid(x):
    x = torch.clamp(x, 1e-6, 1 - 1e-6)
    return torch.log(x) - torch.log(1-x)

def make_learned_ssm():

    alpha_0 = torch.rand(1, generator=generator, device=device, dtype=torch.float32).squeeze()
    beta_0 = torch.rand(1, generator=generator, device=device, dtype=torch.float32).squeeze()
    sigma_0 = torch.rand(1, generator=generator, device=device, dtype=torch.float32).squeeze()
    alpha = torch.nn.Parameter(inverse_sigmoid(alpha_0), requires_grad=True)
    beta = torch.nn.Parameter(inverse_sigmoid(beta_0), requires_grad=True)
    sigma = torch.nn.Parameter(inverse_sigmoid(sigma_0), requires_grad=True)
    #alpha=true_alpha
    #beta = true_beta
    #sigma = true_sigma
    prior = model.StochasticVolatility_Prior(sigma, alpha, generator)
    dynamic = model.StochasticVolatility_Dynamic(sigma, alpha, generator)
    observation = model.StochasticVolatility_Observation(beta, generator)
    return pydpf.FilteringModel(dynamic_model=dynamic, prior_model=prior, observation_model=observation), alpha, beta, sigma


def make_proposal():
    #return model.LSTM_proposal(1, 1, 16, time_extent, generator)
    return model.ConvProposal(1, 1, 64, time_extent, generator)
    #return model.KalmanProposal(1, 1, generator)

def make_trainer_info(train_set, validation_set, test_set):
    train_info = {"n_particles": 64,
                  "batch_size": 32,
                  "collate_fn": train_set.collate,
                  "time_extent": time_extent,
                  "output_function": {"ELBO": dSMC_ELBO()}}
    validation_info = {"n_particles": 128,
                  "batch_size": 128,
                  "collate_fn": validation_set.collate,
                  "time_extent": time_extent,
                  "output_function": {"likelihood": dSMC_ELBO()}}
    test_info = {"n_particles": 128,
                  "batch_size": 128,
                  "collate_fn": test_set.collate,
                  "time_extent": time_extent,
                  "output_function": {"likelihood": dSMC_ELBO()}}
    info = {"train": train_info,
            "validation": validation_info,
            "test": test_info,
            "loss": f"- time_average.ELBO / {time_extent + 1}",
            "print_each_epoch": {"Validation ELBO": "validation.mean.time_average.likelihood", "Train ELBO": "train.mean.time_average.ELBO"},
            "epochs": 300,
            "device": device,
            "target": f"-validation.mean.time_average.likelihood",
            "print": {"Test ELBO": "test.mean.time_average.likelihood"}
            }
    return info

def get_data(data_dir):
    print(data_dir)
    train = pydpf.StateSpaceDataset(data_dir / "train.csv", state_prefix="state", device=torch.device("cuda:0"))
    validation = pydpf.StateSpaceDataset(data_dir / "validation.csv", state_prefix="state", device=torch.device("cuda:0"))
    test = pydpf.StateSpaceDataset(data_dir / "test.csv", state_prefix="state", device=torch.device("cuda:0"))
    return train, validation, test

class run_each_epoch():
    def __init__(self, model):
        self.model = model

    def __call__(self):
        self.model.beta_observation = (self.model.beta_observation - 1) * 0.9 + 1
        self.model.beta_prior = 20.


def make_trainer_routine(model, alpha, beta, sigma, train_set, validation_set, test_set):
    runner = VanillaPydpfRun(model)
    ssm_optim = torch.optim.AdamW([{"params": [alpha], "lr": 0.005, "betas": (0.9, 0.999), "weight_decay": 0.},
                                  {"params": [beta], "lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.},
                                  {"params": [sigma], "lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.},
                                {"params": ParameterSet(model.proposal), "lr":1e-3, "weight_decay": 1e-4}])
    #ssm_optim = torch.optim.RMSprop([{"params": [alpha], "lr": 0.01, "weight_decay": 0.},
    #                              {"params": [beta], "lr": 0.01, "weight_decay": 0.},
    #                              {"params": [sigma], "lr": 0.01, "weight_decay": 0.},
    #                              {"params": ParameterSet(model.proposal), "lr":1e-4, "weight_decay": 1e-3}])
    #ssm_optim = torch.optim.LBFGS(ParameterSet(model.SSM), lr=1., max_iter=20, line_search_fn="strong_wolfe")
    #prop_optim = torch.optim.RMSprop(ParameterSet(model.proposal), lr=1e-3, weight_decay=1e-4)
    #optim = OptimList([ssm_optim, prop_optim])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(ssm_optim, max_lr=[0.01, 0.01, 0.01, 0.0001], epochs=300, steps_per_epoch=16, final_div_factor=1e3)
    stage = TrainingStage(runner, train_set, validation_set, test_set, ssm_optim, ["ground_truth", "observation"],lr_scheduler=lr_scheduler, lr_step_freq="opt_step", run_on_epoch=run_each_epoch(pvmc))
    trainer = Trainer(model, stages=[stage])
    return trainer

if __name__ == '__main__':
    train_data, validation_data, test_data = get_data(data_folder)
    ssm, alpha_p, beta_p, sigma_p = make_learned_ssm()
    ssm.update()
    alpha = ssm.dynamic_model.alpha.clone()
    beta = ssm.observation_model.beta.clone()
    sigma = ssm.dynamic_model.sigma.clone()
    print(alpha)
    print(torch.abs(alpha - true_alpha).item())
    print(beta)
    print(torch.abs(beta - true_beta).item())
    print(sigma)
    print(torch.abs(sigma - true_sigma).item())
    proposal = make_proposal()
    pvmc = ParallelSmoother(proposal, ssm)
    pvmc.beta_observation = 10.
    info = make_trainer_info(train_data, validation_data, test_data)
    routine = make_trainer_routine(pvmc, alpha_p, beta_p, sigma_p, train_data, validation_data, test_data)
    routine.fit("Run", [info], False)
    ssm.update()
    alpha = ssm.dynamic_model.alpha.clone()
    beta = ssm.observation_model.beta.clone()
    sigma = ssm.dynamic_model.sigma.clone()
    print(alpha)
    print(torch.abs(alpha - true_alpha).item())
    print(beta)
    print(torch.abs(beta - true_beta).item())
    print(sigma)
    print(torch.abs(sigma - true_sigma).item())

