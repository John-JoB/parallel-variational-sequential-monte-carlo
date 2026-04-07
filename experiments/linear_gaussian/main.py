"""Script not used to generate any results from the paper"""

import argparse

import pydpf
import torch

from models.linear_gaussian import learned_model, true_model
from parallel_smoother_new import ParallelSmoother
from truncated_pvmc import Truncated
from smoother_outputs import MSE, VAE_ELBO, MarginalSmoothingMean, dSMC_ELBO
from pathlib import Path
from experiments.common.training import TrainingStage, Trainer, VanillaPydpfRun

def make_true_SSM(dx, dy, generator):
    prior_model = true_model.GaussianPrior(dx, generator)
    dynamic_model = true_model.GaussianDynamic(dx, generator)
    obs_model = true_model.GaussianObservation(dx, dy, generator)
    return pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=obs_model)

def make_learned_SSM(dx, dy, generator):
    prior_model = learned_model.GaussianPrior(dx, generator)
    dynamic_model = learned_model.GaussianDynamic(dx, generator)
    obs_model = learned_model.GaussianObservation(dx, dy, generator)
    #prior_model = true_model.GaussianPrior(dx, generator)
    #dynamic_model = true_model.GaussianDynamic(dx, generator)
    #obs_model = true_model.GaussianObservation(dx, dy, generator)
    return pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=obs_model)

def make_kalman_proposal(dx, dy, generator):
    return learned_model.KalmanProposal(dx, dy, generator, False)

def make_pvmc(proposal, SSM):
    return ParallelSmoother(proposal, SSM)

def make_truncated(proposal, SSM):
    return Truncated(proposal, SSM)



def get_data(folder, dx, dy):
    train = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-train.csv", state_prefix="state")
    validation = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-validation.csv", state_prefix="state")
    test = pydpf.StateSpaceDataset(folder / f"{dx}-{dy}-test.csv", state_prefix="state")
    return train, validation, test

def make_generic_stage_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent, outputs):
    train_info = {"n_particles": n_particles[0],
                  "batch_size": batch_size[0],
                  "collate_fn": train_set.collate,
                  "time_extent": time_extent[0],
                  "output_function": outputs}
    validation_info = {"n_particles": n_particles[1],
                  "batch_size": batch_size[1],
                  "collate_fn": validation_set.collate,
                  "time_extent": time_extent[1],
                  "output_function": outputs}
    test_info = {"n_particles": n_particles[2],
                  "batch_size": batch_size[2],
                  "collate_fn": test_set.collate,
                  "time_extent": time_extent[2],
                 "output_function": outputs}
    return train_info, validation_info, test_info

def make_stage_info(train_info, validation_info, test_info, train_loss, target, epochs, printed_dict, final_printed_dict, device):
    info =  {"train": train_info,
        "validation": validation_info,
        "test": test_info,
        "loss": f"-time_average.{train_loss}",
        "print_each_epoch": printed_dict,
        "epochs": epochs,
        "device": device,
        "target": f"validation.mean.time_average.{target}",
        "print": final_printed_dict
        }
    return info

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_particles_train", action="store", type=int, default=50)
    parser.add_argument("--n_particles_validation", action="store", type=int, default=0)
    parser.add_argument("--n_particles_test", action="store", type=int, default=50)
    parser.add_argument("--batch_size_train", action="store", type=int, default=16)
    parser.add_argument("--batch_size_validation", action="store", type=int, default=0)
    parser.add_argument("--batch_size_test", action="store", type=int, default=0)
    parser.add_argument("--time_extent_train", action="store", type=int, default=50)
    parser.add_argument("--time_extent_validation", action="store", type=int, default=0)
    parser.add_argument("--time_extent_test", action="store", type=int, default=0)
    parser.add_argument("--epochs", action="store", type=int, default=50)
    parser.add_argument("--device", action="store", type=str, default="cuda:0")
    parser.add_argument("--output_functions", action="store", type=str, default="elbo", nargs="+")
    parser.add_argument("--loss", action="store", type=str, default="elbo")
    parser.add_argument("--target", action="store", type=str, default="elbo")
    parser.add_argument("--experiment", action="store", type=str, default="PVMC")
    parser.add_argument("--path", action="store", type=str, default="./experiments/linear_gaussian/data/")
    parser.add_argument("--save_model_path", action="store", type=str, default="./experiments/linear_gaussian/saved_models/")
    parser.add_argument("--dx", action="store", type=int, default=5)
    parser.add_argument("--dy", action="store", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    return parser

def parse_tvt_arg(train, validation, test):
    if test == 0:
        test = train
    if validation == 0:
        validation = test
    return train, validation, test

def str_to_output(output_as_str, dx, device):
    match output_as_str.casefold():
        case "mse":
            return MSE()
        case "vae_elbo":
            return VAE_ELBO()
        case "mean":
            return MarginalSmoothingMean()
        case "elbo":
            return dSMC_ELBO()
    raise ValueError(f"No match found for {output_as_str}")

def parse_outputs(output_as_str, dx, device):
    if isinstance(output_as_str, str):
        output_as_str = [output_as_str]
    output_dict = {}
    printed_dict = {}
    final_printed_dict = {}
    for o in output_as_str:
        ouptut_func = str_to_output(o, dx, device)
        printed_dict[f"train {o}"] = f"train.mean.time_average.{o}"
        printed_dict[f"validation {o}"] = f"validation.mean.time_average.{o}"
        final_printed_dict[f"test {o}"] = f"test.mean.time_average.{o}"
        output_dict[o] = ouptut_func
    return output_dict, printed_dict, final_printed_dict

def str_to_experiment(experiment, ssm, proposal):
    match experiment.casefold():
        case "pvmc":
            return make_pvmc(proposal, ssm)
        case "truncated":
            return make_truncated(proposal, ssm)

def make_trainer_stage(experiment, dx, dy, device, train_set, validation_set, test_set):
    ssm = make_learned_SSM(dx, dy, torch.Generator(device=device).manual_seed(0))
    proposal_model = make_kalman_proposal(dx, dy, torch.Generator(device=device).manual_seed(0))
    model = str_to_experiment(experiment, ssm, proposal_model)
    runner = VanillaPydpfRun(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    stage = TrainingStage(runner, train_set, validation_set, test_set, optim, ["ground_truth", "observation"])
    return model, stage

if __name__ == "__main__":
    print("This is testing script it was not used to generate any results in our paper. test_against_kalman_filters.py is the script that generates the main results from Section 5.1")
    args = add_arguments().parse_args()
    device_str = args.device
    device = torch.device(device_str)
    data_path = Path(args.path)
    dx = args.dx
    dy = args.dy
    train_set, validation_set, test_set = get_data(data_path, dx, dy)
    n_particles = parse_tvt_arg(args.n_particles_train, args.n_particles_validation, args.n_particles_test)
    batch_size = parse_tvt_arg(args.batch_size_train, args.batch_size_validation, args.batch_size_test)
    time_extent = parse_tvt_arg(args.time_extent_train, args.time_extent_validation, args.time_extent_test)
    outputs, printed_dict, final_printed_dict = parse_outputs(args.output_functions, dx, device)
    train_info, validation_info, test_info = make_generic_stage_info(train_set, validation_set, test_set, n_particles, batch_size, time_extent, outputs)
    info = make_stage_info(train_info, validation_info, test_info, args.loss, args.target, args.epochs, printed_dict, final_printed_dict, device_str)
    experiments = args.experiment
    if isinstance(experiments, str):
        experiments = [experiments]
    for experiment in experiments:
        model, stage = make_trainer_stage(experiment, dx, dy, device, train_set, validation_set, test_set)
        if args.profile:
            stage.profile(model, info)
        else:
            trainer = Trainer(model, stages=[stage])
            trainer.fit(f"{experiment}-{dx}-{dy}", [info], intermediate_folder = Path(args.save_model_path))