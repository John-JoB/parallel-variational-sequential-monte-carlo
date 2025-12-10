import torch
import pydpf
from models.bearings_only import true_model
from pathlib import Path
import argparse
import os

def create_datasets(device, training_trajectories, validation_trajectories, test_trajectories, time_extent, batch_size):
    prior_gen = torch.Generator(device=device).manual_seed(0)
    true_prior = true_model.true_prior(generator=prior_gen)
    dynamic_gen = torch.Generator(device=device).manual_seed(10)
    true_dynamic  = true_model.true_dynamics(generator=dynamic_gen)
    observation_gen = torch.Generator(device=device).manual_seed(20)
    true_observation = true_model.true_observation(generator=observation_gen)
    SSM = pydpf.FilteringModel(dynamic_model=true_dynamic, observation_model=true_observation, prior_model=true_prior)
    data_folder = Path("./experiments/bearings_only/data/")
    raw_data = data_folder / "raw.csv"
    total_trajectories = training_trajectories + validation_trajectories + test_trajectories
    if pydpf.simulate_and_save(raw_data, SSM, time_extent=time_extent, n_trajectories=total_trajectories, batch_size=batch_size, device=device) == -1:
        return
    raw_data = pydpf.StateSpaceDataset(raw_data, state_prefix="state", device="cpu")
    raw_data.apply(lambda state, **data: state[0, :, :3], "series_metadata")
    raw_data.apply(lambda state, **data: state[..., :3], "state")
    train_set, validation_set, test_set = raw_data.deterministic_split((training_trajectories, validation_trajectories, test_trajectories))
    train_set.save(data_folder / "train.csv", series_metadata_path = data_folder / "train_series_metadata.csv")
    validation_set.save(data_folder / "validation.csv", series_metadata_path = data_folder / "validation_series_metadata.csv")
    test_set.save(data_folder / "test.csv", series_metadata_path = data_folder / "test_series_metadata.csv")
    os.remove(data_folder / "raw.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='bearings_only_generate', description='Generate bearings only data')
    parser.add_argument("--device", action="store", default="cpu", help="The device to run the model on", type=str)
    parser.add_argument("--training_trajectories", action="store", default=5000, help="Number of training trajectories", type=int)
    parser.add_argument("--validation_trajectories", action="store", default=1000, help="Number of validation trajectories", type=int)
    parser.add_argument("--test_trajectories", action="store", default=5000, help="Number of test trajectories", type=int)
    parser.add_argument("--time_extent", action="store", default=50, help="Time extent in seconds", type=int)
    parser.add_argument("--batch_size", action="store", default=200, help="Batch size", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    create_datasets(device, args.training_trajectories, args.validation_trajectories, args.test_trajectories, args.time_extent, args.batch_size)