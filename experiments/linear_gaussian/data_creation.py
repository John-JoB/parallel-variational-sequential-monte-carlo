import pydpf
import torch
from models.linear_gaussian import true_model as tm
from pathlib import Path
import os
import argparse

def generate(dx=5, dy=5, device=torch.device('cpu')):
    prior_gen = torch.Generator(device=device).manual_seed(0)
    prior_model = tm.GaussianPrior(dx, prior_gen)
    dyn_gen = torch.Generator(device=device).manual_seed(10)
    dynamic_model = tm.GaussianDynamic(dx, dyn_gen)
    obs_gen = torch.Generator(device=device).manual_seed(20)
    obs_model = tm.GaussianObservation(dx, dy, obs_gen)
    SSM = pydpf.FilteringModel(dynamic_model=dynamic_model, observation_model=obs_model, prior_model=prior_model)
    data_folder = Path("./experiments/linear_gaussian/data/")
    raw_data = data_folder / "raw.csv"
    total_trajectories = 1000
    if pydpf.simulate_and_save(raw_data, SSM, time_extent=50, n_trajectories=total_trajectories, batch_size=50, device=device) == -1:
        return
    raw_data = pydpf.StateSpaceDataset(raw_data, state_prefix="state", device="cpu")
    train_set, validation_set, test_set = raw_data.deterministic_split((500, 100, 400))
    train_set.save(data_folder / f"{dx}-{dy}-train.csv", series_metadata_path=data_folder / "train_series_metadata.csv")
    validation_set.save(data_folder / f"{dx}-{dy}-validation.csv", series_metadata_path=data_folder / "validation_series_metadata.csv")
    test_set.save(data_folder / f"{dx}-{dy}-test.csv", series_metadata_path=data_folder / "test_series_metadata.csv")
    os.remove(data_folder / "raw.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='linear_gaussian_generate', description='Generate linear gaussian data')
    parser.add_argument("--device", action="store", default="cpu", help="The device to run the model on", type=str)
    parser.add_argument("--dx", action="store", default=5, help="The dimensionality of the state", type=int)
    parser.add_argument("--dy", action="store", default=5, help="The dimensionality of the observations", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    generate(args.dx, args.dy, device)