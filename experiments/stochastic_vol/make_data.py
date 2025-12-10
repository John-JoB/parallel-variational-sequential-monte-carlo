import pydpf
import torch
from models.stochastic_vol import model
from pathlib import Path
import os
import argparse

def generate(alpha, beta, sigma, device=torch.device('cpu')):
    prior_gen = torch.Generator(device=device).manual_seed(0)
    prior_model = model.StochasticVolatility_Prior(sigma, alpha, prior_gen)
    dyn_gen = torch.Generator(device=device).manual_seed(10)
    dynamic_model = model.StochasticVolatility_Dynamic(sigma, alpha, dyn_gen)
    obs_gen = torch.Generator(device=device).manual_seed(20)
    obs_model = model.StochasticVolatility_Observation(beta, obs_gen)
    SSM = pydpf.FilteringModel(dynamic_model=dynamic_model, observation_model=obs_model, prior_model=prior_model)
    data_folder = Path("./experiments/stochastic_vol/data/")
    raw_data = data_folder / "raw.csv"
    total_trajectories = 1000
    if pydpf.simulate_and_save(raw_data, SSM, time_extent=500, n_trajectories=total_trajectories, batch_size=50, device=device) == -1:
        return
    raw_data = pydpf.StateSpaceDataset(raw_data, state_prefix="state", device="cpu")
    train_set, validation_set, test_set = raw_data.deterministic_split((500, 100, 400))
    train_set.save(data_folder / f"train.csv", series_metadata_path=data_folder / "train_series_metadata.csv")
    validation_set.save(data_folder / f"validation.csv", series_metadata_path=data_folder / "validation_series_metadata.csv")
    test_set.save(data_folder / f"test.csv", series_metadata_path=data_folder / "test_series_metadata.csv")
    os.remove(data_folder / "raw.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='linear_gaussian_generate', description='Generate linear gaussian data')
    parser.add_argument("--device", action="store", default="cpu", help="The device to run the model on", type=str)
    parser.add_argument("--alpha", action="store", default=0.91, help="The dimensionality of the state", type=float)
    parser.add_argument("--beta", action="store", default=0.5, help="The dimensionality of the observations", type=float)
    parser.add_argument("--sigma", action="store", default=1.0, help="The dimensionality of the observations", type=float)
    args = parser.parse_args()
    device = torch.device(args.device)
    generate(args.alpha, args.beta, args.sigma, device)