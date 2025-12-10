import pydpf

from experiments.stochastic_vol.main import time_extent
from models.lokta_volterra import true_model as tm
import torch
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate():
    dynamic_model = tm.TrueDynamicModel(device = device)
    observation_model = tm.TrueObservationModel()
    prior_model = tm.TruePriorModel(device = device)
    SSM = pydpf.FilteringModel(dynamic_model=dynamic_model, prior_model=prior_model, observation_model=observation_model)
    data_folder = Path("./experiments/lotka_volterra/data/")
    if pydpf.simulate_and_save(data_folder / "raw.csv", SSM, time_extent = 256, n_trajectories=250, batch_size=100, device=device) == -1:
        return
    raw_data = pydpf.StateSpaceDataset(data_folder / "raw.csv", state_prefix="state", device="cpu")
    train_set, validation_set, test_set = raw_data.deterministic_split((100, 50, 100))
    train_set.save(data_folder / f"train.csv")
    validation_set.save(data_folder / f"validation.csv")
    test_set.save(data_folder / f"test.csv")

if __name__ == "__main__":
    generate()