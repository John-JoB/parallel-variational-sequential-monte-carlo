from pydpf import Module
from torch import Tensor
import torch
from experiments.bearings_only.main import load_datasets
from experiments.bearings_only.pvmc_main import build_conv_encoder
from experiments.common.training import TrainingStage, VanillaPydpfRun, Trainer
from experiments.common.parameter_set import ParameterSet
from pathlib import Path


class SimpleML(Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    @staticmethod
    def print_grad(grad):
        print("Gradient for state:")
        print(grad[3::4])
        print("---")

    def forward(self, n_particles: int,
                time_extent: int,
                aggregation_function: Module | dict,
                observation: Tensor,
                *,
                gradient_regulariser: None = None,
                ground_truth: Tensor | None = None,
                control: Tensor | None = None,
                time: Tensor | None = None,
                series_metadata: Tensor | None = None) -> Tensor|dict:

        observation = observation[:time_extent + 1]
        if ground_truth is not None:
            ground_truth = ground_truth[:time_extent + 1]
        if control is not None:
            control = control[:time_extent + 1]
        if time is not None:
            time = time[:time_extent + 1]
        state = self.net(observation=observation,control=control,time=time, series_metadata=series_metadata)
        scale = torch.tensor([[[10., 10., torch.pi]]], device=state.device)
        return {"MSE": torch.sum((state - ground_truth/scale) ** 2, dim=-1)}


class custom_net(Module):
    def __init__(self, device):
        super().__init__()
        self.lstm = torch.nn.LSTM(8, 8, 2, batch_first=False, bidirectional=False, proj_size=7, device=device, dropout=0.3)

    def forward(self, observation: Tensor, series_metadata: Tensor, **data) -> Tensor:
        transformed_metadata = torch.cat([series_metadata[..., 0:2], torch.sin(series_metadata[..., -1:]), torch.cos(series_metadata[..., -1:])], dim=-1)
        initial_hidden_state = torch.nn.functional.pad(transformed_metadata, (0, 3), mode='constant', value=0)
        zero_tensor = torch.zeros((2, observation.size(1), 8), device=observation.device)
        net_out = self.lstm(observation, (initial_hidden_state.repeat(2, 1, 1), zero_tensor))[0]
        mean_state = torch.cat([net_out[..., 0:2], torch.atan2(net_out[..., 2:3], net_out[..., 3:4])], dim=-1).contiguous()
        return mean_state

def make_training_stage(device, data_folder):
    net = custom_net(device)
    run = SimpleML(net)
    train_set, validation_set, test_set = load_datasets(data_folder)
    encoder = build_conv_encoder(device)

    param = ParameterSet(encoder, run)
    print(param)

    stage_one_opt = torch.optim.Adam([{"params": param, "lr": 1e-3}], weight_decay=1e-5)
    run_func = VanillaPydpfRun(run, {"observation": encoder})

    stage_one_train = TrainingStage(run_func, train_set, validation_set, test_set, stage_one_opt, ["ground_truth", "observation", "series_metadata"])

    info_train = {"n_particles": 50,
                  "batch_size": 32,
                  "collate_fn": train_set.collate,
                  "time_extent": 50}

    info_validation = {"n_particles": 50,
                       "batch_size": 32,
                       "collate_fn": validation_set.collate,
                       "time_extent": 50}
    info_test = {"n_particles": 50,
                 "batch_size": 32,
                 "collate_fn": test_set.collate,
                 "time_extent": 50,
                 "output_function": {"KNLL": None}}

    stage_one_info = {"train": {**info_train, "output_function": {"KNLL": None}},
                      "validation": {**info_validation, "output_function": {"KNLL": None}},
                      "loss": "time_average.MSE",
                      "print_each_epoch": {"train loss": "train.mean.time_average.MSE", "validation loss": "validation.mean.time_average.MSE"},
                      "epochs": 50,
                      "device": "cuda:0"
                      }

    routine = Trainer(run, encoder, stages = [stage_one_train])
    return routine, [stage_one_info]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    routine, info = make_training_stage(device, Path("./experiments/bearings_only/data/"))
    routine.fit("test",
                info,
                False,
                Path("./experiments/bearings_only/saved_models/pvmc/"))

