from pathlib import Path

import pydpf
import models.bearings_only.learnt_model as bo_learnt
import torch
from mdps import MDPS
from experiments.common.training_stage import VanillaPydpfRun, Trainer, ParallelRun, TrainingStage, ExperimentRun


class masked_loss(pydpf.Module):
    def __init__(self, keep_mod, base_loss, identity = 0):
        super().__init__()
        self.identity = identity
        self.base_loss = base_loss
        self.keep_mod = keep_mod

    def forward(self, t, **data):
        temp = self.base_loss(t=t, **data)
        if (t + 1) % self.keep_mod == 0:
            return temp
        return torch.full_like(temp, self.identity)


class min_mod(pydpf.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self, x):
        return torch.minimum(x, self.tensor)

class max_mod(pydpf.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self, x):
        return torch.maximum(x, self.tensor)

def make_kernel(generator, starting_bandwith_pos, starting_bandwith_angle, min_bandwidth_pos, min_bandwidth_angle):
    device = generator.device
    gaussian_part = pydpf.MultivariateGaussian(torch.zeros(2, device=device), torch.nn.Parameter(torch.eye(2, device=device) * starting_bandwith_pos**2, requires_grad=True), diagonal_cov=True, generator=generator)
    vonmises_part = pydpf.VonMises(torch.zeros(1, device=device), torch.nn.Parameter(torch.tensor(1/starting_bandwith_angle, device=device), requires_grad=True), generator)
    #torch.nn.utils.parametrizations.parametrize.register_parametrization(gaussian_part, "cholesky_covariance", max_mod(torch.eye(2, device=device) * min_bandwidth_pos))
    #torch.nn.utils.parametrizations.parametrize.register_parametrization(vonmises_part, "concentration", min_mod(torch.tensor(1/min_bandwidth_angle, device=device)))
    compound_dist = pydpf.CompoundDistribution([gaussian_part, vonmises_part], generator)
    syst_resampler = pydpf.SystematicResampler(generator)
    kernel_mixture = pydpf.KernelMixture(compound_dist, generator, syst_resampler)
    return kernel_mixture

def build_MDPS_model(device):
    fw_gen = torch.Generator(device=device).manual_seed(0)
    forward_dynamics = bo_learnt.DynamicsModel(fw_gen)
    bw_gen = torch.Generator(device=device).manual_seed(10)
    backward_dynamics = bo_learnt.DynamicsModel(bw_gen)
    fw_pr_gen = torch.Generator(device=device).manual_seed(20)
    forward_prior = bo_learnt.forward_prior(fw_pr_gen)
    bw_pr_gen = torch.Generator(device=device).manual_seed(30)
    backward_prior = bo_learnt.backward_prior(bw_pr_gen)
    forward_observation_model = bo_learnt.ObservationModel(device)
    backward_observation_model = bo_learnt.ObservationModel(device)
    comb_model = bo_learnt.CombinationWeights(device=device)
    forward_SSM = pydpf.FilteringModel(prior_model=forward_prior, dynamic_model=forward_dynamics, observation_model=forward_observation_model)
    backward_SSM = pydpf.FilteringModel(prior_model=backward_prior, dynamic_model=backward_dynamics, observation_model=backward_observation_model)
    fk_gen = torch.Generator(device=device).manual_seed(40)
    forward_kernel = make_kernel(fk_gen, 2., 0.03, 0.1, 0.001)
    backward_kernel = make_kernel(bw_gen, 2, 0.03, 0.1, 0.001)
    forward_res_kernel = make_kernel(fk_gen, 0.75, 0.03, 0.1, 0.001)
    backward_res_kernel = make_kernel(bw_gen, 0.75, 0.03, 0.1, 0.001)
    mdps = MDPS(forward_SSM, backward_SSM, comb_model, forward_kernel, backward_kernel, forward_res_kernel, backward_res_kernel)
    return mdps

def build_encoder(device):
    return bo_learnt.ObservationEncoder(device)


def get_mdpfs(mdps):
    forward_mdpf = mdps.forward_filter
    backward_mdpf = mdps.backward_filter
    return forward_mdpf, backward_mdpf



def make_output_function(mdps, device):
    forward_kernel = mdps.forward_kernel
    backward_kernel = mdps.backward_kernel
    gen = torch.Generator(device=device).manual_seed(0)
    output_kernel = make_kernel(gen, 2, 0.03, 0.1, 0.001)
    forward_knll = pydpf.NegLogDataLikelihood_Loss(forward_kernel)
    backward_knll = pydpf.NegLogDataLikelihood_Loss(backward_kernel)
    output_knll = pydpf.NegLogDataLikelihood_Loss(output_kernel)
    return forward_knll, backward_knll, output_knll

def load_datasets(folder):
    if isinstance(folder, str):
        folder = Path(folder)
    train_set = pydpf.StateSpaceDataset(folder / "train.csv", state_prefix = "state", series_metadata_path=folder / "train_series_metadata.csv")
    validation_set = pydpf.StateSpaceDataset(folder / "validation.csv", state_prefix="state", series_metadata_path=folder / "validation_series_metadata.csv")
    test_set = pydpf.StateSpaceDataset(folder / "test.csv", state_prefix="state", series_metadata_path=folder / "test_series_metadata.csv")
    return train_set, validation_set, test_set

def make_training_stages(mdps, device):
    data_folder = Path("./experiments/bearings_only/data/")
    forward_mdpf, backward_mdpf = get_mdpfs(mdps)
    encoder = build_encoder(device)
    train_set, validation_set, test_set = load_datasets(data_folder)
    stage_one_opt_args = [{"params": forward_mdpf.resampler.parameters(), "lr": 1e-5},
                {"params": forward_mdpf.SSM.dynamic_model.parameters(), "lr": 1e-3},
                {"params": forward_mdpf.SSM.observation_model.parameters(), "lr": 1e-3},
                {"params": backward_mdpf.resampler.parameters(), "lr": 1e-5},
                {"params": backward_mdpf.SSM.dynamic_model.parameters(), "lr": 1e-3},
                {"params": backward_mdpf.SSM.observation_model.parameters(), "lr": 1e-3},
                {"params": encoder.parameters(), "lr": 1e-3}]
    stage_one_optimiser = torch.optim.Adam(stage_one_opt_args)
    forward_knll, backward_knll, output_knll = make_output_function(mdps, device)


    forward_run = VanillaPydpfRun(forward_mdpf)
    class _BackwardRun(ExperimentRun):
        def __init__(self, pydpf_run):
            super().__init__(preprocessors=None)
            self.pydpf_run = pydpf_run

        def run(self, mode, run_info, **data):
            backward_data = {}
            for k, v in data.items():
                backward_data[k] = torch.flip(v, (0,))
            return self.pydpf_run.run(mode, run_info, **backward_data)


    backward_run = _BackwardRun(VanillaPydpfRun(backward_mdpf))
    combined_forward_backward_train = ParallelRun(preprocessors={"observation": encoder}, forward = forward_run, backward = backward_run)

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
                            "time_extent": 50}

    forward_info_train = {**info_train, "output_function": {"KNLL": masked_loss( 4, forward_knll)}}
    forward_info_validation = {**info_validation, "output_function": {"KNLL": forward_knll}}
    backward_info_train = {**info_train, "output_function": {"KNLL": masked_loss(4, backward_knll)}}
    backward_info_validation = {**info_validation, "output_function": {"KNLL": backward_knll}}
    stage_one_info = {"forward": {"train": forward_info_train, "validation": forward_info_validation},
                          "backward": {"train": backward_info_train, "validation": backward_info_validation},
                          "train": info_train,
                          "validation": info_validation,
                          "loss": "2. * forward.time_average.KNLL + 2. * backward.time_average.KNLL",
                          "print_each_epoch": {"train loss": "(2 * train.mean.forward.time_average.KNLL + 2 * train.mean.backward.time_average.KNLL)",
                          "validation loss": "(0.5 * validation.mean.forward.time_average.KNLL + 0.5 * validation.mean.backward.time_average.KNLL)",
                          "forward validation loss": "validation.mean.forward.time_average.KNLL",
                          "backward validation loss": "validation.mean.backward.time_average.KNLL"},
                          "epochs": 100,
                          "device" : "cuda:0",
                          "run_test" : False
                          }

    stage_two_info = stage_one_info.copy()
    stage_two_info["epochs"] = 10


    stage_one_train = TrainingStage(combined_forward_backward_train,
                                    train_set,
                                    validation_set,
                                    test_set,
                                    stage_one_optimiser,
                                    ["ground_truth", "observation", "series_metadata"],
                                    )


    stage_two_opt_args = [{"params": forward_knll.parameters(), "lr": 1e-3},
                          {"params": backward_knll.parameters(), "lr": 1e-3}]

    stage_two_optimiser = torch.optim.Adam(stage_two_opt_args)

    stage_two_train = TrainingStage(combined_forward_backward_train,
                                    train_set,
                                    validation_set,
                                    test_set,
                                    stage_two_optimiser,
                                    ["ground_truth", "observation", "series_metadata"],
                                    )

    combined_info_train = {**info_train, "output_function": {"KNLL": masked_loss(4, output_knll)}}
    combined_info_validation = {**info_validation, "output_function": {"KNLL": output_knll}}
    combined_info_test = {**info_test, "output_function": {"KNLL": output_knll}}

    stage_three_info = {"train": combined_info_train,
                             "validation": combined_info_validation,
                             "loss": "4 * time_average.KNLL",
                             "print_each_epoch": {"train loss": "4 * train.mean.time_average.KNLL", "validation loss": "validation.mean.time_average.KNLL"},
                             "epochs": 100,
                             "device": "cuda:0"
                             }
    stage_four_info = stage_three_info.copy()
    stage_four_info["test"] = combined_info_test
    stage_four_info["print"] = {"test loss": "test.mean.time_average.KNLL"}

    stage_three_opt_args = [{"params": mdps.combination_model.parameters(), "lr": 1e-3},
                            {"params": output_knll.parameters(), "lr": 1e-4},]

    stage_three_optimiser = torch.optim.Adam(stage_three_opt_args)

    stage_three_run = VanillaPydpfRun(mdps, preprocessors={"observation": encoder})
    stage_three_train = TrainingStage(stage_three_run,
                                      train_set,
                                      validation_set,
                                      test_set,
                                      stage_three_optimiser,
                                      ["ground_truth", "observation", "series_metadata"],
                                      )

    stage_four_opt_args = [{"params": forward_mdpf.resampler.parameters(), "lr": 1e-5},
                           {"params": backward_mdpf.resampler.parameters(), "lr": 1e-5},
                           {"params": forward_mdpf.SSM.dynamic_model.parameters(), "lr": 1e-4},
                           {"params": backward_mdpf.SSM.dynamic_model.parameters(), "lr": 1e-4},
                           {"params": forward_mdpf.SSM.observation_model.parameters(), "lr": 1e-4},
                           {"params": backward_mdpf.SSM.observation_model.parameters(), "lr": 1e-4},
                           {"params": encoder.parameters(), "lr": 1e-4},
                           {"params": mdps.combination_model.parameters(), "lr": 1e-4},
                           {"params": output_knll.parameters(), "lr": 1e-5},
                           ]

    stage_four_optimiser = torch.optim.Adam(stage_four_opt_args)

    stage_four_train = TrainingStage(stage_three_run,
                                      train_set,
                                      validation_set,
                                      test_set,
                                      stage_four_optimiser,
                                      ["ground_truth", "observation", "series_metadata"],
                                      )
    return [stage_one_train, stage_two_train, stage_three_train, stage_four_train], forward_knll, backward_knll, output_knll, encoder, [stage_one_info, stage_two_info, stage_three_info, stage_four_info]


def make_training_routine(mdps, device):
    stages, forward_knll, backward_knll, output_knll, encoder, info, = make_training_stages(mdps, device)

    routine = Trainer(mdps, forward_knll, backward_knll, output_knll, encoder, stages = stages)
    return routine, info

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mdps = build_MDPS_model(device)
    routine, info = make_training_routine(mdps, device)
    routine.fit("first_test",
                    info,
                    True,
                    Path("./experiments/bearings_only/saved_models/"))
