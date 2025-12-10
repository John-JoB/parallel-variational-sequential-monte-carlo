from pathlib import Path
import torch
import models.bearings_only.pvmc_learnt_model as lm
import pydpf
from parallel_smoother_new import ParallelSmoother
from experiments.bearings_only.main import make_kernel, load_datasets
from smoother_outputs import NegativeKernelLogLikelihood as KNLL
from smoother_outputs import MSE
from proposal_to_output import ProposalRunner
from experiments.common.training import VanillaPydpfRun, TrainingStage, Trainer, ModuleList
from experiments.common.parameter_set import ParameterSet
from truncated_pvmc import Truncated


class masked_loss(pydpf.Module):
    def __init__(self, keep_mod, base_loss, identity = 0):
        super().__init__()
        self.identity = identity
        self.base_loss = base_loss
        self.keep_mod = keep_mod

    def forward(self, **data):
        temp = self.base_loss( **data)
        mask = torch.zeros_like(temp)
        mask[0::self.keep_mod] = 1
        return temp * mask

def build_model_comps(device):
    proposal_gen = torch.Generator(device=device).manual_seed(0)
    dynamic_model = lm.DynamicModel(proposal_gen)
    observation_model = lm.ObservationModel(device)
    prior_model = lm.PriorModel(proposal_gen)
    proposal_model = lm.ProposalModel(proposal_gen)
    return pydpf.FilteringModel(dynamic_model=dynamic_model, observation_model=observation_model, prior_model=prior_model), proposal_model

def build_pvmc(SSM, proposal_model):
    return ParallelSmoother(proposal_model, SSM)

def build_output_funcs(device):
    gen = torch.Generator(device=device)
    first_kernel = make_kernel(gen, 2, 0.03, 0.1, 0.001)
    real_kernel = make_kernel(gen, 2, 0.03, 0.1, 0.001)
    first_knll = KNLL(first_kernel)
    real_knll = KNLL(real_kernel)
    return first_knll, real_knll


def build_proposal_only_model(proposal_model):
    return ProposalRunner(proposal_model)

def build_truncated(SSM, proposal_model):
    return Truncated(proposal_model, SSM)

def build_conv_encoder(device):
    return lm.ObservationEncoder(device=device)

def make_training_stages(data_folder, device):

    train_set, validation_set, test_set = load_datasets(data_folder)
    SSM, proposal_model = build_model_comps(device)
    pvmc = build_pvmc(SSM, proposal_model)
    proposal_only_model = build_proposal_only_model(proposal_model)
    truncated = build_truncated(SSM, proposal_model)
    first_knll, _ = build_output_funcs(device)
    first_knll = MSE()
    encoder = build_conv_encoder(device)
    proposal_only_run = VanillaPydpfRun(proposal_only_model, preprocessors={"observation": encoder})
    pvmc_run = VanillaPydpfRun(pvmc, preprocessors={"observation": encoder})
    truncated_run = VanillaPydpfRun(truncated, preprocessors={"observation": encoder})

    prop_only_param = ParameterSet(proposal_only_model)
    pvmc_param = ParameterSet(pvmc)
    encoder_params = ParameterSet(encoder)
    first_knll_params = ParameterSet(first_knll)

    scale_params = (#ParameterSet(SSM.dynamic_model.tolerance_dist)
                    #+ ParameterSet(SSM.prior_model.tolerance_dist)
                    ParameterSet(proposal_model.dist)
                    + ParameterSet(first_knll))

    dyn_params = ParameterSet(SSM.dynamic_model) + ParameterSet(SSM.prior_model)


    stage_one_opt = torch.optim.Adam([{"params": prop_only_param - scale_params + encoder_params, "lr": 1e-3}, {"params": scale_params - ParameterSet(first_knll), "lr": 1e-2, "weight_decay": 0}], weight_decay=1e-5)
    stage_two_opt = torch.optim.AdamW([{"params": prop_only_param - scale_params  + encoder_params, "lr": 1e-3}, {"params": scale_params, "lr": 1e-2, "weight_decay": 0}], weight_decay=1e-5)
    stage_three_opt = torch.optim.Adam([{"params": pvmc_param - prop_only_param - scale_params - dyn_params, "lr": 1e-3}], weight_decay=1e-5)
    stage_three_opt = torch.optim.Adam([{"params": dyn_params - scale_params, "lr": 1e-1}, {"params": pvmc_param + encoder_params - dyn_params - scale_params, "lr": 1e-3}, {"params": scale_params - first_knll_params, "lr": 1e-3, "weight_decay": 0}], weight_decay=1e-5)
    stage_four_opt = torch.optim.Adam([{"params": dyn_params - scale_params, "lr": 1e-1}, {"params": pvmc_param + encoder_params - dyn_params - scale_params, "lr": 1e-3},  {"params": scale_params, "lr": 1e-3, "weight_decay": 0}], weight_decay=1e-5)

    stage_one_train = TrainingStage(proposal_only_run, train_set, validation_set, test_set, stage_one_opt, ["ground_truth", "observation", "series_metadata"])
    stage_two_train = TrainingStage(proposal_only_run, train_set, validation_set, test_set, stage_two_opt, ["ground_truth", "observation", "series_metadata"])
    stage_three_train = TrainingStage(pvmc_run, train_set, validation_set, test_set, stage_three_opt, ["ground_truth", "observation", "series_metadata"])
    stage_four_train = TrainingStage(pvmc_run, train_set, validation_set, test_set, stage_four_opt, ["ground_truth", "observation", "series_metadata"])


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
                 "output_function": {"KNLL": first_knll}}



    stage_one_info  =  {"train": { **info_train, "output_function": {"KNLL": masked_loss(4, first_knll) }},
                                        "validation": { **info_validation, "output_function": {"KNLL": first_knll}},
                                        "loss": "4 * time_average.KNLL",
                                        "print_each_epoch": {"train loss": "4 * train.mean.time_average.KNLL", "validation loss": "validation.mean.time_average.KNLL"},
                                        "epochs": 50,
                                        "device": "cuda:0"
                                        }

    stage_two_info = stage_one_info.copy()
    stage_two_info["epochs"] = 50

    stage_three_info = {"train": { **info_train, "output_function": {"KNLL": masked_loss(4, first_knll) }},
                                        "validation": { **info_validation, "output_function": {"KNLL": first_knll}},
                                        "loss": "4 * time_average.KNLL",
                                        "print_each_epoch": {"train loss": "4 * train.mean.time_average.KNLL", "validation loss": "validation.mean.time_average.KNLL"},
                                        "epochs": 50,
                                        "device": "cuda:0",
                                        }

    stage_four_info = stage_three_info.copy()
    stage_three_info["train"] = stage_three_info["train"].copy()
    stage_three_info["train"]["time_extent"] = 50
    stage_four_info["test"] = info_test
    stage_four_info["epochs"] = 50
    stage_four_info["print"] = {"test loss": "test.mean.time_average.KNLL"}

    return [stage_one_train, stage_two_train, stage_three_train, stage_four_train], pvmc, encoder, first_knll, [stage_one_info, stage_two_info, stage_three_info, stage_four_info]


def make_training_routine(data_folder, device):
    stages, pvmc, encoder, first_knll, info, = make_training_stages(data_folder, device)

    #stages[2].profile(ModuleList([pvmc, encoder, first_knll]), info[2])
    #raise SystemExit(0)
    routine = Trainer(pvmc, encoder, first_knll, stages = stages)
    return routine, info


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    routine, info = make_training_routine(Path("./experiments/bearings_only/data/"), device)
    routine.fit("first_test",
                    info,
                    True,
                    Path("./experiments/bearings_only/saved_models/pvmc/"))