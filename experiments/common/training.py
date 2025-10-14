from pathlib import Path
import time
import pickle
import json
import torch
from sympy.codegen.cfunctions import expm1
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
import pydpf
import ast
from models.generic_nets.module_list import  ModuleList
from experiments.common.dict_handling import *


class Trainer:

    def __init__(self, *complete_model, stages):
        self.complete_model = ModuleList(complete_model)
        self.stages = stages

    @staticmethod


    def fit(self, run_name, run_info, save_intermediate_models = True, intermediate_folder = None, outputs_save_format = "pickle", verbose = True):
        if not outputs_save_format in ["pickle", "json"]:
            raise ValueError("save_format must be either 'pickle' or 'json'")

        if intermediate_folder is None:
            intermediate_folder = Path().cwd()
        training_start = time.time()
        outputs = None
        for i, t in enumerate(self.stages):
            stage_start = time.time()
            print("===================================")
            print(f"Beginning stage {i+1} of {len(self.stages)}")
            print("===================================")
            print("\n\n\n")
            t.initialise(outputs, run_info[i])
            t.fit(self.complete_model, run_info[i], verbose)
            print("===================================")
            print(f"Done stage {i+1} of {len(self.stages)}")
            if "print" in run_info[i]:
                print_output(t.logged_data, run_info[i]["print"])
            print(f"Time elapsed: {time.time() - stage_start}")
            print(f"Total time elapsed: {time.time() - training_start}")
            print("===================================")
            print("\n")
            if "retain" in run_info[i]:
                outputs = parse_dictionary(t.logged_data, run_info[i]["retain"])

            if "save" in run_info[i]:
                save_info = parse_dictionary(t.logged_data, run_info[i]["save"])
                if outputs_save_format == "pickle":
                    with open(intermediate_folder / f"{run_name}_stage_{i+1}_outputs.pkl", "wb") as f:
                        pickle.dump(save_info, f, pickle.HIGHEST_PROTOCOL)
                if outputs_save_format == "json":
                    with open(intermediate_folder / f"{run_name}_stage_{i+1}_outputs.json", "w") as f:
                        json.dump(save_info, f)
            if save_intermediate_models:
                torch.save(self.complete_model.state_dict(), intermediate_folder / f"{run_name}_stage_{i+1}_model_state.pt")


class TrainingStage:

    def __init__(self,
                 run_func,
                 train_dataset,
                 validation_dataset,
                 test_dataset,
                 optimiser,
                 data_order,
                 lr_scheduler = None,
                 lr_step_freq = "never",
                 ):
        if not lr_step_freq in ["epoch", "opt_step", "never", "all"]:
            raise ValueError("lr_scheduler must be either 'epoch' or 'opt_step' or 'never' or 'all'")
        self.lr_step_freq = lr_step_freq
        self.run_func = run_func
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.optimiser = optimiser
        self.data_order = data_order
        self.lr_scheduler = lr_scheduler
        self.outputs = None
        self.logged_data = {}
        self.stage_output_print = []
        self.stage_output_save = []
        self.stage_output_retain = None

    def clear_data(self):
        del self.logged_data

    def initialise(self, prev_output_dict, run_data):
        pass

    def run_on_step(self):
        pass

    def run_on_epoch(self):
        pass


    def to_test_runner(self):
        return Test_Runner(self.run_func, self.test_dataset, self.data_order)


    def profile(self, complete_model, run_info, sort_by = "cuda_memory_usage"):
        self.logged_data = {}
        self.logged_data["train_batch_size"] = run_info["train"]["batch_size"]
        self.logged_data["validation_batch_size"] = run_info["validation"]["batch_size"]
        self.logged_data["epochs"] = run_info["epochs"]
        self.logged_data["device"] = run_info["device"]

        train_loader = torch.utils.data.DataLoader(self.train_dataset, **get_dataloader_info(run_info["train"]))
        validation_loader = torch.utils.data.DataLoader(self.validation_dataset, **get_dataloader_info(run_info["validation"]))

        device = torch.device(run_info["device"])

        complete_model.train()

        for i, datum in enumerate(train_loader):
            complete_model.update()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as prof:
                with torch.profiler.record_function("fetch input"):
                    data_dict = get_data_dict(self.data_order, datum, device)
                with torch.profiler.record_function("forward"):
                    train_output, batch_dict = self.run_func("train", run_info, **data_dict)
                print(torch.cuda.max_memory_allocated() / 1e6)
                with torch.profiler.record_function("loss"):
                    loss = parse_formula_strip(train_output, run_info["loss"]).mean()
                print(torch.cuda.max_memory_allocated() / 1e6)
                with torch.profiler.record_function("backward"):
                    loss.backward()
                print(torch.cuda.max_memory_allocated() / 1e6)
            print("Training profile")
            print(prof.key_averages().table(sort_by=sort_by))
            break

        for i, datum in enumerate(validation_loader):
            #complete_model.update()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as prof:
                with torch.inference_mode():
                    with torch.profiler.record_function("fetch input"):
                        data_dict = get_data_dict(self.data_order, datum, device)
                    with torch.profiler.record_function("forward"):
                        validation_outputs, val_batch_dict = self.run_func("validation", run_info, **data_dict)
            print("Inference profile")
            print(prof.key_averages().table(sort_by=sort_by))
            break

    def fit(self, complete_model, run_info, verbose = True):

        self.logged_data = {}
        self.logged_data["train_batch_size"] = run_info["train"]["batch_size"]
        self.logged_data["validation_batch_size"] = run_info["validation"]["batch_size"]
        self.logged_data["epochs"] = run_info["epochs"]
        self.logged_data["device"] = run_info["device"]

        train_loader = torch.utils.data.DataLoader(self.train_dataset, **get_dataloader_info(run_info["train"]))
        validation_loader = torch.utils.data.DataLoader(self.validation_dataset, **get_dataloader_info(run_info["validation"]))
        train_iterable = train_loader
        validation_iterable = validation_loader
        if "test" in run_info:
            test_loader = torch.utils.data.DataLoader(self.test_dataset, **get_dataloader_info(run_info["test"]))
            test_iterable = test_loader
            self.logged_data["test_batch_size"] = run_info["test"]["batch_size"]


        try:
            target = run_info["target"]
        except KeyError:
            target = None

        device = torch.device(run_info["device"])

        best_target = torch.inf
        best_dict = complete_model.state_dict()

        batch_dict = {}
        val_batch_dict = {}
        test_batch_dict = {}

        for epoch in range(run_info["epochs"]):
            complete_model.train()
            step_losses = []
            train_logs = {}
            if verbose:
                print(f"Staring epoch {epoch+1} of {run_info["epochs"]}")
                train_iterable = tqdm(train_loader, desc="Training: ")

            for i, datum in enumerate(train_iterable):
                self.optimiser.zero_grad()
                complete_model.update()
                data_dict = get_data_dict(self.data_order,datum, device)
                train_output, batch_dict = self.run_func("train", run_info, **data_dict)
                loss = parse_formula_strip(train_output, run_info["loss"]).mean()
                loss.backward()
                if epoch == 0 and i == 0:
                    for n, p in complete_model.named_parameters():
                        print(n)
                        if p.grad is not None:
                            print(torch.mean(p.grad))
                self.optimiser.step()
                self.run_on_step()
                if self.lr_scheduler is not None and self.lr_step_freq == "opt_step":
                    self.lr_scheduler.step()
                step_losses.append(loss.item())
                train_logs = append_dict(train_logs, dict_to_numpy(train_output), batch_dict)

            if self.lr_scheduler is not None and (self.lr_step_freq == "epoch"):
                    self.lr_scheduler.step()
            if verbose:
                print("Finished training")
                validation_iterable = tqdm(validation_loader, desc="Validating: ")

            complete_model.update()
            complete_model.eval()
            validation_logs = {}
            with torch.inference_mode():
                for datum in validation_iterable:
                    data_dict = get_data_dict(self.data_order, datum, device)
                    validation_outputs, val_batch_dict = self.run_func("validation", run_info, **data_dict)
                    validation_logs = append_dict(validation_logs, dict_to_numpy(validation_outputs), val_batch_dict)

                train_logs["train_loss"] = np.array(step_losses)
                batch_dict["train_loss"] = 0
                mean_train_logs = mean_dict(train_logs, len(self.train_dataset), batch_dict)
                mean_validation_logs = mean_dict(validation_logs, len(self.validation_dataset), val_batch_dict)
                epoch_logs = {"train": {"raw": train_logs, "mean": mean_train_logs}, "validation": {"raw": validation_logs, "mean": mean_validation_logs}}
                if target is not None:
                    t = parse_formula_strip(epoch_logs, target)

                    if t < best_target:
                        best_dict = complete_model.state_dict()
                        best_target = t
                self.logged_data = stack_dict(self.logged_data, epoch_logs)
            self.run_on_epoch()

            if verbose:
                print("Finished Validation")
                for k, v in run_info["print_each_epoch"].items():
                    print(f"{k}: {parse_formula_strip(epoch_logs, v)}")

        if target is not None:
            complete_model.load_state_dict(best_dict, strict=True)

        complete_model.eval()
        if "test" in run_info:
            test_logs = {}
            if verbose:
                test_iterable = tqdm(test_loader, desc="Testing: ")
            with torch.inference_mode():
                for datum in test_iterable:
                    data_dict = get_data_dict(self.data_order, datum, device)
                    test_outputs, test_batch_dict = self.run_func("test", run_info, **data_dict)
                    test_logs = append_dict(test_logs, dict_to_numpy(test_outputs), test_batch_dict)
                mean_test_logs = mean_dict(test_logs, len(self.test_dataset), test_batch_dict)
                self.logged_data = {**self.logged_data, "test": {"raw": test_logs, "mean": mean_test_logs}, **test_logs}
        self.logged_data = self.logged_data | {"final_optim": self.optimiser}






class ExperimentRun(ABC):
    def __init__(self, *args, preprocessors=None):
        super().__init__()
        if preprocessors is None:
            preprocessors = {}
        self.preprocessors = preprocessors

    @abstractmethod
    def run(self, mode, run_info, **data):
        raise NotImplementedError

    @staticmethod
    def _preprocess(preprocessors, mode, run_info, **data):
        processed_data = data
        for k, v in preprocessors.items():
            if isinstance(v, tuple):
                int_data = processed_data | ExperimentRun._preprocess(v[1], mode, run_info, **data)
                processed_data = processed_data | v[0](mode, run_info, **int_data)
            else:
                processed_data[k] = v(mode, run_info, **data)
        return processed_data


    def preprocess_and_run(self, mode, run_info, **data):
        return self.run(mode, run_info, **ExperimentRun._preprocess(self.preprocessors, mode, run_info, **data))


    def __call__(self, mode, run_info, **data):
        return self.preprocess_and_run(mode, run_info, **data)

class VanillaPydpfRun(ExperimentRun):
    def __init__(self, model, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.model = model

    def run(self, mode, run_info, **data):
        if "gradient_regulariser" in run_info[mode]:
            raw_output = self.model(run_info[mode]["n_particles"], run_info[mode]["time_extent"], run_info[mode]["output_function"], run_info[mode]["gradient_regulariser"], **data)
        else:
            raw_output = self.model(run_info[mode]["n_particles"], run_info[mode]["time_extent"], run_info[mode]["output_function"], **data)
        means = {}
        batch_dict = {"time_average": {}}
        for k,v in raw_output.items():
            batch_dict[k] = 1
            batch_dict["time_average"][k] = 0
            means[k] = torch.mean(v, dim=0)
        raw_output["time_average"] = means
        return raw_output, batch_dict

class ParallelRun(ExperimentRun):
    def __init__(self, preprocessors=None, **runs):
        super().__init__(preprocessors=preprocessors)
        self.runs = runs

    def run(self, mode, run_info, **data):
        outputs = {}
        batch_dict = {}
        for k,v in self.runs.items():
            outputs[k], batch_dict[k] = v(mode, run_info[k], **data)
        return outputs, batch_dict
