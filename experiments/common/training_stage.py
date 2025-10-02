from pathlib import Path
import time
import pickle
import json
import torch
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
import pydpf
import ast


class _CustomNameSpace:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _CustomNameSpace(v))
            else:
                setattr(self, k, v)


class _CustomTreeParser:
    def __init__(self, ):
        self.names = []

    def search_tree(self, node):
        if isinstance(node, ast.Name):
            if node.id != "torch":
                self.names.append(node.id)
        else:
            for child in ast.iter_child_nodes(node):
                self.search_tree(child)

def parse_formula_strip(dictionary, formula_strip):
    fs = ast.parse(formula_strip, mode='eval')
    ns = _CustomNameSpace(dictionary)
    names_ob = _CustomTreeParser()
    names_ob.search_tree(fs)
    names = names_ob.names
    env = {k: getattr(ns, k) for k in names}
    env["torch"] = "torch"
    compiled = compile(fs, filename="<ast>", mode="eval")
    return eval(compiled, env)

def parse_dictionary(read_dict, key_dict):
    out = {}
    for k,v in key_dict.items():
        out[k] = parse_formula_strip(read_dict, v)
    return out

def print_output(read_dict, key_dict):
    d = parse_dictionary(read_dict, key_dict)
    for k, v in d.items():
        print(f"{k}: {v}")


def index_by_compound_key(d, k):
    sub_keys = k.split(".")
    c_d = d
    for k in sub_keys:
        c_d = c_d[k]
    return c_d

class _ModuleList(pydpf.Module):

    def __init__(self, model_list):
        super().__init__()
        occurrence_dict = {}
        for model in model_list:
            if isinstance(model, torch.nn.Module):
                name = model.__class__.__name__
                if name in occurrence_dict:
                    occurrence_dict[name] += 1
                else:
                    occurrence_dict[name] = 0
                setattr(self, f"{name}_{occurrence_dict[name]}", model)


class Trainer:

    def __init__(self, *complete_model, stages):
        self.complete_model = _ModuleList(complete_model)
        self.stages = stages

    @staticmethod
    def _read_dict_keys(dict, keys):
        splits = [key.split(":") for key in keys]
        output = {}
        vs = [index_by_compound_key(dict, key) for key in keys]
        for split, v in zip(splits, vs):
            wd = output
            for i in range(len(split) - 1):
                if split[i] not in wd:
                    wd[split[i]] = {}
                elif not isinstance(wd[split[i]], dict):
                    raise ValueError("Key collision, check that the keys are distinct")
                wd = wd[split[i]]
            wd[split[-1]] = v
        return output

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

    def _get_data_dict(self, data, device):
        if not isinstance(data, tuple):
            data = (data,)
        return {cat: d.to(device=device) for cat, d in zip(self.data_order, data)}

    @staticmethod
    def _dict_to_numpy(input_dict):
        output_dict = {}
        for k, v in input_dict.items():
            if isinstance(v, dict):
                output_dict[k] = TrainingStage._dict_to_numpy(v)
            elif isinstance(v, np.ndarray):
                output_dict[k] = v
            elif isinstance(v, torch.Tensor):
                output_dict[k] = v.detach().cpu().numpy()
            else:
                try:
                    output_dict[k] = np.array(v)
                except TypeError:
                    output_dict[k] = v
        return output_dict

    @staticmethod
    def _mean_dict(input_dict, total_items, batch_dict):
        output = {}
        for k, v in input_dict.items():
            if isinstance(v, dict):
                output[k] = TrainingStage._mean_dict(v, total_items, batch_dict[k])
                continue
            try:
                output[k] = np.sum(v / total_items, axis=batch_dict[k])
                if not isinstance(output[k], np.ndarray):
                    output[k] = np.array(output[k])
            except TypeError:
                pass
        return output

    @staticmethod
    def _append_dict(a, b, batch_dict):
        output_dict = {}
        for k in b:
            if not k in a:
                if isinstance(b[k], dict):
                    output_dict[k] = TrainingStage._append_dict({}, b[k], batch_dict[k])
                elif isinstance(b[k], np.ndarray):
                    output_dict[k] = b[k]
                else:
                    output_dict[k] = [b[k]]
                continue
            if isinstance(a[k], dict):
                output_dict[k] = TrainingStage._append_dict(a[k], b[k], batch_dict[k])
            elif isinstance(a[k], np.ndarray):
                output_dict[k] = np.concatenate((a[k], b[k]), axis=batch_dict[k])
            elif isinstance(a[k], list):
                output_dict[k] = a[k].append(b[k])
            else:
                assert(False)
        return output_dict

    @staticmethod
    def _stack_dict(a, b):
        output_dict = {}
        for k in b:
            if not k in a:
                if isinstance(b[k], dict):
                    output_dict[k] = TrainingStage._stack_dict({}, b[k])
                elif isinstance(b[k], np.ndarray):
                    output_dict[k] = b[k][None, ...]
                else:
                    output_dict[k] = [b[k]]
                continue
            if isinstance(a[k], dict):
                output_dict[k] = TrainingStage._stack_dict(a[k], b[k])
            elif isinstance(a[k], np.ndarray):
                output_dict[k] = np.concatenate((a[k], b[k][None, ...]), axis=0)
            elif isinstance(a[k], list):
                output_dict[k] = a[k].append(b[k])
            else:
                print(a[k].__class__.__name__)
                print(k)
                assert (False)
        return output_dict



    def _get_dataloader_info(self, full_dict):
        options = ["batch_size",
                   "shuffle",
                   "num_workers",
                   "pin_memory",
                   "sampler",
                   "collate_fn",
                   "pin_memory",
                   "drop_last",
                   "timeout",
                   "worker_init_fn",
                   "multiprocessing_context",
                   "generator",
                   "prefetch_factor",
                   "persistent_workers",
                   "pin_memory_device"]
        return {k:v for k,v in full_dict.items() if k in options}


    def fit(self, complete_model, run_info, verbose = True):

        self.logged_data = {}
        self.logged_data["train_batch_size"] = run_info["train"]["batch_size"]
        self.logged_data["validation_batch_size"] = run_info["validation"]["batch_size"]
        self.logged_data["epochs"] = run_info["epochs"]
        self.logged_data["device"] = run_info["device"]

        train_loader = torch.utils.data.DataLoader(self.train_dataset, **self._get_dataloader_info(run_info["train"]))
        validation_loader = torch.utils.data.DataLoader(self.validation_dataset, **self._get_dataloader_info(run_info["validation"]))
        train_iterable = train_loader
        validation_iterable = validation_loader
        if "test" in run_info:
            test_loader = torch.utils.data.DataLoader(self.test_dataset, **self._get_dataloader_info(run_info["test"]))
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
            step_losses = []
            train_logs = {}
            if verbose:
                print(f"Staring epoch {epoch+1} of {run_info["epochs"]}")
                train_iterable = tqdm(train_loader, desc="Training: ")

            for i, datum in enumerate(train_iterable):
                self.optimiser.zero_grad()
                complete_model.update()
                data_dict = self._get_data_dict(datum, device)
                train_output, batch_dict = self.run_func("train", run_info, **data_dict)
                loss = parse_formula_strip(train_output, run_info["loss"]).mean()
                loss.backward()
                self.optimiser.step()
                self.run_on_step()
                if self.lr_scheduler is not None and self.lr_step_freq == "opt_step":
                    self.lr_scheduler.step()
                step_losses.append(loss.item())
                train_logs = TrainingStage._append_dict(train_logs, TrainingStage._dict_to_numpy(train_output), batch_dict)

            if self.lr_scheduler is not None and (self.lr_step_freq == "epoch"):
                    self.lr_scheduler.step()
            if verbose:
                print("Finished training")
                validation_iterable = tqdm(validation_loader, desc="Validating: ")

            complete_model.update()

            validation_logs = {}
            with torch.inference_mode():
                for datum in validation_iterable:
                    data_dict = self._get_data_dict(datum, device)
                    validation_outputs, val_batch_dict = self.run_func("validation", run_info, **data_dict)
                    validation_logs = TrainingStage._append_dict(validation_logs, TrainingStage._dict_to_numpy(validation_outputs), val_batch_dict)

                train_logs["train_loss"] = np.array(step_losses)
                batch_dict["train_loss"] = 0
                mean_train_logs = self._mean_dict(train_logs, len(self.train_dataset), batch_dict)
                mean_validation_logs = self._mean_dict(validation_logs, len(self.validation_dataset), val_batch_dict)
                epoch_logs = {"train": {"raw": train_logs, "mean": mean_train_logs}, "validation": {"raw": validation_logs, "mean": mean_validation_logs}}
                if target is not None:
                    t = parse_formula_strip(epoch_logs, target)
                    if t < best_target:
                        best_dict = complete_model.state_dict()
                        best_target = t
                self.logged_data = TrainingStage._stack_dict(self.logged_data, epoch_logs)
            self.run_on_epoch()

            if verbose:
                print("Finished Validation")
                for k, v in run_info["print_each_epoch"].items():
                    print(f"{k}: {parse_formula_strip(epoch_logs, v)}")

        if "test" in run_info:
            test_logs = {}
            if verbose:
                test_iterable = tqdm(test_loader, desc="Testing: ")
            with torch.inference_mode():
                if target is not None:
                    complete_model.load_state_dict(best_dict, strict=True)
                for datum in test_iterable:
                    data_dict = self._get_data_dict(datum, device)
                    test_outputs, test_batch_dict = self.run_func("test", run_info, **data_dict)
                    test_logs = TrainingStage._append_dict(test_logs, TrainingStage._dict_to_numpy(test_outputs), test_batch_dict)
                mean_test_logs = self._mean_dict(test_logs, len(self.test_dataset), test_batch_dict)
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
