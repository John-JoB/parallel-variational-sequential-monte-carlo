from models.generic_nets.module_list import ModuleList
from experiments.common.dict_handling import  *
from pathlib import Path
import pickle
import json
from pydpf import State, Weight
from matplotlib import pyplot as plt
from experiments.common.dict_handling import semi_deep_copy
import time

class Test_Runner:
    def __init__(self,
                 *complete_model,
                 run_func,
                 dataset,
                 data_order):
        self.complete_model = ModuleList(complete_model)
        self.run_func = run_func
        self.dataset = dataset
        self.data_order = data_order
        self.logged_data = {}

    def test(self,run_name, info, outputs_save_format = "pickle", save_folder = None):
        if save_folder is None:
            save_folder = Path().cwd()
        print("===================================")
        print(f"Testing Model -- {run_name}")
        print("===================================")
        self.run_test(info)
        if "print" in info:
            print_output(self.logged_data, info["print"])
        if "save" in info:
            save_info = parse_dictionary(self.logged_data, info["save"])
            if outputs_save_format == "pickle":
                with open(save_folder / f"{run_name}_outputs.pkl", "wb") as f:
                    pickle.dump(save_info, f, pickle.HIGHEST_PROTOCOL)
            if outputs_save_format == "json":
                with open(save_folder / f"{run_name}_outputs.json", "w") as f:
                    json.dump(save_info, f)
        if "return" in info:
            return parse_dictionary(self.logged_data, info["return"])

    def draw_particles(self, info, dim1, dim2, dir_dim=None, xlim=None, ylim=None, plotted_gt=None):
        info = semi_deep_copy(info)
        info["output_function"] = {"state": State(), "weight": Weight()}
        info["shuffle"] = False
        print(info)
        t_dataset = self.dataset
        self.dataset = self.dataset.select(range(1))
        info["collate_fn"] = self.dataset.collate
        self.run_test(info)
        print(self.logged_data)
        weight = np.squeeze(self.logged_data["weight"])
        state = np.squeeze(self.logged_data["state"])
        self.dataset = t_dataset

        for i in range(len(weight)):
            fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if dir_dim is None:
                ax.scatter(state[i, :, dim1], state[i, :, dim2], color="b", alpha=np.exp(weight[i] - np.max(weight[i])))
            else:
                ax.quiver(state[i, :, dim1], state[i, :, dim2], np.cos(state[i, :, dir_dim]), np.sin(state[i, :, dir_dim]), pivot="mid", color="b", alpha=np.exp(weight[i] - np.max(weight[i])))
            if plotted_gt is not None:
                if dir_dim is None:
                    ax.scatter(plotted_gt[i, dim1], plotted_gt[i, dim2], color="r")
                else:
                    ax.quiver(plotted_gt[i, dim1], plotted_gt[i, dim2], np.cos(plotted_gt[i, dir_dim]), np.sin(plotted_gt[i, dir_dim]), pivot="mid", color="r")
            fig.show()






    def initialise(self, run_info):
        pass

    @staticmethod
    def _duplicate_dictionary(dictionary):
        if "test" not in dictionary:
            dictionary["test"] = {}
        for k, v in dictionary.items():
            if k == "test":
                continue
            if k not in dictionary["test"]:
                dictionary["test"][k] = v
        for k, v in dictionary["test"].items():
            if k not in dictionary:
                dictionary[k] = v

    def run_test(self, run_info):
        self.initialise(run_info)
        run_info = semi_deep_copy(run_info)
        Test_Runner._duplicate_dictionary(run_info)
        self.logged_data = {}
        self.logged_data["batch_size"] = run_info["batch_size"]
        self.logged_data["device"] = run_info["device"]
        dataloader = torch.utils.data.DataLoader(self.dataset, **get_dataloader_info(run_info))
        device = torch.device(run_info["device"])
        self.complete_model.update()
        start_time = time.time()
        with torch.inference_mode():
            test_logs = {}
            for datum in dataloader:
                data_dict = get_data_dict(self.data_order, datum, device)
                test_outputs, test_batch_dict = self.run_func("test", run_info, **data_dict)
                test_logs = append_dict(test_logs, dict_to_numpy(test_outputs), test_batch_dict)
            mean_test_logs = mean_dict(test_logs, len(self.dataset), test_batch_dict)
            self.logged_data = {**self.logged_data, "mean": mean_test_logs, **test_logs}
        end_time = time.time()
        self.logged_data["time"] = end_time - start_time

