from experiments.common.training import ExperimentRun


class TCVAERun(ExperimentRun):
    def __init__(self, model, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.model = model

    def run(self, mode, run_info, **data):
        l1_loss, kld  = self.model.forward(run_info[mode]["time_extent"], **data)
        batch_dict = {"time_average": {}}
        raw_output = {"time_average": {}}
        batch_dict["l1_loss"] = 0
        batch_dict["kld"] = 0
        batch_dict["time_average"]["l1_loss"] = 0
        batch_dict["time_average"]["kld"] = 0
        raw_output["l1_loss"] = l1_loss
        raw_output["kld"] = kld
        raw_output["time_average"]["l1_loss"] = l1_loss
        raw_output["time_average"]["kld"] = kld
        return raw_output, batch_dict
