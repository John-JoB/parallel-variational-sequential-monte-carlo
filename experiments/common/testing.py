import pydpf
import torch


class Test_Runner:
    def __init__(self, *complete_model,
                 run_func,
                 dataset,
                 data_order):
        self.run_func = run_func
        self.dataset = dataset
        self.data_order = data_order

    def test(self, ):

