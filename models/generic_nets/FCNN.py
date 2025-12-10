import pydpf
import torch
from pydpf import Module
from typing import Iterable
from collections import OrderedDict
from models.generic_nets.activation import activation_function_from_string





class FCNN(Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int|Iterable[int], activation_function: Module|Iterable[Module]|str|Iterable[str], output_function: Module|None, n_hidden_layers:int|None, device):
        super().__init__()
        layers = OrderedDict()
        if output_function is None:
            if not (isinstance(activation_function, torch.nn.Module) or isinstance(activation_function, str)):
                raise TypeError("If there is no output function, then the output is set to the activation function, but the activation function is not a Module.")
            output_function = activation_function

        output_function = activation_function_from_string(output_function)


        if hidden_dim is None or hidden_dim == 0 or (hasattr(hidden_dim, "__len__") and len(hidden_dim) == 0) or n_hidden_layers == 0:
            layers["LT"] = torch.nn.Linear(input_dim, output_dim)
            layers["Output"] = output_function
            self.net = torch.nn.Sequential(layers)
            self.to(device=device)
            return

        if not hasattr(activation_function, "forward"):
            activation_function = [activation_function] * n_hidden_layers

        for i, fun in enumerate(activation_function):
            activation_function[i] = activation_function_from_string(fun)

        if hasattr(hidden_dim, "__len__"):
            n_hidden_layers = len(hidden_dim)
            if hasattr(activation_function, "__len__") & len(hidden_dim) != len(activation_function):
                raise AssertionError("The length of hidden_dim must be equal to the length of activation_function")
        elif hasattr(activation_function, "__len__") and not isinstance(hidden_dim, Iterable):
            n_hidden_layers = len(activation_function)

        if not hasattr(hidden_dim, "__len__"):
            hidden_dim = [hidden_dim] * n_hidden_layers





        layers["Input LT"] = torch.nn.Linear(input_dim, hidden_dim[0])
        for i in range(n_hidden_layers):
            layers[f"Activation {i+1}"] = activation_function[i]
            if i == n_hidden_layers - 1:
                layers[f"Output LT"] = torch.nn.Linear(hidden_dim[i], output_dim)
            else:
                layers[f"LT {i+2}"] = torch.nn.Linear(hidden_dim[i], hidden_dim[i+1])
        layers["Output Function"] = output_function
        self.net = torch.nn.Sequential(layers)
        self.to(device=device)


    def forward(self, x):
        return self.net(x)

"""
class ResNet(pydpf.Module):

    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int|Iterable[int], activation_function: Module|Iterable[Module]|str|Iterable[str], output_function: Module|None, n_hidden_layers:int|None, device, block_size):
        if not input_dim == hidden_dim:
            self.proj = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.proj = lambda x: x
        for block in range((n_hidden_layers+1)//block_size):



        super().__init__(input_dim, output_dim, hidden_dim, activation_function, output_function, n_hidden_layers, device)



    def forward(self, x):
        print(torch.sum(torch.isnan(super().forward(x))))
        print(torch.sum(torch.isnan(self.proj(x))))
        return super().forward(x) + self.proj(x)

"""