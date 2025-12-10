from pydpf import Module
import torch

class _l2norm(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.linalg.vector_norm(x, p=2, dim=-1, keepdim=True)

class _normalise(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x / torch.linalg.vector_norm(x, p=2, dim=-1, keepdim=True)


def activation_function_from_string(str_fun):
    if not isinstance(str_fun, str):
        return str_fun
    caseless_string = str_fun.casefold()
    match caseless_string:
        case 'relu':
            return torch.nn.ReLU()
        case 'sigmoid':
            return torch.nn.Sigmoid()
        case 'tanh':
            return torch.nn.Tanh()
        case 'prelu':
            return torch.nn.PReLU()
        case 'leaky_relu':
            return torch.nn.LeakyReLU()
        case 'softplus':
            return torch.nn.Softplus()
        case 'softmax':
            return torch.nn.Softmax()
        case 'softmin':
            return torch.nn.Softmin()
        case 'l2norm':
            return _l2norm()
        case 'normalise':
            return _normalise()
        case 'id':
            return torch.nn.Identity()
        case 'swish':
            return torch.nn.SiLU()
    raise ValueError(f"Activation function '{str_fun}' not recognized")