import pydpf
import torch

class ConvProp(pydpf.Module):
    def __init__(self, kernel_size, input_dim, output_dim, ):
        super().__init__()
