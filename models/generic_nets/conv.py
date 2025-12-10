import pydpf
import torch
import einops
from models.generic_nets.activation import activation_function_from_string

class ConvWithEdgeHandling(pydpf.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_offset, left_input_size, right_input_size, groups=1, bias=True, device="cpu", dtype=None):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd number")
        if kernel_offset > kernel_size // 2:
            raise ValueError("kernel_offset must be less than half the size of the kernel")
        even_out_pad = kernel_size // 2
        self.left_out_pad = even_out_pad + kernel_offset
        self.right_out_pad = even_out_pad - kernel_offset
        if self.left_out_pad > 0:
            self.need_left = True
            self.left_linear = torch.nn.Linear(left_input_size*in_channels, self.left_out_pad*out_channels, bias=bias, device=device, dtype=dtype)
        if self.right_out_pad > 0:
            self.need_right = True
            self.right_linear = torch.nn.Linear(right_input_size*in_channels, self.right_out_pad*out_channels, bias=bias, device=device, dtype=dtype)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, bias=bias, device=device, dtype=dtype)
        self.left_input_size = left_input_size
        self.right_input_size = right_input_size


    def forward(self, series):
        conv = self.conv(series)
        if self.need_left:
            left_out = self.left_linear(torch.flatten(series[..., :self.left_input_size], start_dim=1, end_dim=-1))
            left_out = einops.rearrange(left_out, 'b (d t) -> b d t', t = self.left_out_pad)
        if self.need_right:
            right_out = self.right_linear(torch.flatten(series[..., -self.right_input_size:], start_dim=1, end_dim=-1))
            right_out = einops.rearrange(right_out, 'b (d t) -> b d t', t = self.right_out_pad)
            if self.need_left:
                return torch.cat((left_out, conv, right_out), dim=-1)
            else:
                return torch.cat((conv, right_out), dim=-1)
        else:
            return torch.cat((left_out, conv), dim=-1)

class LinearWithInvertedDim(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device="cpu", dtype=None):
        super(LinearWithInvertedDim, self).__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input):
        rearr_input = einops.rearrange(input, 'b d t -> b t d')
        rearr_output = super().forward(rearr_input)
        return einops.rearrange(rearr_output, 'b t d -> b d t')

class ConvEncoder(pydpf.Module):
    def __init__(self, layers_info, device, dtype = torch.float32):
        super().__init__()
        layers = []
        for i, layer in enumerate(layers_info):
            if layer["type"] != "dropout" and layer["type"] != "batchnorm":
                layer["dtype"] = dtype
                layer["device"] = device
                activation = activation_function_from_string(layer["activation"])
                del layer["activation"]
            type = layer["type"]
            del layer["type"]
            if type == "conv":
                layers.append(ConvWithEdgeHandling(**layer))
                layers.append(activation)
            if type == "linear":
                layers.append(LinearWithInvertedDim(**layer))
                layers.append(activation)
            if type == "dropout":
                layers.append(torch.nn.Dropout1d(**layer))
            if type == "batchnorm":
                layers.append(torch.nn.BatchNorm1d(**layer))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        rearr_input = einops.rearrange(input, 't b d -> b d t')
        rearr_output = self.layers(rearr_input)
        return einops.rearrange(rearr_output, 'b d t -> t b d')