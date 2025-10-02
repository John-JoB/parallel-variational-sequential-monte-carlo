import pydpf
import torch
from torch import Tensor
import einops

def op_helper(left, right, op, second = False):
    if left.size(0) != right.size(0):
        output = op(left[:-1], right)
        if second:
            return output
        return torch.concat([output, left[-1:]], dim=0)
    return op(left, right)

def tree_recurse(tensor: Tensor, op):
    left_1 = tensor[::2]
    right_1 = tensor[1::2]
    combine_1 = op_helper(left_1, right_1, op)
    left_2 = right_1
    right_2 = tensor[2::2]
    combine_2 = op_helper(left_2, right_2, op, True)
    combine_2 = torch.concat([tensor[0:1], combine_2], dim=0)
    new_tensor = [combine_2, combine_1]
    dims = "".join([f"dim_{i} " for i in range(combine_1.dim() - 2)])
    return einops.rearrange(new_tensor, f'p t s {dims} -> t (p s) {dims}')

def op_helper_de(left, right, op):
    if left.size(0) != right.size(0):
        output = op(left[:-1], right)
        return torch.concat([output, left[-1:]], dim=0)
    return op(left, right)

def expand_to_size(tensor, template):
    extra_dims = template.dim() - tensor.dim()
    output = pydpf.multiple_unsqueeze(tensor, extra_dims, 0)
    return output.expand_as(template)

def tree_recurse_de(tensor, op, identity):
    left_1 = tensor[::2]
    right_1 = tensor[1::2]
    combine_1 = op_helper_de(left_1, right_1, op)
    left_2 = right_1
    right_2 = tensor[2::2]
    combine_2 = op_helper_de(left_2, right_2, op)
    combine_2 = torch.concat([tensor[0:1], combine_2], dim=0)
    if combine_1.size(0) != combine_2.size(0):
        idt = expand_to_size(identity, combine_1[:1])
        combine_1 = torch.concat([combine_1, idt], dim=0)
    new_tensor = [combine_2, combine_1]
    dims = "".join([f"dim_{i} " for i in range(combine_1.dim() - 2)])
    return einops.rearrange(new_tensor, f'p t s {dims} -> t (p s) {dims}')

def parallel_associative_scan_de(tensor:Tensor, op, identity = 0):
    if not isinstance(identity, Tensor):
        identity = torch.tensor(identity)
    running_tensor = tensor.unsqueeze(1)
    while running_tensor.size(0) >2:
        running_tensor = tree_recurse_de(running_tensor, op, identity)
    forward_tensor = running_tensor[0, :tensor.size(0)]
    backward_tensor = torch.concat([forward_tensor[-1:], running_tensor[1, :tensor.size(0)-1]], dim=0)
    return forward_tensor, backward_tensor


def parallel_associative_scan(tensor:Tensor, op):
    if tensor.size(0) < 2:
        return tensor
    running_tensor = tensor.unsqueeze(1)
    while running_tensor.size(0) > 2:
        running_tensor = tree_recurse(running_tensor, op)
    remaining_size = tensor.size(0) - running_tensor.size(1)
    remaining_results = op(running_tensor[0,:remaining_size], running_tensor[1,:remaining_size])
    return torch.concat([running_tensor[0], remaining_results], dim=0)

def parallel_associative_reduce(tensor:Tensor, op, keepdim=False):
    if tensor.size(0) < 2:
        return tensor
    running_tensor = tensor
    while running_tensor.size(0) > 1:
        left = running_tensor[::2]
        right = running_tensor[1::2]
        if left.size(0) != right.size(0):
            running_tensor = torch.concat([op(left[:-1], right), left[-1:]], dim=0)
            continue
        running_tensor = op(left, right)
    if keepdim is False:
        return running_tensor.squeeze(0)
    return running_tensor




