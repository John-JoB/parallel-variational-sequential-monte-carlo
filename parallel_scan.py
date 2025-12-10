import pydpf
import torch
from torch import Tensor
import einops

def op_helper(op,left_list, right_list, second = False):
    if left_list[0].size(0) != right_list[0].size(0):
        output = op([left[:-1] for left in left_list], right_list)
        if second:
            return output
        return [torch.concat([o, left[-1:]], dim=0) for o,left in zip(output, left_list)]
    return op(left_list, right_list)

def tree_recurse(op, tensor_list):
    left_1 = [tensor[::2] for tensor in tensor_list]
    right_1 = [tensor[1::2] for tensor in tensor_list]
    combine_1 = list(op_helper(op, left_1, right_1))
    left_2 = right_1
    right_2 = [tensor[2::2] for tensor in tensor_list]
    combine_2 = list(op_helper(op, left_2, right_2, True))
    combine_2 = [torch.concat([tensor[0:1], c2], dim=0) for tensor,c2 in zip(tensor_list, combine_2)]
    dims = "".join([f"dim_{i} " for i in range(combine_1[0].dim() - 2)])
    return [einops.rearrange([c2, c1], f'p t s {dims} -> t (p s) {dims}') for c1,c2 in zip(combine_1, combine_2)]

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


def parallel_associative_scan(op, *tensor_list):
    sequence_length = tensor_list[0].size(0)
    for tensor in tensor_list:
        if tensor.size(0) != sequence_length:
            raise ValueError("All tensors must have the same length")
    if sequence_length < 2:
        return tensor_list
    running_tensor = [tensor.unsqueeze(1) for tensor in tensor_list]
    while running_tensor[0].size(0) > 2:
        running_tensor = tree_recurse(op, running_tensor)
    remaining_size = tensor_list[0].size(0) - running_tensor[0].size(1)
    remaining_results = op([tensor[0,:remaining_size] for tensor in running_tensor], [tensor[1,:remaining_size] for tensor in running_tensor])
    return [torch.concat([rt[0], rr], dim=0) for rt,rr in zip(running_tensor, remaining_results)]

def single_element_par(op, keepdim, tensor):
    if tensor.size(0) < 2:
        if keepdim:
            return tensor
        else:
            return tensor.squeeze(0)
    running_tensor= tensor
    while running_tensor.size(0) > 1:
        left = running_tensor[::2]
        right = running_tensor[1::2]
        if left.size(0) == right.size(0):
            running_tensor = op(left, right)
            continue
        temp = op(left[:-1], right)
        running_tensor = torch.concat([temp, left[-1:]], dim=0)
    if keepdim is False:
        return running_tensor.squeeze(0)
    return running_tensor

def parallel_associative_reduce(op, reshaper, keepdim=False, *tensor_list):
    if len(tensor_list) == 1:
        return single_element_par(op, keepdim, tensor_list[0])
    sequence_length = tensor_list[0].size(0)
    for tensor in tensor_list:
        if tensor.size(0) != sequence_length:
            raise ValueError("All tensors must have the same length")
    if tensor_list[0].size(0) < 2:
        return tensor_list
    running_tensor_list = tensor_list
    while running_tensor_list[0].size(0) > 1:
        left = [running_tensor[::2] for running_tensor in running_tensor_list]
        right = [running_tensor[1::2] for running_tensor in running_tensor_list]
        if left[0].size(0) == right[0].size(0):
            running_tensor_list = op(left, right)
            continue
        short_left = [t[:-1] for t in left]
        temp = op(short_left, right)
        running_tensor_list = tuple([torch.concat([t, reshaper(l[-1:])], dim=0) for t, l in zip(temp, left)])
    if keepdim is False:
        return tuple([t.squeeze(0) for t in running_tensor_list])
    return running_tensor_list




