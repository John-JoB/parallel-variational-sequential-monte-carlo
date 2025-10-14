import ast
import numpy as np
import torch


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
        try:
            out[k] = parse_formula_strip(read_dict, v)
        except Exception as e:
            out[k] = e
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


def get_data_dict(data_order, data, device):
    if not isinstance(data, tuple):
        data = (data,)
    return {cat: d.to(device=device) for cat, d in zip(data_order, data)}

def dict_to_numpy(input_dict):
    output_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            output_dict[k] = dict_to_numpy(v)
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

def mean_dict(input_dict, total_items, batch_dict):
    output = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            output[k] = mean_dict(v, total_items, batch_dict[k])
            continue
        try:
            output[k] = np.sum(v / total_items, axis=batch_dict[k])
            if not isinstance(output[k], np.ndarray):
                output[k] = np.array(output[k])
        except TypeError:
            pass
    return output


def append_dict(a, b, batch_dict):
    output_dict = {}
    for k in b:
        if not k in a:
            if isinstance(b[k], dict):
                output_dict[k] = append_dict({}, b[k], batch_dict[k])
            elif isinstance(b[k], np.ndarray):
                output_dict[k] = b[k]
            else:
                output_dict[k] = [b[k]]
            continue
        if isinstance(a[k], dict):
            output_dict[k] = append_dict(a[k], b[k], batch_dict[k])
        elif isinstance(a[k], np.ndarray):
            output_dict[k] = np.concatenate((a[k], b[k]), axis=batch_dict[k])
        elif isinstance(a[k], list):
            output_dict[k] = a[k].append(b[k])
        else:
            assert(False)
    return output_dict


def stack_dict(a, b):
    output_dict = {}
    for k in b:
        if not k in a:
            if isinstance(b[k], dict):
                output_dict[k] = stack_dict({}, b[k])
            elif isinstance(b[k], np.ndarray):
                output_dict[k] = b[k][None, ...]
            else:
                output_dict[k] = [b[k]]
            continue
        if isinstance(a[k], dict):
            output_dict[k] = stack_dict(a[k], b[k])
        elif isinstance(a[k], np.ndarray):
            output_dict[k] = np.concatenate((a[k], b[k][None, ...]), axis=0)
        elif isinstance(a[k], list):
            output_dict[k] = a[k].append(b[k])
        else:
            print(a[k].__class__.__name__)
            print(k)
            assert (False)
    return output_dict

def get_dataloader_info(full_dict):
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