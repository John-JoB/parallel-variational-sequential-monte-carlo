import pydpf
import torch

class ModuleList(pydpf.Module):

    def __init__(self, model_list):
        super().__init__()
        occurrence_dict = {}
        for model in model_list:
            if isinstance(model, torch.nn.Module):
                name = model.__class__.__name__
                if name in occurrence_dict:
                    occurrence_dict[name] += 1
                else:
                    occurrence_dict[name] = 0
                setattr(self, f"{name}_{occurrence_dict[name]}", model)