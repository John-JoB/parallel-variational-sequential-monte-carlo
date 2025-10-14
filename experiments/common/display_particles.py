import pydpf
import torch
import numpy as np
from matplotlib import pyplot as plt



def _get_data_dict(data, device, data_cats):
    if not isinstance(data, tuple):
        data = (data,)
    return {cat: d.to(device=device) for cat, d in zip(data_cats, data)}

def draw_particles(complete_model, run_func, run_info, dataset, data_order, dims_to_plot, dir_dim=None):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate)
    complete_model.update()
    try:
        complete_model.load_state_dict(torch.load(run_info["weights_path"]))
    except KeyError:
        pass
    complete_model.update()
    for n, p in complete_model.named_parameters():
        print(n)
        print(p)

    run_info["test"]["output_function"] = {"state": pydpf.State(), "weight": pydpf.Weight()}
    result_dict = {}
    gt_exists = False
    with torch.inference_mode():
        for i, datum in enumerate(dataloader):
            data_dict = _get_data_dict(datum, run_info["device"], data_order)
            result_dict, _ = run_func("test", run_info, **data_dict)
            if "ground_truth" in data_order:
                gt_exists = True
                ground_truth = data_dict["ground_truth"]
            break
    weight = result_dict["weight"].squeeze().cpu().numpy()
    state_1 = result_dict["state"][..., dims_to_plot[0]].squeeze().cpu().numpy()
    state_2 = result_dict["state"][..., dims_to_plot[1]].squeeze().cpu().numpy()
    if dir_dim is not None:
        state_a = result_dict["state"][..., dir_dim].squeeze().cpu().numpy()
        cos_state_a = np.cos(state_a)
        sin_state_a = np.sin(state_a)
    if gt_exists:
        gt_state_1 = ground_truth[..., dims_to_plot[0]].cpu().numpy()
        gt_state_2 = ground_truth[..., dims_to_plot[1]].cpu().numpy()
        if dir_dim is not None:
            state_a = ground_truth[..., dir_dim].cpu().numpy()
            gt_cos_state_a = np.cos(state_a)
            gt_sin_state_a = np.sin(state_a)



    fig_arr = []
    for i in range(len(weight)):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        if gt_exists:
            if dir_dim is None:
                ax.scatter(gt_state_1[i], gt_state_2[i], color="r")
            else:
                ax.quiver(gt_state_1[i], gt_state_2[i], gt_cos_state_a[i], gt_sin_state_a[i], pivot = "mid", color="r")
        if dir_dim is None:
            ax.scatter(state_1[i], state_2[i], color="b", alpha=np.exp(weight[i] - np.max(weight[i])))
        else:
            ax.quiver(state_1[i], state_2[i], cos_state_a[i], sin_state_a[i], pivot = "mid", color="b", alpha=np.exp(weight[i] - np.max(weight[i])))
        fig_arr.append(fig)
    return fig_arr





