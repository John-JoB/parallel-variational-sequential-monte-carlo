import einops
import pydpf
import torch
from matplotlib import pyplot as plt
from pytorch_forecasting import autocorrelation
from pathlib import Path
import numpy as np
import matplotlib.lines as mlines
from sympy.printing.pretty.pretty_symbology import line_width

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate_paths(SSM, time_extent):
    #series_metadata = torch.ones((100, 1), device=device, dtype=torch.float32)
    pydpf.simulate_and_save("./experiments/SPX/data/paths.csv", SSM, time_extent=time_extent, n_trajectories=1000, batch_size=128, device=device, bypass_ask=True)
    paths = pydpf.StateSpaceDataset("./experiments/SPX/data/paths.csv")
    return paths

def from_log_ret(observation, **data):
    log_ret = torch.cumsum(observation, dim=0)
    return torch.exp(log_ret)

def make_skewness(returns):
    sample_mean = torch.mean(returns, dim=0)
    third_moment = torch.mean((returns - sample_mean)**3, dim=0)
    second_moment = torch.mean((returns - sample_mean)**2, dim=0)
    return (third_moment/torch.pow((second_moment * (returns.size(0) /( returns.size(0) - 1))), 1.5)).cpu().numpy()


def make_kurtosis(returns):
    sample_mean = torch.mean(returns, dim=0)
    fourth_moment = torch.mean((returns - sample_mean)**4, dim=0)
    second_moment = torch.mean((returns - sample_mean)**2, dim=0)
    sample_var = (returns.size(0) /( returns.size(0) - 1)) * second_moment
    return (fourth_moment/(sample_var**2)).cpu().numpy()


def hist_and_true_value(ests, true_value, label):
    fig, ax = plt.subplots()
    ax.hist(ests, bins=ests.shape[0]//50, density=True, alpha = 0.6, color="dimgrey", label="Simulated SPX")
    for i, v in enumerate(true_value):
        if i == 0:
            ax.axvline(v, linestyle="--", linewidth=2, color="r", label="True SPX")
        else:
            ax.axvline(v, linestyle="--", linewidth=2, color="r")
    ax.set_xlabel(label)
    ax.set_ylabel("frequency")
    ax.legend()
    plt.title(f"Histogram of simulated trajectory {label}")
    plt.show()

def make_ret_hist(spx, sim):
    fig, ax = plt.subplots()
    ax.hist(sim.flatten(), bins=50, density=True, alpha = 0.4, color="tab:orange", label="Simulated SPX")
    ax.hist(spx.flatten(), bins=50, density=True, alpha=0.4, color="tab:blue", label="SPX")
    ax.set_xlabel("return")
    ax.set_ylabel("frequency")
    ax.legend()
    ax.set_xlim(-0.1, 0.1)
    plt.title(f"Distribution of returns")
    plt.show()

def plot_auto_corrs(returns, spx, recon_obs=None):
    rescaled_returns = torch.empty_like(returns)
    rescaled_returns[1:] = returns[1:] / returns[:-1]
    rescaled_returns[0] = returns[0]
    rescaled_returns = rescaled_returns - 1

    #rescaled_recon_obs = torch.empty_like(recon_obs)
    #rescaled_recon_obs[1:] = recon_obs[1:] / recon_obs[:-1]
    #rescaled_recon_obs[0] = recon_obs[0]
    #rescaled_recon_obs = rescaled_recon_obs - 1
    spx = torch.tensor(spx, device=device)
    spx_returns = spx[1:] / spx[:-1] - 1.
    print(torch.max(spx_returns))
    make_ret_hist(spx_returns.cpu().numpy(), rescaled_returns.cpu().numpy())
    hist_and_true_value(make_skewness(rescaled_returns), make_skewness(spx_returns), "skewness")
    hist_and_true_value(make_kurtosis(rescaled_returns), make_kurtosis(spx_returns), "kurtosis")
    #print(np.mean(make_skewness(rescaled_recon_obs)))
    plt.plot(torch.mean(autocorrelation(spx_returns, dim=0), dim=1).cpu().numpy()[:60])
    #plt.plot(torch.mean(autocorrelation(rescaled_recon_obs, dim=0), dim=1).cpu().numpy()[:60])
    plt.plot(torch.mean(autocorrelation(rescaled_returns, dim=0), dim=1).cpu().numpy()[:60])
    plt.title("Autocorrelation of returns")
    plt.ylabel("Autocorrelation")
    plt.xlabel("Time gap (days)")
    #plt.legend(["SPX", "Reconstructed SPX", "Simulated SPX"])
    plt.legend(["SPX", "Reconstructed SPX", "Simulated SPX"])
    plt.show()
    plt.plot(torch.mean(autocorrelation(spx_returns**2, dim=0), dim=1).cpu().numpy()[:60])
    #plt.plot(torch.mean(autocorrelation(rescaled_recon_obs ** 2, dim=0), dim=1).cpu().numpy()[:60])
    plt.plot(torch.mean(autocorrelation(rescaled_returns ** 2, dim=0), dim=1).cpu().numpy()[:60])
    plt.title("Autocorrelation of square returns")
    plt.ylabel("Autocorrelation")
    plt.xlabel("Time gap (days)")
    #plt.legend(["Full SPX", "Reconstructed SPX", "Simulated SPX"])
    plt.legend(["SPX", "Simulated SPX"])
    plt.show()
    plt.plot(torch.mean(autocorrelation(torch.abs(spx_returns), dim=0), dim=1).cpu().numpy()[:60])
    #plt.plot(torch.mean(autocorrelation(torch.abs(rescaled_recon_obs), dim=0), dim=1).cpu().numpy()[:60])
    plt.plot(torch.mean(autocorrelation(torch.abs(rescaled_returns), dim=0), dim=1).cpu().numpy()[:60])
    plt.title("Autocorrelation of absolute returns")
    plt.ylabel("Autocorrelation")
    plt.xlabel("Time gap (days)")
    plt.legend(["Full SPX", "Simulated SPX"])
    plt.show()


def plot_paths(SSM, time_extent, mean, sd, dataset, name=None, epoch=100):
    if isinstance(SSM, pydpf.FilteringModel):
        paths = simulate_paths(SSM, time_extent)
        paths.apply(lambda observation, **data: observation * sd.cpu() + mean.cpu())
        paths.apply(from_log_ret)
        returns = paths.observation.squeeze()
    else:
        paths = SSM.generate(1000, time_extent)
        returns = paths.detach().cpu() * sd.cpu() + mean.cpu()
        returns = from_log_ret(returns)


    data_path = Path("./experiments/SPX/data/")
    raw_data = np.load(data_path / "raw.npy")
    spx_raw = raw_data[:, 0]
    spx = np.lib.stride_tricks.as_strided(spx_raw, shape=(spx_raw.shape[0] - time_extent + 1, time_extent), strides=(spx_raw.strides[0], spx_raw.strides[0]))
    spx = spx[::10]
    spx = spx[1:] / spx[:-1, 0:1]
    n_years = len(spx_raw) // time_extent
    cut_spx = einops.rearrange(spx_raw[:n_years*time_extent], "(n s) -> s n", s = time_extent, n = n_years)


    #paths.apply(lambda observation, **data: torch.exp(observation))
    #paths2.apply(lambda observation, **data: torch.exp(observation))

    np_returns = returns.cpu().numpy()
    if name is not None:
        np.save(Path(f"./experiments/SPX/results/{name}_{epoch}.npy"), np_returns)
    plt.title(f"Comparison of simulated and real paths {epoch}")
    plt.plot(np_returns.squeeze()[:, :100], alpha=0.1, marker='o', color='blue', linewidth=1, markersize=0.1)
    plt.plot(np.transpose(spx), alpha=0.1, marker='o', color='red', linewidth=1, markersize=0.1)
    legend_lines = [
        mlines.Line2D([], [], color='blue', alpha=1.0, label="Simulated"),
        mlines.Line2D([], [], color='red', alpha=1.0, label="Real"),
    ]
    plt.legend(handles=legend_lines)
    plt.ylabel("Returns")
    plt.xlabel("Time passed (days)")
    plt.ylim(0, min(plt.ylim()[1], 10))
    plt.show()
    plt.title(f"Comparison of simulated and real short paths {epoch}")
    plt.plot(np_returns.squeeze()[:120, :100], alpha=0.1, marker='o', color='blue', linewidth=1, markersize=0.1)
   # plt.plot(recon_obs.cpu()[:, :10].numpy(), alpha=0.1, marker='o', color='green', linewidth=1, markersize=0.1)
    plt.plot(np.transpose(spx)[:120], alpha=0.1, marker='o', color='red', linewidth=1, markersize=0.1)
    legend_lines = [
        mlines.Line2D([], [], color='blue', alpha=1.0, label="Simulated"),
        #mlines.Line2D([], [], color='green', alpha=1.0, label="Reconstructed"),
        mlines.Line2D([], [], color='red', alpha=1.0, label="Real"),
    ]
    plt.legend(handles=legend_lines)
    plt.ylabel("Returns")
    plt.xlabel("Time passed (days)")
    plt.ylim(0, min(plt.ylim()[1], 10))
    plt.show()
    plot_auto_corrs(returns, cut_spx)#, recon_obs)

