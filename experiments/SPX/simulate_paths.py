import pydpf
import torch
from matplotlib import pyplot as plt
from pytorch_forecasting import autocorrelation
from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate_paths(SSM, time_extent, vix):
    series_metadata = torch.ones((100, 1), device=device, dtype=torch.float32) * vix
    pydpf.simulate_and_save("./experiments/SPX/data/paths.csv", SSM, time_extent=time_extent, n_trajectories=100, batch_size=128, device=device, series_metadata=series_metadata, bypass_ask=True)
    paths = pydpf.StateSpaceDataset("./experiments/SPX/data/paths.csv")
    return paths

def from_log_ret(observation, **data):
    log_ret = torch.cumsum(observation, dim=0)
    return torch.exp(log_ret)

def make_skewness(returns):
    sample_mean = torch.mean(returns, dim=0)
    third_moment = torch.mean((returns - sample_mean)**3, dim=0)
    second_moment = torch.mean((returns - sample_mean)**2, dim=0)
    return third_moment/torch.pow((second_moment * (returns.size(0) /( returns.size(0) - 1))), 1.5)

def plot_auto_corrs(returns, spx, recon_obs, cut_spx):
    rescaled_returns = torch.empty_like(returns)
    rescaled_returns[1:] = returns[1:] / returns[:-1]
    rescaled_returns[0] = returns[0]
    rescaled_returns = rescaled_returns - 1

    rescaled_recon_obs = torch.empty_like(recon_obs)
    rescaled_recon_obs[1:] = recon_obs[1:] / recon_obs[:-1]
    rescaled_recon_obs[0] = recon_obs[0]
    rescaled_recon_obs = rescaled_recon_obs - 1

    rescaled_cut_spx = torch.empty_like(cut_spx)
    rescaled_cut_spx[1:] = cut_spx[1:] / cut_spx[:-1]
    rescaled_cut_spx[0] = cut_spx[0]
    rescaled_cut_spx = rescaled_cut_spx - 1
    spx_returns = torch.tensor(spx[1:] / spx[:-1] - 1, device=device)
    print(torch.mean(make_skewness(rescaled_returns)))
    print(torch.mean(make_skewness(rescaled_recon_obs)))
    print(torch.mean(make_skewness(rescaled_cut_spx)))
    print(make_skewness(spx_returns))

    plt.plot(torch.mean(autocorrelation(rescaled_returns, dim=0),dim=1).cpu().numpy()[:100])
    plt.plot(torch.mean(autocorrelation(rescaled_recon_obs, dim=0), dim=1).cpu().numpy())
    plt.plot(torch.mean(autocorrelation(rescaled_cut_spx, dim=0), dim=1).cpu().numpy())
    plt.plot(autocorrelation(spx_returns, dim=0).cpu().numpy()[:100])
    plt.show()
    plt.plot(torch.mean(autocorrelation(rescaled_returns**2, dim=0),dim=1).cpu().numpy()[:100])
    plt.plot(torch.mean(autocorrelation(rescaled_recon_obs**2, dim=0), dim=1).cpu().numpy())
    plt.plot(torch.mean(autocorrelation(rescaled_cut_spx, dim=0), dim=1).cpu().numpy())
    plt.plot(autocorrelation(spx_returns**2, dim=0).cpu().numpy()[:100])
    plt.show()
    plt.plot(torch.mean(autocorrelation(torch.abs(rescaled_returns), dim=0),dim=1).cpu().numpy()[:100])
    plt.plot(torch.mean(autocorrelation(torch.abs(rescaled_recon_obs), dim=0), dim=1).cpu().numpy())
    plt.plot(torch.mean(autocorrelation(torch.abs(rescaled_cut_spx), dim=0), dim=1).cpu().numpy())
    plt.plot(autocorrelation(torch.abs(spx_returns), dim=0).cpu().numpy()[:100])
    plt.show()


def plot_paths(SSM, proposal, time_extent, mean, sd, dataset):
    paths = simulate_paths(SSM, time_extent, 0.5)
    paths2 = simulate_paths(SSM, time_extent, 0.)
    with torch.inference_mode():
        encoded_obs, _ = proposal(n_particles=1, observation = dataset.observation, series_metadata = dataset.series_metadata)
        recon_obs = SSM.observation_model.sample(state = encoded_obs.reshape((encoded_obs.size(0) * encoded_obs.size(1), encoded_obs.size(2), encoded_obs.size(3)))).squeeze()
        recon_obs = recon_obs.reshape((encoded_obs.size(0), encoded_obs.size(1)))
        recon_obs = -recon_obs * sd + mean
        recon_obs = from_log_ret(recon_obs)
        cut_spx = dataset.observation.squeeze()
        cut_spx = -cut_spx * sd + mean
        cut_spx = from_log_ret(cut_spx)
        #cut_spx = torch.exp(cut_spx)



    data_path = Path("./experiments/SPX/data/")
    raw_data = np.load(data_path / "raw.npy")
    spx_raw = raw_data[:, 0]
    spx = np.lib.stride_tricks.as_strided(spx_raw, shape=(spx_raw.shape[0] - time_extent + 1, time_extent), strides=(spx_raw.strides[0], spx_raw.strides[0]))
    spx = spx[::10]
    spx = spx[1:] / spx[:-1, 0:1]
    paths.apply(lambda observation, **data: -observation * sd.cpu() + mean.cpu())
    paths2.apply(lambda observation, **data: -observation * sd.cpu() + mean.cpu())
    paths.apply(from_log_ret)
    #paths.apply(lambda observation, **data: torch.exp(observation))
    #paths2.apply(lambda observation, **data: torch.exp(observation))
    paths2.apply(from_log_ret)
    returns = paths.observation.squeeze()
    returns2 = paths2.observation.squeeze()
    np_returns = returns.cpu().numpy()
    np_returns2 = returns2.cpu().numpy()
    plt.plot(np_returns.squeeze(), alpha=0.1, marker='o', color='blue', linewidth=1, markersize=0.1)
    plt.plot(np.transpose(spx), alpha=0.1, marker='o', color='red', linewidth=1, markersize=0.1)
    plt.ylim(0, min(plt.ylim()[1], 10))
    plt.show()
    plt.plot(np_returns.squeeze()[:120], alpha=0.1, marker='o', color='blue', linewidth=1, markersize=0.1)
    plt.plot(recon_obs.cpu()[:, :10].numpy(), alpha=0.1, marker='o', color='green', linewidth=1, markersize=0.1)
    plt.plot(np.transpose(spx)[:120], alpha=0.1, marker='o', color='red', linewidth=1, markersize=0.1)
    plt.ylim(0, min(plt.ylim()[1], 10))
    plt.show()
    plt.plot(np_returns.squeeze(), alpha=0.1, marker='o', color='blue', linewidth=1, markersize=0.1)
    plt.plot(np_returns2.squeeze(), alpha=0.1, marker='o', color='red', linewidth=1, markersize=0.1)
    plt.ylim(0, min(plt.ylim()[1], 10))
    plt.show()
    plot_auto_corrs(returns, spx_raw, recon_obs, cut_spx)

