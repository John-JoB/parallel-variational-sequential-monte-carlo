from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import einops
import torch
from pytorch_forecasting import autocorrelation
import matplotlib as mpl


def set_icml_style():
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.0,
        rc={
            # Fonts
            "font.family": "serif",
            "font.serif": ["Times", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",

            # Axes
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,

            # Ticks
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,

            # Lines
            "lines.linewidth": 1.8,
            "lines.markersize": 4,

            # Legend
            "legend.fontsize": 8,
            "legend.frameon": False,

            # Figure
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",

            # Grid (off by default)
            "axes.grid": False,
        }
    )

    # Matplotlib fine-tuning
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42



result_path = Path("./experiments/SPX/results/")
methods = ["PVMC", "DMM", "TCVAE", "P-VAE", "Soft"]
time_extent = 360
sns.set_theme(style="whitegrid", context="paper")

def apply_over_dict(f, d):
    out = {}
    for k, v in d.items():
        out[k] = f(v)
    return out


def get_autocorr(arr):
    ten = torch.tensor(arr)
    auto_corr = autocorrelation(ten, dim=0).mean(dim=1).cpu().numpy()[1:50]
    return auto_corr



def plot_autocorr(arrs, title):
    arrs = apply_over_dict(get_autocorr, arrs)
    index = np.arange(len(arrs["SPX"]))
    index = np.tile(index, len(arrs))
    methods = list(arrs.keys())
    methods = np.repeat(methods, len(arrs["SPX"]))
    auto_corrs = np.concat(list(arrs.values()), axis=0)
    df = pd.DataFrame({"x": index, "method": methods, "autocorr": auto_corrs})
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    sns.lineplot(data=df[df["method"] == "SPX"], x="x", y="autocorr", color="black", label="SPX", linewidth=2)
    sns.lineplot(data = df[df["method"] != "SPX"], x="x", y="autocorr", hue="method", palette="muted", linewidth=2)
    ax.set_ylabel(f"Autocorrelation")
    ax.set_xlabel("Days")
    legend = ax.get_legend()
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(handles, labels, loc="upper right", frameon=False, bbox_to_anchor=(1.04, 1.03))
    #ax.legend(
    #    title="Method",
     #   bbox_to_anchor=(1.05, 1),
     #   loc="upper left",
     #   borderaxespad=0.0,
      #  frameon=False,
    #)
    plt.tight_layout()
    plt.savefig(result_path / f"figs/{title}.pdf")
    plt.show()

def make_kurtosis(returns):
    sample_mean = np.mean(returns, axis=0)
    fourth_moment = np.mean((returns - sample_mean) ** 4, axis=0)
    second_moment = np.mean((returns - sample_mean) ** 2, axis=0)
    sample_var = (returns.shape[0] / (returns.shape[0] - 1)) * second_moment
    return fourth_moment / (sample_var ** 2)

def make_skewness(returns):
    sample_mean = np.mean(returns, axis=0)
    third_moment = np.mean((returns - sample_mean) ** 3, axis=0)
    second_moment = np.mean((returns - sample_mean) ** 2, axis=0)
    return (third_moment / np.pow((second_moment * (returns.shape[0] / (returns.shape[0] - 1))), 1.5))

def make_format_for_seaborn(points):
    labels = list(points.keys())
    data = list(points.values())
    repeated_labels = np.repeat(labels, [len(element) for element in data])
    cat_data = np.concat(data)
    return pd.DataFrame({"method": repeated_labels, "value": cat_data})

def plot_histogram_skewness(points):
    df = make_format_for_seaborn(points)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.kdeplot(data=df[df["method"] != "SPX"], x="value", hue="method", palette="bright", fill=True, linewidth=2, cut=0, edgecolor="grey", ax=ax, legend=True)
    ax.set_xlim(-1.5,0.7)
    ax.set_xlabel("Skewness")
    ax.set_ylabel("Frequency")
    for i, v in enumerate(points["SPX"]):
        if i == 0:
            line = ax.axvline(v, linestyle="-", linewidth=1.5, color="black")
        else:
            ax.axvline(v, linestyle="-", linewidth=1.5, color="black")

    legend = ax.get_legend()
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(handles, labels, loc="upper right", frameon=False, bbox_to_anchor=(1.03, 1.0))
    #handles.append(line)
    #labels.append("SPX")
    #ax.legend(handles=handles, labels=labels)
    plt.tight_layout()
    plt.savefig(result_path/"figs/histogram_skewness.pdf")
    plt.show()

def plot_histogram_returns(points):
    points = apply_over_dict(lambda x: x.flatten(), points)
    df = make_format_for_seaborn(points)
    df.rename(columns={"value": "Daily Return"}, inplace=True)
    spx = df[df["method"] == "SPX"]
    others = df[df["method"] != "SPX"]
    ref_expanded = (
        spx.assign(_facet_key=1)
        .merge(
            others[["method"]].drop_duplicates().assign(_facet_key=1),
            on="_facet_key"
        )
        .drop(columns="_facet_key")
        .rename(columns={"method_y": "facet"})
        .rename(columns={"method_x": "method"})
    )
    others = others.assign(facet=others["method"])
    spx_only = spx.assign(facet="SPX")
    df = pd.concat([others, ref_expanded, spx_only], ignore_index=True)

    palette = {
        "SPX": "C1"
    }
    order = ["SPX"]
    for x in df["method"].unique():
        if x != "SPX":
            order.append(x)
            palette[x] = "C0"

    fig, ax = plt.subplots(figsize=(4, 3))
    bins = np.linspace(-0.05, 0.05, 50)
    g = sns.FacetGrid(
        df,
        col="facet",
        col_wrap=3,
        sharex=True,
        sharey=True,
        hue="method",
        palette=palette,
        hue_order=order,
    )
    g.set_titles("{col_name}")
    g.map(sns.histplot, "Daily Return", stat="density", element="step", common_norm=False, linewidth=2, bins=bins, alpha=0.6)
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.set_title("Returns")
    ax.set_xlim(-0.05,0.05)
    plt.tight_layout()
    plt.savefig(result_path / "figs/histogram_returns.pdf")
    plt.show()

def plot_histogram_kurtosis(points):
    df = make_format_for_seaborn(points)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.kdeplot(data=df[df["method"] != "SPX"], x="value", hue="method", palette="bright", fill=True, linewidth=2, cut=0, edgecolor="grey", ax=ax)
    ax.set_xlim(2,20)
    ax.set_xlabel("Kurtosis")
    ax.set_ylabel("Frequency")
    handles, labels = ax.get_legend_handles_labels()
    for i, v in enumerate(points["SPX"]):
        if i == 0:
            line = ax.axvline(v, linestyle="-", linewidth=1.5, color="black")
        else:
            ax.axvline(v, linestyle="-", linewidth=1.5, color="black")

    legend = ax.get_legend()
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(handles, labels, loc="upper right", frameon=False, bbox_to_anchor=(1.03, 1.0))
    plt.tight_layout()
    plt.savefig(result_path / "figs/histogram_kurtosis.pdf")
    plt.show()


def make_daily_returns(year_ret):
    day_returns = np.empty_like(year_ret)
    day_returns[1:] = year_ret[1:] / year_ret[:-1]
    day_returns[0] = year_ret[0]
    return day_returns - 1

if __name__ == "__main__":
    set_icml_style()
    data_path = Path("./experiments/SPX/data/")
    raw_data = np.load(data_path / "raw.npy")
    spx_raw = raw_data[:, 0]
    spx_windows = np.lib.stride_tricks.as_strided(spx_raw, shape=(spx_raw.shape[0] - time_extent + 1, time_extent + 1), strides=(spx_raw.strides[0], spx_raw.strides[0]))
    spx_windows = spx_windows[::10]
    spx_windows = spx_windows[1:] / spx_windows[:-1, 0:1]
    spx_windows = spx_windows.transpose()
    spx_returns = spx_raw[1:] / spx_raw[:-1] - 1
    n_years = len(spx_raw) // (time_extent + 1)
    spx_sections = einops.rearrange(spx_returns[:n_years * (time_extent+1)], "(n s) -> s n", s=time_extent+1, n=n_years)

    year_returns = {}
    for method in methods:
        year_returns[method] = np.load(result_path / f"{method}.npy").squeeze()
    year_returns["SPX"] = spx_windows
    daily_returns = apply_over_dict(make_daily_returns, year_returns)
    daily_returns["SPX"] = spx_sections
    square_returns = apply_over_dict(lambda x: x**2, daily_returns)
    abs_returns = apply_over_dict(np.abs, daily_returns)
    plot_autocorr(abs_returns, "Autocorrelation of absolute daily returns")
    plot_autocorr(square_returns, "Autocorrelation of squared daily returns")
    plot_autocorr(daily_returns, "Autocorrelation of daily returns")
    skewness = apply_over_dict(make_skewness, daily_returns)
    plot_histogram_skewness(skewness)
    kurtosis = apply_over_dict(make_kurtosis, daily_returns)
    plot_histogram_kurtosis(kurtosis)
    plot_histogram_returns(daily_returns)

