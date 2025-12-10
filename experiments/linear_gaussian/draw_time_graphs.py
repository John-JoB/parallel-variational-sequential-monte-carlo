import seaborn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import einops
import torch
from pytorch_forecasting import autocorrelation
from sympy.stats import skewness, kurtosis
import matplotlib.ticker as mticker
import matplotlib as mpl


results_path = Path("experiments/linear_gaussian/results_timing.csv")

def set_icml_style():
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.0,
        rc={
            # Fonts
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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


if __name__ == "__main__":
    set_icml_style()
    fig, ax = plt.subplots(figsize=(4, 3))
    df = pd.read_csv(results_path)
    df.loc[df["Method"] == "dSMC", "Method"] = "d-SMC"
    seaborn.lineplot(data=df[df["Time Extent"] > 30], x="Time Extent", y="Forward", hue="Method", palette="muted", hue_order=["PVMC", "MDPS", "Soft", "Stop-Grad", "Diffusion", "d-SMC", "TFS"])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Average Forward Time (s)")
    ax.set_xlabel("Trajectory Length")

    #plt.grid()
    ax.legend(
        title="Method",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )
    ticks = [50, 100, 200, 300, 500]
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig("experiments/linear_gaussian/forward_time.pdf")
    plt.show()


    fig, ax = plt.subplots(figsize=(4, 3))
    seaborn.lineplot(data=df[(df["Time Extent"] > 30) & (df["Method"] != "TFS") &  (df["Method"] != "dSMC")], x="Time Extent", y="Backward", palette="muted", hue="Method", hue_order=["PVMC", "MDPS", "Soft", "Stop-Grad", "Diffusion"])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Average Backward Time (s)")
    ax.set_xlabel("Trajectory Length")

    # plt.grid()
    ax.legend(
        title="Method",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )
    ticks = [50, 100, 200, 300, 500]
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig("experiments/linear_gaussian/backward_time.pdf")
    plt.show()