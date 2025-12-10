from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

prefix = "time"

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


def make_format_for_seaborn(labels, data):
    repeated_labels = np.repeat(labels, [len(element) for element in data])
    cat_data = np.concat(data)
    return pd.DataFrame({"label": repeated_labels, "value": cat_data})

if __name__ == "__main__":
    set_icml_style()
    raw_results = pd.read_csv("./experiments/lotka_volterra/results/raw_results.csv", index_col=0)
    res = []
    experiments = []
    for experiment in raw_results.index:
        if experiment.startswith(prefix):
            experiments.append(experiment[(len(prefix) + 1):])
            if experiments[-1] == "DSSM":
                experiments[-1] = "P-VAE"
            res.append(raw_results.loc[experiment].to_numpy())
            res[-1] = res[-1][~np.isnan(res[-1])]
    plot_df = make_format_for_seaborn(experiments, res)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))

    sns.boxplot(data=plot_df, x="label", y="value", whis=(0, 100), showfliers=False, palette="vlag", hue="label")

    sns.stripplot(data = plot_df, x="label", y="value", color="black", marker="x", size=4, linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f'./experiments/lotka_volterra/results/{prefix}_boxplot.pdf', bbox_inches='tight')
    plt.show()