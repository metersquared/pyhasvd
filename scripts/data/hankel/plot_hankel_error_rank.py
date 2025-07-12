# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csvs = [
    "hankel_error_rank_data_bench_fid_seed42.csv",
]
seed = ["FID, seed = 42"]

for idx, file in enumerate(csvs):
    # Import the data
    df = pd.read_csv(file)

    error_df = df[["omega", "eps", "err", "err0", "err1", "err2", "err3", "err4"]]
    error_df["eps_self"] = error_df["eps"]

    # Define new labels for the plot
    label_map = {
        "err": "$SVD_{{gesdd}}$",
        "err0": "$HASVD_{{dist}}$",
        "err1": "$HASVD_{{inc}}$",
        "err2": "$HASVD_{{TLBD}}$",
        "err3": "$HASVD_{{TLBI}}$",
        "err4": "$HASVD_{{TLAI}}$",
        "eps_self": "Reference",
    }

    rank_df = df[["omega", "eps", "r", "r0", "r1", "r2", "r3", "r4", "r_true"]]

    # Define new labels for the plot
    label_map_2 = {
        "r": "$SVD_{{gesdd}}$",
        "r0": "$HASVD_{{dist}}$",
        "r1": "$HASVD_{{inc}}$",
        "r2": "$HASVD_{{TLBD}}$",
        "r3": "$HASVD_{{TLBI}}$",
        "r4": "$HASVD_{{TLAI}}$",
        "r_true": "Reference",
    }
    import matplotlib.pyplot as plt
    import numpy as np

    # Figure size in inches
    width_in = 150 / 25.4  # 150 mm
    height_in = width_in * 0.75 + 0.9  # 4:3 ratio

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(width_in, height_in))
    (ax1, ax2), (ax3, ax4) = axs

    # Plot styles
    markers = ["o", "s", "d", "s", "d", "p", "*"]
    colors = ["k", "g", "r", "c", "m", "b", "k"]
    linestyles = ["-", ":", ":", "-.", "-.", ":", "--"]
    xlim = [1, 1e-15]
    xlabel = r"Prescribed relative error $\epsilon^*$"
    ylabel_error = "$\langle\|A-U\Sigma V^T\|_F/\|A\|_F\\rangle$"
    ylabel_rank = "$\langle\\mathrm{rank}(U\Sigma V^T)\\rangle$"

    # Plot error (top row)
    for ax, omega in [(ax1, 0.1), (ax2, 0.9)]:
        for i, col in enumerate(
            ["err", "err0", "err1", "err2", "err3", "err4", "eps_self"]
        ):
            marker = None if col == "eps_self" else markers[i % len(markers)]
            error_df[error_df["omega"] == omega].plot(
                x="eps",
                y=col,
                ax=ax,
                label=label_map[col],
                color=colors[i],
                linestyle=linestyles[i],
                marker=marker,
            )
        ax.set_title(rf"$\omega={omega}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_error)
        ax.get_legend().remove()

    # Plot rank (bottom row)
    for ax, omega in [(ax3, 0.1), (ax4, 0.9)]:
        for i, col in enumerate(["r", "r0", "r1", "r2", "r3", "r4", "r_true"]):
            marker = None if col == "r_true" else markers[i % len(markers)]
            rank_df[rank_df["omega"] == omega].plot(
                x="eps",
                y=col,
                ax=ax,
                label=label_map_2[col],
                color=colors[i],
                linestyle=linestyles[i],
                marker=marker,
            )
        ax.set_title(rf"$\omega={omega}$")
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_rank)
        ax.get_legend().remove()

    # Shared legend above all plots
    handles, labels = ax1.get_legend_handles_labels()

    ax1.text(
        -0.1, 1.1, "(a)", transform=ax1.transAxes, fontsize=10, va="top", ha="right"
    )
    ax2.text(
        -0.1, 1.1, "(b)", transform=ax2.transAxes, fontsize=10, va="top", ha="right"
    )
    ax3.text(
        -0.1, 1.1, "(c)", transform=ax3.transAxes, fontsize=10, va="top", ha="right"
    )
    ax4.text(
        -0.1, 1.1, "(d)", transform=ax4.transAxes, fontsize=10, va="top", ha="right"
    )
    fig.suptitle(seed[idx], y=0.1)

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.2)  # Leave room for legend
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))

    # Save figure
    newName = file.replace(".csv", ".pdf")
    fig.savefig(newName, format="pdf", dpi=300, transparent=True)

# fig.savefig("hankel-error-rank-fid.pgf")  # Optional

# %%
