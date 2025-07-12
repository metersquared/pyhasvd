import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hasvd.utils.matrix import random_hankel
from hasvd.utils.svd import svd_with_tol

# --- Config ---
rng = np.random.default_rng(42)
rank = 50
tol = 1e-14
trials = 10
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

# === Thesis-style figure size (in inches) ===
width_in = 150 / 25.4  # 150 mm
height_in = (width_in) / 2 * 0.75 + 0.5
figsize = (width_in, height_in)

# === Matplotlib configuration ===
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
    }
)


# --- Plot 1: Singular Values of SSA-LRF ---
def plot_singular_values_ssa_lrf():
    ns = [100, 300, 500, 700, 900]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, n in enumerate(ns):
        A = random_hankel(n, n, rng, "lrf", rank)
        _, S, _ = np.linalg.svd(A)
        ax.plot(S, label=rf"$n={n}$", linestyle=linestyles[i])
    ax.set_yscale("log")
    ax.set_xlim([0, 80])
    ax.set_ylim([1e-16, 10])
    ax.set_xlabel(r"Index $i$")
    ax.set_ylabel(r"Singular values $\sigma_i$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("singular_values_ssa_lrf.pdf")
    plt.close(fig)


# --- Plot 2: Mean Computational Rank 2D (Contour) ---
def plot_mean_rank_2d():
    ms, ns_grid = np.meshgrid(np.arange(100, 901, 100), np.arange(100, 901, 100))
    ave_ranks = np.zeros_like(ms, dtype=float)
    for i in range(ms.shape[0]):
        for j in range(ms.shape[1]):
            m, n = ms[i, j], ns_grid[i, j]
            ave_ranks[i, j] = np.mean(
                [
                    len(
                        svd_with_tol(
                            random_hankel(m, n, rng, "lrf", rank), truncate_tol=tol
                        )[1]
                    )
                    for _ in range(trials)
                ]
            )
    fig, ax = plt.subplots(figsize=(6, 4))
    contour = ax.contourf(ms, ns_grid, ave_ranks, levels=30, cmap="viridis")
    fig.colorbar(contour, label=r"Mean computational rank $\tilde{r}$")
    ax.set_xlabel(r"Row size $m$")
    ax.set_ylabel(r"Column size $n$")
    fig.tight_layout()
    fig.savefig("mean_rank_2d.pdf")
    plt.close(fig)


# --- Plot 3: 3D Surface of Rank Error ---
def plot_mean_rank_3d():
    ms, ns_grid = np.meshgrid(np.arange(100, 901, 100), np.arange(100, 901, 100))
    ave_ranks = np.zeros_like(ms, dtype=float)
    for i in range(ms.shape[0]):
        for j in range(ms.shape[1]):
            m, n = ms[i, j], ns_grid[i, j]
            ave_ranks[i, j] = np.mean(
                [
                    len(
                        svd_with_tol(
                            random_hankel(m, n, rng, "lrf", rank), truncate_tol=tol
                        )[1]
                    )
                    for _ in range(trials)
                ]
            )
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(ms, ns_grid, ave_ranks, cmap="viridis", edgecolor="b", alpha=0.7)
    error = np.abs(ave_ranks - rank)
    mask = error > 10
    ax.scatter(
        ms[mask],
        ns_grid[mask],
        ave_ranks[mask],
        color="red",
        s=40,
        label=r"$|\tilde{r} - r| > 10$",
    )
    ax.set_xlabel(r"Row size $m$")
    ax.set_ylabel(r"Column size $n$")
    ax.set_zlabel(r"Mean rank $\tilde{r}$")
    ax.set_xlim([900, 100])
    ax.set_ylim([100, 900])
    ax.legend()
    fig.tight_layout()
    fig.savefig("mean_rank_3d.pdf")
    plt.close(fig)


# --- Plot 4: FID vs LRF Singular Values ---
def plot_fid_vs_lrf_singular_values():
    ns = [100, 500, 1000, 5000]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, n in enumerate(ns):
        A = random_hankel(n, n, rng, "fid", rank)
        _, S, _ = np.linalg.svd(A)
        marker = "x" if n == 5000 else None
        ax.plot(S, label=rf"$n_{{FID}}={n}$", linestyle=linestyles[i], marker=marker)
    A = random_hankel(5000, 5000, rng, "lrf", rank)
    _, S, _ = np.linalg.svd(A)
    ax.plot(S, marker="+", label=r"$n_{\text{LRF}}=5000$", linestyle=linestyles[4])
    ax.set_yscale("log")
    ax.set_xlim([0, 80])
    ax.set_ylim([1e-16, 10])
    ax.set_xlabel(r"Index $i$")
    ax.set_ylabel(r"Singular values $\sigma_i$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fid_vs_lrf_singular_values.pdf")
    plt.close(fig)


# --- Run All ---
plot_singular_values_ssa_lrf()
plot_mean_rank_2d()
plot_mean_rank_3d()
plot_fid_vs_lrf_singular_values()

print("✅ All thesis-style PDF plots saved.")
