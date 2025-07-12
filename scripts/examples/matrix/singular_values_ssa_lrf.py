# %%
import numpy as np
import matplotlib.pyplot as plt
from hasvd.utils.matrix import random_hankel
from hasvd.utils.svd import svd_with_tol

# === Figure size for full A4 width ===
width_in = 150 / 25.4  # A4 width in inches
height_in = (width_in / 2) * 0.75 + 0.2  # Adjusted for good aspect
# === Use LaTeX for all text ===
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

rng = np.random.default_rng(42)
rank = 50


# --- Subplot 2: Mean Computational Rank ---
ms = np.arange(100, 901, 100)
ns = np.arange(100, 901, 100)
ms_grid, ns_grid = np.meshgrid(ms, ns)


def mean_comp_rank(m, n, trials=10):
    total = 0
    for _ in range(trials):
        A = random_hankel(m, n, rng, "lrf", rank)
        total += np.linalg.matrix_rank(A)
    return total / trials


ave_ranks = np.zeros_like(ms_grid, dtype=float)
for i in range(ms_grid.shape[0]):
    for j in range(ms_grid.shape[1]):
        m, n = ms_grid[i, j], ns_grid[i, j]
        ave_ranks[i, j] = mean_comp_rank(m, n)
# %%

fig, axs = plt.subplots(1, 2, figsize=(width_in, height_in), constrained_layout=True)

# --- Subplot 1: Singular Values ---
ns = [100, 300, 500, 700, 900]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

for i, n in enumerate(ns):
    A = random_hankel(n, n, rng, "lrf", rank)
    _, S, _ = np.linalg.svd(A)
    axs[0].plot(S, label=rf"$n={n}$", linestyle=linestyles[i])

axs[0].set_yscale("log")
axs[0].set_xlim([0, 80])
axs[0].set_ylim([1e-16, 10])
axs[0].set_ylabel("Singular values $\\sigma_i$")
axs[0].set_xlabel("Index $i$")
axs[0].legend(loc="upper right", fontsize=8)

cs = axs[1].contourf(ms_grid, ns_grid, ave_ranks, levels=20, cmap="viridis")
fig.colorbar(cs, ax=axs[1], label="Mean computational rank $\\tilde{r}$")
axs[1].set_xlabel("Row size $m$")
axs[1].set_ylabel("Column size $n$")

axs[0].text(
    -0.15, 1.05, "(a)", transform=axs[0].transAxes, fontsize=10, va="top", ha="left"
)
axs[1].text(
    -0.15, 1.05, "(b)", transform=axs[1].transAxes, fontsize=10, va="top", ha="left"
)

# --- Save ---
fig.savefig("ssa_lrf_combined_wide.pdf", format="pdf", dpi=600, transparent=True)

# %%
