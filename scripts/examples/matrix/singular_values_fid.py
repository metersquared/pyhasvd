# %%
import numpy as np
import matplotlib.pyplot as plt
from hasvd.utils.matrix import random_hankel
from hasvd.utils.svd import svd_with_tol

# === Larger figure for visibility ===
width_in = 150 / 25.4
height_in = (width_in / 2) * 0.75  # Adjusted for good aspect
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

fig, ax = plt.subplots(figsize=(width_in, height_in))

ns = [100, 500, 1000, 5000]
linestyles = ["-", "--", "-.", ":"]

for i, n in enumerate(ns):
    A = random_hankel(n, n, rng, "fid", rank)
    _, S, _ = np.linalg.svd(A)
    marker = "x" if n == 5000 else None
    ax.plot(
        S, label=rf"$n_{{\mathrm{{FID}}}}={n}$", marker=marker, linestyle=linestyles[i]
    )

n = 5000
A = random_hankel(n, n, rng, "lrf", rank)
_, S, _ = np.linalg.svd(A)
ax.plot(S, marker="+", label=r"$n_{\mathrm{LRF}}=5000$", linestyle=(0, (5, 2)))

ax.set_yscale("log")
ax.set_xlim([0, 50])
ax.set_ylim([1e-16, 10])
ax.set_ylabel("Singular values $\\sigma_i$")
ax.set_xlabel("Index $i$")
ax.legend(fontsize=9)

fig.savefig(
    "fid_vs_lrf_singvals.pdf",
    dpi=600,
    format="pdf",
    transparent=True,
    bbox_inches="tight",
)

# %%
