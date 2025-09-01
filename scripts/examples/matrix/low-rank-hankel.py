# %%
from hasvd.utils.matrix import random_hankel
from hasvd.utils.svd import method_of_snapshots
import numpy as np
import matplotlib.pyplot as plt

# %%
# This script shows how singular values of SSA-LRF explodes with high matrix sizes

rng = np.random.Generator(np.random.MT19937(42))
ns = [100, 300, 500, 700, 900]

# 6 distinct linestyles
linestyles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 2)),
]  # last two are custom dash patterns

for i, n in enumerate(ns):
    A = random_hankel(n, n, rng, "lrf", 50)
    _, S, _ = np.linalg.svd(A)
    plt.plot(
        S,
        label="$n=" + str(n) + "$",
        # linestyle=linestyles[i],
    )

plt.yscale("log")
plt.xlim([0, 80])
plt.ylim([1e-16, 10])
plt.ylabel("Singular values $\sigma_i$")
plt.xlabel("Index $i$")
plt.legend()

plt.savefig("sing-val-ssalrf.png", format="png", dpi=600, transparent=True)
# %%
ms = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
ns = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
ms, ns = np.meshgrid(ms, ns)
rank = 50
trials = 10


def mean_comp_rank(m, n, trials):
    ave_rank = 0

    for _ in range(trials):
        A = random_hankel(m, n, rng, "lrf", rank)
        _, S, _ = svd_with_tol(A, truncate_tol=1e-14)
        ave_rank = ave_rank + len(S)

    return ave_rank / trials


ave_ranks = np.zeros_like(ms, dtype=float)

for i in range(ms.shape[0]):
    for j in range(ms.shape[1]):
        m = ms[i, j]
        n = ns[i, j]
        ave_ranks[i, j] = mean_comp_rank(m, n, trials)

plt.plot(ms, ns, ave_ranks, marker="o", label="$n=" + str(n) + "$", color="k")

plt.ylabel("Mean computational rank $\\tilde r$")
plt.xlabel("Rank-to-size ratio $\\frac{{r}}{{mn}}$")
# %%
# Prepare 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Compute rank-to-size ratio
rank_ratio = rank / (ms * ns)


# Plot
ax.plot_surface(ms, ns, ave_ranks, cmap="viridis", edgecolor="b", alpha=0)

# Threshold for significant deviation
threshold = 10
error = np.abs(ave_ranks - rank)

# Find indices where error > threshold
mask = error > threshold
x_bad = ms[mask]
y_bad = ns[mask]
z_bad = ave_ranks[mask]

# Plot these as red scatter points
ax.scatter(x_bad, y_bad, z_bad, color="red", s=50, label="$|\\tilde r -  r|$ > 10")
ax.legend()

ax.set_xlabel("Row size $m$")
ax.set_ylabel("Column size $n$")
ax.set_zlabel("Mean Computational Rank $\\tilde{r}$")
ax.set_xlim([900, 100])
ax.set_ylim([100, 900])

plt.savefig("sing-val-dist-ssalrf.png", format="png", dpi=600, transparent=True)
# %%

rng = np.random.Generator(np.random.MT19937(42))

rank = 50
tol = 1e-14

ns = [100, 500, 1000, 5000]

# 6 distinct linestyles
linestyles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 2)),
]  # last two are custom dash patterns

for i, n in enumerate(ns):
    A = random_hankel(n, n, rng, "fid", rank)
    _, S, _ = np.linalg.svd(A)
    if i == 3:
        marker = "x"
    else:
        marker = None
    plt.plot(
        S,
        label="$n_{{FID}}=" + str(n) + "$",
        marker=marker,
        linestyle=linestyles[i],
    )

n = 5000

A = random_hankel(n, n, rng, "lrf", rank)
_, S, _ = np.linalg.svd(A)
plt.plot(
    S,
    marker="+",
    label="$n_{{LRF}}=" + str(n) + "$",
    linestyle=linestyles[4],
)

plt.yscale("log")
plt.xlim([0, 80])
plt.ylim([1e-16, 1e1])
plt.ylabel("Singular values $\sigma_i$")
plt.xlabel("Index $i$")
plt.legend()
plt.savefig("sing-val-dist-fid.png", format="png", dpi=600, transparent=True)
# %%
