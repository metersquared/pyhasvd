# %%

# HANKEL MATRIX GENERATION
import numpy as np

from hasvd.utils.matrix import array_to_hankel, fid_signal_sequence, lrf_sequence

seed = 42  # RNG seed

(total_m, total_n) = (2000, 2000)
r = 400  # Desired rank

matrix_rng = np.random.Generator(np.random.MT19937(seed))


A_array = lrf_sequence(r, total_m + total_n - 1, matrix_rng, dtype=np.float64)

A = array_to_hankel(A_array, (total_m, total_n))

print("Generated random Hankel matrix...")
print("Prescribed rank:", r)
print("Matrix rank sanity check:", np.linalg.matrix_rank(A))
print("Matrix shape:", A.shape)


from hasvd.utils.svd import svd_with_tol, method_of_snapshots
from time import time

# PARAMETERS

tol = 1e-7 * np.linalg.norm(A)  # Prescribe tolerance and control parameter
omega = 0.1

svd_method = svd_with_tol
print("\u0332".join("LAPACK gesdd"))
# SVD of the whole matrix
Ut, Etrue, Vht = svd_method(A, full_matrices=False, truncate_tol=None)
print("Error:", np.linalg.norm(A - Ut @ np.diag(Etrue) @ Vht))
print("Rank of U*S*Vt:", np.linalg.matrix_rank(Ut @ np.diag(Etrue) @ Vht), "\n")

partitions = [2, 4, 5, 8, 10, 16, 20, 25, 40, 50]
omegas = [1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

Es = []
E_omegas = []


from hasvd.utils.trees import (
    tlbd_dist_hasvd_tree,
    tlbd_inc_hasvd_tree,
    assert_shape_consistency,
)

from hasvd.utils.trees import (
    tlbd_hankelarray_ltb_map,
    branch_node_count,
    non_leaf_count,
)
from hasvd.utils.errors import tight_error, naive_error

from hasvd.utils.svd import hasvd, simple_cache_map

for partition in partitions:
    (M, N) = (partition, partition)
    (m, n) = (int(total_m / M), int(total_n / N))

    # TREE CONSTRUCTION

    outer_direction = 0  # Direction of the root-level aggregation
    tree_choice = 0  # Choice of tree: 0 distributed, 1 incremental

    match outer_direction:
        case 0:
            num_outer_slices = N
            num_inner_slices = M
        case 1:
            num_outer_slices = M
            num_inner_slices = N

    trees = [
        tlbd_dist_hasvd_tree(
            num_outer_slices, num_inner_slices, outer_direction, (m, n)
        ),
        tlbd_inc_hasvd_tree(
            num_outer_slices, num_inner_slices, outer_direction, (m, n)
        ),
    ]

    for tree in trees:
        assert_shape_consistency(tree)

    tree = trees[tree_choice]

    btl_map = tlbd_hankelarray_ltb_map(
        A_array, M, N, m, n, outer_direction
    )  # Makes a map to block of a node
    error_choice = 0  # Choice of error_prescription: 0 tight, 1 naive

    nodal_errors = [
        lambda node: tight_error(
            node,
            tol,
            omega,
            branch_node_count(tree),
        ),
        lambda node: naive_error(
            node,
            tol,
            omega,
            non_leaf_count(tree),
        ),
    ]
    nodal_error = nodal_errors[error_choice]

    print("\u0332".join("HASVD"))
    start = time()
    (U, E, Vh, rank_dict), svd_cache = hasvd(
        tree,
        btl_map,
        nodal_error,
        svd_method=svd_method,
        track_ranks=True,
        cache_map=simple_cache_map,
        track_svd=True,
    )
    duration = time() - start
    # print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))
    print("Runtime:", duration, " seconds")
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U @ np.diag(E) @ Vh), "\n")

    Es.append(svd_cache[("r", 2000, 2000)][1])

partition = 10

for omega in omegas:
    (M, N) = (partition, partition)
    (m, n) = (int(total_m / M), int(total_n / N))

    # TREE CONSTRUCTION

    outer_direction = 0  # Direction of the root-level aggregation
    tree_choice = 0  # Choice of tree: 0 distributed, 1 incremental

    match outer_direction:
        case 0:
            num_outer_slices = N
            num_inner_slices = M
        case 1:
            num_outer_slices = M
            num_inner_slices = N

    trees = [
        tlbd_dist_hasvd_tree(
            num_outer_slices, num_inner_slices, outer_direction, (m, n)
        ),
        tlbd_inc_hasvd_tree(
            num_outer_slices, num_inner_slices, outer_direction, (m, n)
        ),
    ]

    for tree in trees:
        assert_shape_consistency(tree)

    tree = trees[tree_choice]

    btl_map = tlbd_hankelarray_ltb_map(
        A_array, M, N, m, n, outer_direction
    )  # Makes a map to block of a node
    error_choice = 0  # Choice of error_prescription: 0 tight, 1 naive

    nodal_errors = [
        lambda node: tight_error(
            node,
            tol,
            omega,
            branch_node_count(tree),
        ),
        lambda node: naive_error(
            node,
            tol,
            omega,
            non_leaf_count(tree),
        ),
    ]
    nodal_error = nodal_errors[error_choice]

    print("\u0332".join("HASVD"))
    start = time()
    (U, E, Vh, rank_dict), svd_cache = hasvd(
        tree,
        btl_map,
        nodal_error,
        svd_method=svd_method,
        track_ranks=True,
        cache_map=simple_cache_map,
        track_svd=True,
    )
    duration = time() - start
    # print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))
    print("Runtime:", duration, " seconds")
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U @ np.diag(E) @ Vh), "\n")

    E_omegas.append(svd_cache[("r", 2000, 2000)][1])


# %%
import matplotlib.pyplot as plt

# === Figure setup for full A4 width ===
width_in = 150 / 25.4  # Full A4 page width in inches
height_in = width_in * 0.45  # Adjust for short horizontal space

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

# Create 1 row, 2 columns
fig, axs = plt.subplots(1, 2, figsize=(width_in, height_in), sharey=True)

# Style setup
markers = ["o", "s", "^", "d", "v", ">", "<", "p"]
linestyles = [":", "--", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), "-", "--", "-."]

# === Left subplot: Partition sweep ===
ax = axs[0]
ax.plot(Etrue, label=r"$SVD_{gesdd}$", linestyle="-", color="black", linewidth=1.0)

for i, partition in enumerate(partitions):
    ax.plot(
        Es[i],
        label=rf"$N={partition}$",
        linestyle=linestyles[i % len(linestyles)],
        marker=markers[i % len(markers)],
        markersize=3,
        linewidth=0.8,
        markevery=50,
    )

ax.set_yscale("log")
ax.set_xlim([0, 2000])
ax.set_ylim([1e-16, 1e-4])
ax.set_xlabel(r"Index $i$")
ax.set_ylabel(r"Singular values $\sigma_i$")
ax.set_title(r"(a) Partition sweep")
ax.legend(ncol=1, loc="upper right", frameon=False)

# === Right subplot: Omega sweep ===
ax = axs[1]
ax.plot(Etrue, label=r"$SVD_{gesdd}$", linestyle="-", color="black", linewidth=1.0)

for i, omega_val in enumerate(omegas):
    ax.plot(
        E_omegas[i],
        label=rf"$\omega={omega_val}$",
        linestyle=linestyles[i % len(linestyles)],
        marker=markers[i % len(markers)],
        markersize=3,
        linewidth=0.8,
        markevery=50,
    )

ax.set_yscale("log")
ax.set_xlim([0, 2000])
ax.set_ylim([1e-16, 1e-4])
ax.set_xlabel(r"Index $i$")
ax.set_title(r"(b) $\omega$ sweep at $N=10$")
ax.legend(ncol=1, loc="upper right", frameon=False)

# === Layout and save ===
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
fig.savefig(
    "hasvd_partition_omega_sweep.pdf",
    format="pdf",
    dpi=600,
    bbox_inches="tight",
    transparent=True,
)

# %%
import numpy as np
import matplotlib.pyplot as plt

# HEAT MAP PLOT

# === Thesis‐style figure settings ===
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)
width_in = 150 / 25.4  # 150 mm
height_in = width_in * 0.45

# === Prepare data matrix for heatmaps ===
max_i = 2000  # truncate/pad all singular‐value vectors to length 2000


def build_heatmap_array(list_of_E):
    """Stack, truncate/pad to shape (len(list_of_E), max_i)."""
    arr = []
    for E in list_of_E:
        v = np.asarray(E)
        if v.size < max_i:
            v = np.pad(v, (0, max_i - v.size), constant_values=np.nan)
        else:
            v = v[:max_i]
        arr.append(v)
    return np.vstack(arr)


heat_part = build_heatmap_array(Es)  # shape (len(partitions), max_i)
heat_omega = build_heatmap_array(E_omegas)  # shape (len(omegas), max_i)

import numpy as np
import matplotlib.pyplot as plt

# … [after computing heat_part and heat_omega as before] …

# 1) Create masked arrays so NaNs aren’t treated as “missing data”
mpart = np.ma.masked_invalid(heat_part)
momega = np.ma.masked_invalid(heat_omega)

# 2) Grab a copy of the viridis colormap and set its “bad” color to the lowest color
cmap = plt.cm.viridis.copy()
cmap.set_bad(cmap(0))  # use the first color in the colormap for NaNs

# 3) Compute shared color limits in log‐space
log_mp = np.log10(mpart)
log_mo = np.log10(momega)
vmin = min(np.nanmin(log_mp), np.nanmin(log_mo))
vmax = max(np.nanmax(log_mp), np.nanmax(log_mo))

# 4) Plot
fig, axs = plt.subplots(1, 2, figsize=(width_in, height_in), constrained_layout=True)
for ax, data, yvals, title in zip(
    axs,
    (mpart, momega),
    (partitions, omegas),
    ("$\omega=0.1$", "$N=10$"),
):
    im = ax.imshow(
        np.log10(data),
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, max_i, yvals[-1], yvals[0]],
    )
    ax.set_title(title)
    ax.set_xlabel("Index $i$")
    ax.set_ylabel(
        "Partition $N$"
        if title.startswith("$\omega=0.1$")
        else "Control parameter $\\omega$"
    )
    # === Add Etrue contours ===
    Etrue_grid = np.tile(np.log10(Etrue[:max_i]), (data.shape[0], 1))

    contour = ax.contour(
        np.arange(max_i),
        yvals,
        Etrue_grid,
        levels=[-15, -5, 0],
        colors="white",
        linewidths=0.4,
        linestyles="dashed",
        alpha=0.7,
    )
    ax.clabel(contour, inline=True, fontsize=7, fmt=r"$10^{%.0f}$")

axs[0].text(
    -0.1, 1.1, "(a)", transform=axs[0].transAxes, fontsize=10, va="top", ha="right"
)
axs[1].text(
    -0.1, 1.1, "(b)", transform=axs[1].transAxes, fontsize=10, va="top", ha="right"
)

# 5) Colorbar
cbar = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
cbar.set_label(r"$\log_{10}(\sigma_i)$")

fig.savefig(
    "hasvd_spectra_heatmaps_masked.pdf",
    dpi=600,
    format="pdf",
    bbox_inches="tight",
    transparent=True,
)

# %%
