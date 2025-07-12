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

# PARAMETERS

tol = 1e-7 * np.linalg.norm(A)  # Prescribe tolerance and control parameter
omega = 0.1

svd_method = svd_with_tol
print("\u0332".join("LAPACK gesdd"))
# SVD of the whole matrix
Ut, Etrue, Vht = svd_method(A, full_matrices=False, truncate_tol=None)
print("Error:", np.linalg.norm(A - Ut @ np.diag(Etrue) @ Vht))
print("Rank of U*S*Vt:", np.linalg.matrix_rank(Ut @ np.diag(Etrue) @ Vht), "\n")

partitions = [2, 4, 5, 8, 10, 20]

Es = []

from hasvd.utils.svd import svd_with_tol, method_of_snapshots
from time import time

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
    print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))
    print("Runtime:", duration, " seconds")
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U @ np.diag(E) @ Vh), "\n")

    Es.append(svd_cache[("r", 2000, 2000)][1])

# %%
import matplotlib.pyplot as plt
import scipy as sc

fig = plt.figure()
plt.plot(Etrue, label="$SVD_{gesdd}$")
for 
plt.plot(
    Es[0],
    label="$\sigma$ of $\\rho$ node, before truncation",
    linestyle=":",
)

plt.yscale("log")
plt.ylabel("$\\sigma_i$")
plt.xlabel("Index $i$")
plt.title("Tolerance $\\epsilon*$= " + str(tol))

plt.legend()

plt.xlim([0, 1000])


# %%
