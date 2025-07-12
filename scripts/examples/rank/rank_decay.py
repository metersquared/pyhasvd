# %%
import numpy as np
import matplotlib.pyplot as plt


def print_ul(string: str):
    print("\033[4m" + string + "\033[0m")


# Hankel matrix generation
from hasvd.utils.matrix import (
    random_hankel,
    mersenne_twister,
    lrf_sequence,
    fid_signal_sequence,
    array_to_hankel,
    random_matrix,
)

seq = "lrf"

tot_m = 2000
tot_n = 2000

rank = 1600

rng = mersenne_twister(42)

if seq == "lrf":
    array = lrf_sequence(rank, tot_m + tot_n - 1, rng)
elif seq == "fid":
    array = fid_signal_sequence(rank, tot_m + tot_n - 1, rng)
else:
    ValueError("No valid sequnece given!")

A = array_to_hankel(array, (tot_m, tot_n))

print("Matrix rank:", np.linalg.matrix_rank(A))
print_ul("LAPACK gesdd")
U, S, Vt = np.linalg.svd(A)
normA = np.linalg.norm(A)

fig1 = plt.figure()
plt.plot(array)

fig2 = plt.figure()
plt.semilogy(S)

# %%

# Tree preparation

import hasvd.utils.trees as ts

tree_num = 5

Us = [None] * 5
Ss = [None] * 5
Vts = [None] * 5
dicts = [None] * 5

M = 10
N = 10
direction = 0

m = int(tot_m / M)
n = int(tot_n / N)

if direction == 0:
    lin_part = N
    lin_shape = (tot_m, n)

    out_part = N
    in_part = M

elif direction == 1:
    lin_part = M
    lin_shape = (m, tot_n)

    out_part = M
    in_part = N

tree_name = [
    "Dist. tree",
    "Inc. tree",
    "Dist. TLB tree",
    "Inc. TLB tree",
    "Alt. Inc. tree",
]

trees = [
    ts.dist_hasvd_tree(lin_part, direction, lin_shape),
    ts.inc_hasvd_tree(lin_part, direction, lin_shape),
    ts.tlbd_dist_hasvd_tree(out_part, in_part, direction, (m, n)),
    ts.tlbd_inc_hasvd_tree(out_part, in_part, direction, (m, n)),
    ts.regular_alt_inc_tree(m, tot_m, direction),
]

for tree in trees:
    tree.validity_check()

maps = [
    ts.linear_general_ltb_map(A, *lin_shape, direction),
    ts.linear_general_ltb_map(A, *lin_shape, direction),
    ts.tlbd_general_ltb_map(A, M, N, m, n, direction),
    ts.tlbd_general_ltb_map(A, M, N, m, n, direction),
    ts.regular_alt_inc_general_ltb_map(A, m, direction),
]


# %%
from hasvd.utils.errors import tight_error
from hasvd.utils.trees import branch_node_count
from hasvd.utils.svd import hasvd, svd_with_tol, method_of_snapshots

tol = 1e-1 * normA
print("Prescription error:", tol)
omega = 0.1

method = svd_with_tol

for idx, tree in enumerate(trees):

    print_ul(tree_name[idx])
    nodal_error = lambda node: tight_error(node, tol, omega, branch_node_count(tree))
    Us[idx], Ss[idx], Vts[idx], dicts[idx] = hasvd(
        tree, maps[idx], nodal_error, svd_method=method, track_ranks=True
    )
    Aprime = Us[idx] @ np.diag(Ss[idx]) @ Vts[idx]
    print("Error:", np.linalg.norm(A - Aprime))
    print("Rank:", np.linalg.matrix_rank(Aprime))

# %%
from hasvd.utils.trees import plot_rank_graph

# Print rank graph
"""
for idx, tree in enumerate(trees):

    print_ul(tree_name[idx])
    plot_rank_graph(tree, dicts[idx], bound=(0, 800))
"""
# %%

from functools import partial


def generate_leaf_rank_contour_matrix(rank_map, map_generator, shape):
    """
    Build a contour matrix of leaf‐node ranks.

    Parameters
    ----------
    rank_map : dict
        Maps each hasvd_Node → its integer rank.
    map_generator : Callable
        A function taking the full matrix `contour` and returning
        another function `leaf_map(node)` that returns the block-array
        view corresponding to `node`.
    shape : (int, int)
        Shape of the full matrix, e.g. A.shape.

    Returns
    -------
    contour : np.ndarray
        A 2D array where each leaf‐block is filled with its tracked rank.
    """
    contour = np.zeros(shape, dtype=int)

    # map_generator(contour) -> leaf_map(node) → view of contour[...]
    leaf_map = map_generator(contour)

    for node, rank in rank_map.items():
        if not node.is_leaf:
            continue
        block = leaf_map(node)
        # fill the entire block with the node’s rank
        block[:] = rank

    return contour


map_generators = [
    partial(
        ts.linear_general_ltb_map,  # we’ll pass `contour` later
        m=lin_shape[0],
        n=lin_shape[1],
        direction=direction,
    ),
    partial(
        ts.linear_general_ltb_map,  # we’ll pass `contour` later
        m=lin_shape[0],
        n=lin_shape[1],
        direction=direction,
    ),
    partial(
        ts.tlbd_general_ltb_map,
        M=M,
        N=N,  # we’ll pass `contour` later
        m=m,
        n=n,
        outer_direction=direction,
    ),
    partial(
        ts.tlbd_general_ltb_map,
        M=M,
        N=N,  # we’ll pass `contour` later
        m=m,
        n=n,
        outer_direction=direction,
    ),
    partial(
        ts.regular_alt_inc_general_ltb_map,
        dblock_length=m,
        outer_direction=direction,
    ),
]

contours = [None] * tree_num

for idx, tree in enumerate(trees):
    contours[idx] = generate_leaf_rank_contour_matrix(
        dicts[idx], map_generators[idx], (tot_m, tot_n)
    )
    print_ul(tree_name[idx])
    fig = plt.figure()
    plt.imshow(
        contours[idx],
    )
    plt.colorbar()

# %%
