# Example of HASVD with Linear trees : Follows similar steps to the ipynb. Check that if unclear.

# %%
import numpy as np
from hasvd.utils.matrix import random_matrix, random_hankel, hankel_to_array
from time import time  # For timing

seed = 42

(M, N) = (10, 10)  # Number of blocks along row and columns
(m, n) = (500, 500)  # Block size
r = 10
tol = 1e-5
omega = 0.1

direction = 1  # Direction of aggregation
matrix_type = 1  # 0 General matrix/ 1 Hankel matrix

if direction == 0:
    partitions = N
    block_shape = (M * m, n)
elif direction == 1:
    partitions = M
    block_shape = (m, N * n)

matrix_rng = np.random.Generator(np.random.MT19937(seed))  # Seed generation

if matrix_type == 0:
    A = random_matrix(M * m, N * n, r, matrix_rng, "qr")
elif matrix_type == 1:
    A = random_hankel(M * m, N * n, matrix_rng, "fid", r)
    A_array = hankel_to_array(A)

print("Generated random general matrix...")
print("Prescribed rank:", r)
print("Matrix rank sanity check:", np.linalg.matrix_rank(A))
print("Matrix shape:", A.shape)
print("Block shape:", "(", m, ",", n, ")")
print("Block partitions:", "(", M, ",", N, ")\n")

# %%
from hasvd.utils.svd import (
    svd_with_tol,
)  # Standard routine for SVD with truncate tolerance

print("\u0332".join("LAPACK gesdd"))
# SVD of the whole matrix
start = time()
U, E, Vh = svd_with_tol(
    A,
    full_matrices=False,
    truncate_tol=tol,
)
duration = time() - start
print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))
print("Runtime:", duration, " seconds")
print("Rank of U*S*Vt:", np.linalg.matrix_rank(U @ np.diag(E) @ Vh), "\n")

# %%
from hasvd.utils.trees import (
    dist_hasvd_tree,
    inc_hasvd_tree,
    assert_shape_consistency,
)

tree_choice = 1  # Choice of tree: 0 distributed, 1 incremental


trees = [
    dist_hasvd_tree(partitions, direction, block_shape),
    inc_hasvd_tree(partitions, direction, block_shape),
]

for tree in trees:
    assert_shape_consistency(tree)

tree = trees[tree_choice]
print(tree)  # Print to check shape of tree if necessary.

# %%
from hasvd.utils.trees import (
    linear_general_btl_map,
    linear_hankelarray_btl_map,
    branch_node_count,
    non_leaf_count,
)
from hasvd.utils.errors import tight_error, naive_error

from hasvd.utils.svd import hasvd

if matrix_type == 0:
    btl_map = linear_general_btl_map(
        A, block_shape[0], block_shape[1], direction
    )  # Makes a map to block of a node
elif matrix_type == 1:
    btl_map = linear_hankelarray_btl_map(
        A_array, partitions, block_shape[0], block_shape[1], direction
    )

error_choice = 0  # Choice of error_prescription: 0 naive, 1 tight

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
# SVD of the whole matrix
start = time()
U, E, Vh, rank_dict = hasvd(tree, btl_map, nodal_error, track_ranks=True)
duration = time() - start
print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))
print("Runtime:", duration, " seconds")
print("Rank of U*S*Vt:", np.linalg.matrix_rank(U @ np.diag(E) @ Vh), "\n")


# %%
from hasvd.utils.trees import plot_rank_graph

plot_rank_graph(tree, rank_dict)
