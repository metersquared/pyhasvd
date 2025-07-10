# %%
from hasvd.utils.matrix import random_hankel, mersenne_twister

import numpy as np

rng = mersenne_twister(42)


# Reference
from hasvd.utils.svd import svd_with_tol

tol = 1e-6

A = random_hankel(5000, 5000, rng, "lrf", 50)

# Alternating incremental tree
from hasvd.utils.trees import (
    regular_alt_inc_tree,
    regular_alt_inc_general_ltb_map,
    tlbd_dist_hasvd_tree,
    tlbd_general_ltb_map,
)

print(np.linalg.matrix_rank(A))

d_lengths = [100, 900, 1000, 2000, 3000, 3000]  # Prepare diagonal sizes
outer_dir = 1

tree = regular_alt_inc_tree(500, 5000, outer_dir, "hankel")

ltb_map = regular_alt_inc_general_ltb_map(A, 500, outer_dir)

# Ascribe nodal error and omega
from hasvd.utils.errors import tight_error
from hasvd.utils.trees import branch_node_count

omega = 0.1

nodal_error = lambda node: tight_error(node, tol, omega, branch_node_count(tree))

# HASVD Alt inc
from hasvd.utils.svd import hasvd, symmetric_cache_map, simple_cache_map
from time import perf_counter

print("HASVD Alternating Incremental")
start = perf_counter()
Uh, Sh, Vth = hasvd(tree, ltb_map, nodal_error, cache_map=symmetric_cache_map)
duration = perf_counter() - start
print("Error:", np.linalg.norm(A - Uh @ np.diag(Sh) @ Vth))
print("Runtime (pruned):", duration)

# HASVD Dist

tree = tlbd_dist_hasvd_tree(10, 10, outer_dir, (500, 500), recursion="hankel")

ltb_map = tlbd_general_ltb_map(A, 10, 10, 500, 500, outer_dir)

nodal_error = lambda node: tight_error(node, tol, omega, branch_node_count(tree))

print("HASVD Two-Level Distributed")
start = perf_counter()
Uh, Sh, Vth = hasvd(tree, ltb_map, nodal_error, cache_map=simple_cache_map)
duration = perf_counter() - start
print("Error:", np.linalg.norm(A - Uh @ np.diag(Sh) @ Vth))
print("Runtime (pruned):", duration)

# %%
