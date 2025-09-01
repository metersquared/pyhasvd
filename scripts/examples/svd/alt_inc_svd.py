# %%
from hasvd.utils.matrix import random_hankel, mersenne_twister

import numpy as np

rng = mersenne_twister(42)


# Reference
from hasvd.utils.svd import svd_with_tol

tol = 1e-5

A = random_hankel(2000, 2000, rng, "lrf", 400)

print(np.linalg.matrix_rank(A))

U, S, Vt = svd_with_tol(A, truncate_tol=tol)

# Alternating incremental tree
from hasvd.utils.trees import alt_inc_tree, alt_inc_general_ltb_map

d_lengths = [200] * 10  # Prepare diagonal sizes
outer_dir = 0

tree = alt_inc_tree(d_lengths, outer_dir)

ltb_map = alt_inc_general_ltb_map(A, d_lengths, outer_dir)

# Ascribe nodal error and omega
from hasvd.utils.errors import tight_error
from hasvd.utils.trees import branch_node_count

omega = 0.1

nodal_error = lambda node: tight_error(node, tol, omega, branch_node_count(tree))

# HASVD
from hasvd.utils.svd import hasvd

Uh, Sh, Vth = hasvd(tree, ltb_map, nodal_error)

print(np.linalg.norm(A - Uh @ np.diag(Sh) @ Vth))

import matplotlib.pyplot as plt

plt.semilogy(Sh)
plt.semilogy(S)

# %%
