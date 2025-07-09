# %%
from hasvd.utils.matrix import random_matrix, mersenne_twister

import numpy as np

rng = mersenne_twister(42)


# Reference
from hasvd.utils.svd import svd_with_tol

tol = 1e-3

A = random_matrix(100, 100, 3, rng)

U, S, Vt = svd_with_tol(A, truncate_tol=tol)

# Alternating incremental tree
from hasvd.utils.trees import alt_inc_tree, alt_inc_general_ltb_map

d_lengths = [20, 50, 30]  # Prepare diagonal sizes
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

print(A)


for node in tree.traverse():
    if node.is_leaf:
        print("Node tag:", node.tag)
        print(ltb_map(node))

Uh, Sh, Vth = hasvd(tree, ltb_map, nodal_error)

print(np.linalg.norm(A - Uh @ np.diag(Sh) @ Vth))


def find_geometric_sequences(M, min_a=10):
    """
    Find all strictly increasing geometric sequences of positive integers such that:
    - Common ratio is an integer > 1
    - Sum of the sequence equals M
    - First term a > min_a

    Returns:
        List of sequences sorted by (length, first_term)
    """
    solutions = []

    for r in range(2, M + 1):
        for n in range(2, 64):
            r_power_n = r**n
            denom = r_power_n - 1
            numer = M * (r - 1)

            if denom > numer:  # Early exit: 'a' would be zero or negative
                break

            if numer % denom != 0:
                continue

            a = numer // denom
            if a <= min_a:
                continue

            sequence = [a * r**i for i in range(n)]
            solutions.append(sequence)

    # Sort: first by length, then by first term
    solutions.sort(key=lambda seq: (len(seq), seq[0]))
    return solutions


b = find_geometric_sequences(1784)
print(b)
# %%
