import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
import numpy as np
import numpy.linalg as la

M = 100
N = 100
m = 100
n = 100

rng = np.random.Generator(np.random.MT19937(42))
set_rank = 20
direction = 0
trials = 10


def bench(tol, omega):

    tree_arr = [
        trees.dist_hasvd_tree(N, direction, (M * m, n)),
        trees.inc_hasvd_tree(N, direction, (M * m, n)),
        trees.tlbd_dist_hasvd_tree(N, M, 0, (m, n)),
        trees.tlbd_inc_hasvd_tree(N, M, 0, (m, n)),
    ]

    nodal_errors = [
        lambda node: errors.tight_error(node, tol, omega, trees.branch_node_count(tree))
        for tree in tree_arr
    ]

    err = np.zeros(len(tree_arr))
    rk = np.zeros(len(tree_arr))
    rank_true = 0
    error = 0
    rank = 0

    for _ in range(trials):

        A_array = matrix.lrf_sequence(set_rank, M * m + N * n - 1, rng)
        A = matrix.array_to_hankel(A_array, (M * m, N * n))
        rank_true += la.matrix_rank(A)

        # Standard
        U, E, Vh = svd.svd_with_tol(A, full_matrices=False, truncate_tol=tol)
        error += la.norm(A - U @ np.diag(E) @ Vh)
        rank += len(E)

        ltb_map = [
            trees.linear_hankelarray_ltb_map(A_array, N, M * m, n, direction),
            trees.linear_hankelarray_ltb_map(A_array, N, M * m, n, direction),
            trees.tlbd_hankelarray_ltb_map(A_array, M, N, m, n, direction),
            trees.tlbd_hankelarray_ltb_map(A_array, M, N, m, n, direction),
        ]

        # HASVD
        for idx, tree in enumerate(tree_arr):
            U, E, Vh = svd.hasvd(
                tree,
                ltb_map[idx],
                nodal_errors[idx],
            )
            err[idx] += la.norm(A - U @ np.diag(E) @ Vh)
            rk[idx] += len(E)

    rank_true /= trials
    error /= trials
    rank /= trials

    for i in range(len(tree_arr)):
        err[i] /= trials
        rk[i] /= trials

    return rank_true, rank, error, rk, err


if __name__ == "__main__":
    eps = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    omega = [0.1, 0.9]
    methods_num = 4

    combinations = [(eps_val, omega_val) for eps_val in eps for omega_val in omega]

    import pandas as pd

    df = pd.DataFrame(
        combinations,
        columns=[
            "eps",
            "omega",
        ],
    )

    df["r_true"] = np.nan
    df["r"] = np.nan
    df["err"] = np.nan

    for i in range(methods_num):
        df["r" + str(i)] = np.nan
        df["err" + str(i)] = np.nan

    for i, (eps_val, omega_val) in enumerate(combinations):
        print(f"Running benchmark {i + 1}/{len(combinations)}")
        (rank_true, rank, error, rk, err) = bench(eps_val, omega_val)

        df.loc[i] = [eps_val, omega_val, rank_true, rank, error, *rk, *err]

        df.to_csv("hankel_error_rank_data_bench.csv", index=False)
