import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
import numpy as np
import numpy.linalg as la

M = 20
N = 20
m = 100
n = 100

seed=42
rng = np.random.Generator(np.random.MT19937(seed))
set_rank = 400
direction = 0
trials = 1

A_array = matrix.fid_signal_sequence(set_rank, M * m + N * n - 1, rng)
A = matrix.array_to_hankel(A_array, (M * m, N * n))

normA=np.linalg.norm(A)

def bench(tol, omega):

    tree_arr = [
        trees.dist_hasvd_tree(N, direction, (M * m, n)),
        trees.inc_hasvd_tree(N, direction, (M * m, n)),
        trees.tlbd_dist_hasvd_tree(N, M, 0, (m, n)),
        trees.tlbd_inc_hasvd_tree(N, M, 0, (m, n)),
        trees.regular_alt_inc_tree(m,M*m,direction)
    ]

    nodal_errors = []
    for tree in tree_arr:
        count = trees.branch_node_count(tree)
        nodal_errors.append(
            lambda node, count=count: errors.tight_error(node, tol, omega, count)
        )

    err = np.zeros(len(tree_arr))
    rk = np.zeros(len(tree_arr))
    error = 0
    rank = 0
    
    rank_true = la.matrix_rank(A)

    for _ in range(trials):

        # Standard
        U, E, Vh = svd.svd_with_tol(A, full_matrices=False, truncate_tol=tol)
        error += la.norm(A - U @ np.diag(E) @ Vh)
        rank += la.matrix_rank(U @ np.diag(E) @ Vh)

        ltb_map = [
            trees.linear_hankelarray_ltb_map(A_array, N, M * m, n, direction),
            trees.linear_hankelarray_ltb_map(A_array, N, M * m, n, direction),
            trees.tlbd_hankelarray_ltb_map(A_array, M, N, m, n, direction),
            trees.tlbd_hankelarray_ltb_map(A_array, M, N, m, n, direction),
            trees.regular_alt_inc_general_ltb_map(A,m,direction )
        ]

        # HASVD
        for idx, tree in enumerate(tree_arr):
            U, E, Vh = svd.hasvd(
                tree,
                ltb_map[idx],
                nodal_errors[idx],
            )
            err[idx] += la.norm(A - U @ np.diag(E) @ Vh)
            rk[idx] += la.matrix_rank(U @ np.diag(E) @ Vh)

    rank_true /= trials
    error /= trials
    rank /= trials

    for i in range(len(tree_arr)):
        err[i] /= trials
        rk[i] /= trials

    return rank_true, rank, error, rk, err


if __name__ == "__main__":
    eps = [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15]
    omega = [0.1,0.5, 0.9]
    methods_num = 5

    combinations = [(eps_val, omega_val) for eps_val in eps for omega_val in omega]

    import pandas as pd

    df = pd.DataFrame(
        combinations,
        columns=[
            "eps",
            "omega",
        ],
    )
    
    df["eps_norm"]=np.nan

    df["r_true"] = np.nan
    df["r"] = np.nan
    df["err"] = np.nan

    for i in range(methods_num):
        df["r" + str(i)] = np.nan
    
    for i in range(methods_num):
        df["err" + str(i)] = np.nan

    for i, (eps_val, omega_val) in enumerate(combinations):
        print(f"Running benchmark {i + 1}/{len(combinations)}")
         
        (rank_true, rank, error, rk, err) = bench(eps_val, omega_val)
        
        df.loc[i] = [eps_val, omega_val, eps_val*normA, rank_true, rank, error, *rk, *err]

        df.to_csv(f"hankel_error_rank_data_bench_fid_seed{seed}.csv", index=False)
