# %%
import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
import numpy as np
import time as tic


rng = np.random.Generator(np.random.MT19937(42))
trials = 3
direction = 1
set_rank = 20
cond_num = 1e3


def bench(tol, omega, partitions, total_m, total_n):

    if direction == 0:
        N = partitions
        M = partitions
        m = int(total_m / M)
        n = int(total_n / N)
        partition = N
        block_n = n
        block_m = M * m
        outer_slices = N
        inner_slices = M
    else:
        N = partitions
        M = partitions
        m = int(total_m / M)
        n = int(total_n / N)
        partition = M
        block_n = N * n
        block_m = m
        outer_slices = M
        inner_slices = N

    tree_list = [
        trees.dist_hasvd_tree(partition, direction, (block_m, block_n)),
        trees.inc_hasvd_tree(partition, direction, (block_m, block_n)),
        trees.tlbd_dist_hasvd_tree(outer_slices, inner_slices, direction, (m, n)),
        trees.tlbd_inc_hasvd_tree(outer_slices, inner_slices, direction, (m, n)),
    ]

    for tree in tree_list:
        trees.assert_shape_consistency(tree)

    err_lapack = [None] * trials
    rank = [None] * trials
    time_lapack = [None] * trials

    err_tight = [None] * len(tree_list)
    rank_tight = [None] * len(tree_list)
    time = [None] * len(tree_list)

    for i in range(len(tree_list)):
        err_tight[i] = [None] * trials
        rank_tight[i] = [None] * trials
        time[i] = [None] * trials

    for i in range(trials):
        A = matrix.random_matrix(m * M, n * N, set_rank, rng, cond_num)
        start = tic.time()
        U, S, Vt = svd.svd_with_tol(A, truncate_tol=tol)
        time_lapack[i] = tic.time() - start
        err_lapack[i] = np.linalg.norm(A - U @ np.diag(S) @ Vt)
        rank[i] = len(S)

        for idx, tree in enumerate(tree_list):
            branch_count = trees.branch_node_count(tree)

            def nodal_error(node):
                return errors.tight_error(node, tol, omega, branch_count)

            if idx < 2:

                leaf_to_block_map = trees.linear_general_ltb_map(
                    A, block_m, block_n, direction
                )

            else:

                leaf_to_block_map = trees.tlbd_general_ltb_map(
                    A, M, N, block_m, block_n, direction
                )

            start = tic.time()
            U, S, Vt = svd.hasvd(tree, leaf_to_block_map, local_eps=nodal_error)
            time[idx][i] = tic.time() - start
            err_tight[idx][i] = np.linalg.norm(A - U @ np.diag(S) @ Vt)
            rank_tight[idx][i] = len(S)

    err_lapack = np.mean(err_lapack)
    rank = np.mean(rank)
    time_lapack = np.mean(time_lapack)
    for i in range(len(tree_list)):
        err_tight[i] = np.mean(err_tight[i])
        rank_tight[i] = np.mean(rank_tight[i])
        time[i] = np.mean(time[i])

    return err_lapack, rank, time_lapack, err_tight, rank_tight, time


if __name__ == "__main__":
    sizes = [100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000]
    partitions = [10]
    eps = [1e-5, 1e-10]
    omegas = [0.1, 0.9]

    combinations = [
        (part_val, size_val, eps_val, omega_val)
        for size_val in sizes
        for part_val in partitions
        for eps_val in eps
        for omega_val in omegas
    ]

    import pandas as pd

    df = pd.DataFrame(
        combinations,
        columns=[
            "size",
            "partitions",
            "eps",
            "omega",
        ],
    )

    df["err"] = np.nan
    df["rk"] = np.nan
    df["time"] = np.nan
    # df["err_naive_0"] = np.nan
    # df["rk_naive_0"] = np.nan
    df["err_tight_0"] = np.nan
    df["rk_tight_0"] = np.nan
    df["time_tight_0"] = np.nan
    # df["err_naive_1"] = np.nan
    # df["rk_naive_1"] = np.nan
    df["err_tight_1"] = np.nan
    df["rk_tight_1"] = np.nan
    df["time_tight_1"] = np.nan
    df["err_tight_2"] = np.nan
    df["rk_tight_2"] = np.nan
    df["time_tight_2"] = np.nan
    df["err_tight_3"] = np.nan
    df["rk_tight_3"] = np.nan
    df["time_tight_3"] = np.nan

    for i, (part_val, size_val, eps_val, omega_val) in enumerate(combinations):
        print(f"Running benchmark {i + 1}/{len(combinations)}")
        (err_lapack, rank, time_lapack, err_tight, rank_tight, time) = bench(
            eps_val, omega_val, part_val, size_val, size_val
        )
        df.loc[i] = [
            size_val,
            part_val,
            eps_val,
            omega_val,
            err_lapack,
            rank,
            time_lapack,
            # err_naive[0],
            # rank_naive[0],
            err_tight[0],
            rank_tight[0],
            time[0],
            # err_naive[1],
            # rank_naive[1],
            err_tight[1],
            rank_tight[1],
            time[1],
            err_tight[2],
            rank_tight[2],
            time[2],
            err_tight[3],
            rank_tight[3],
            time[3],
        ]

        df.to_csv("general_time_error_data_bench_2.csv", index=False)

# %%
