import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
import numpy as np
import numpy.linalg as la
from time import perf_counter

seed = 42
rng = np.random.Generator(np.random.MT19937(seed))
set_rank = 100
trials = 10


def bench(tol, omega, partition, size, direction):

    total_n = size
    N = partition
    n = int(total_n / N)

    if direction == 0:
        linear_shape = (total_n, n)
    elif direction == 1:
        linear_shape = (n, total_n)

    A_array = matrix.lrf_sequence(set_rank, 2 * total_n - 1, rng)
    A = matrix.array_to_hankel(A_array, (total_n, total_n))
    normA = np.linalg.norm(A)

    tree_arr = [
        trees.dist_hasvd_tree(N, direction, linear_shape),
        trees.inc_hasvd_tree(N, direction, linear_shape),
        trees.tlbd_dist_hasvd_tree(N, N, direction, (n, n)),
        trees.tlbd_inc_hasvd_tree(N, N, direction, (n, n)),
        trees.regular_alt_inc_tree(n, total_n, direction),
    ]

    tol *= normA

    nodal_errors = []
    for tree in tree_arr:
        count = trees.branch_node_count(tree)
        nodal_errors.append(
            lambda node, count=count: errors.tight_error(node, tol, omega, count)
        )

    err = np.zeros(len(tree_arr))
    tim = np.zeros(len(tree_arr))
    error = 0
    time = 0

    for _ in range(trials):

        # Standard
        start = perf_counter()
        U, E, Vh = svd.svd_with_tol(A, full_matrices=False, truncate_tol=tol)
        time += perf_counter() - start
        error += la.norm(A - U @ np.diag(E) @ Vh)

        ltb_map = [
            trees.linear_hankelarray_ltb_map(
                A_array, N, linear_shape[0], linear_shape[1], direction
            ),
            trees.linear_hankelarray_ltb_map(
                A_array, N, linear_shape[0], linear_shape[1], direction
            ),
            trees.tlbd_hankelarray_ltb_map(A_array, N, N, n, n, direction),
            trees.tlbd_hankelarray_ltb_map(A_array, N, N, n, n, direction),
            trees.regular_alt_inc_general_ltb_map(A, n, direction),
        ]

        # HASVD
        for idx, tree in enumerate(tree_arr):
            start = perf_counter()
            U, E, Vh = svd.hasvd(
                tree,
                ltb_map[idx],
                nodal_errors[idx],
            )
            tim[idx] += perf_counter() - start
            err[idx] += la.norm(A - U @ np.diag(E) @ Vh)

    error /= trials
    time /= trials

    for i in range(len(tree_arr)):
        err[i] /= trials
        tim[i] /= trials

    return tol, error, time, err, tim


if __name__ == "__main__":
    eps = [1e-1, 1e-5, 1e-10]
    omega = [0.01, 0.1, 0.5, 0.9]
    partitions = [10, 20, 50]
    sizes = [100, 200, 500, 800, 1000, 2000, 5000, 8000, 10000]
    direction = [0, 1]
    methods_num = 5

    combinations = [
        (eps_val, omega_val, size_val, part_val, dir_val)
        for eps_val in eps
        for omega_val in omega
        for size_val in sizes
        for part_val in partitions
        for dir_val in direction
    ]

    import pandas as pd

    df = pd.DataFrame(
        combinations,
        columns=["eps", "omega", "size", "parts", "direction"],
    )

    df["eps_norm"] = np.nan
    df["time"] = np.nan
    df["err"] = np.nan

    for i in range(methods_num):
        df["tim" + str(i)] = np.nan

    for i in range(methods_num):
        df["err" + str(i)] = np.nan

    for i, (eps_val, omega_val, size_val, part_val, dir_val) in enumerate(combinations):
        print(f"Running benchmark {i + 1}/{len(combinations)}")

        (tolnorm, error, time, err, tim) = bench(
            eps_val, omega_val, part_val, size_val, dir_val
        )

        df.loc[i] = [
            eps_val,
            omega_val,
            size_val,
            part_val,
            dir_val,
            tolnorm,
            time,
            error,
            *tim,
            *err,
        ]

        df.to_csv(f"hankel_time_bench_size_fid_seed{seed}.csv", index=False)
