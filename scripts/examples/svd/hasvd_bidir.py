# %%
import hasvd.utils.trees as trees
import hasvd.utils.errors as errors
import hasvd.utils.matrix as matrix
import hasvd.utils.svd as svd
import time
import numpy as np


def nodemap_hasvd(tree: trees.hasvd_Node, A_array, m, n, M, N):
    """
    Create a mapping of nodes to their corresponding matrix blocks in a internal linear HASVD tree.
    Parameters
    ----------
    tree : utils.hasvd_Node
        The root node of the tree.
    Returns
    -------
    node_to_block : dict
        A dictionary mapping nodes to their corresponding matrix blocks.
    """
    node_to_block = {}
    for node in tree.traverse():
        # Check if the node is a leaf node
        if node.is_leaf:
            block_coord = (
                np.floor_divide(node.tag, N) * m,
                ((node.tag) % N) * n,
            )
            block_shape = (m, n)
            # Get the block corresponding to the node
            block = matrix.array_to_hankel(
                A_array,
                (m * M, n * N),
                block_coord,
                block_shape,
            )
            node_to_block[node] = block
    return node_to_block


if __name__ == "__main__":

    M = 10
    N = 20
    m = 150
    n = 150
    rng = np.random.default_rng(12)
    eps = 1.0e-4
    omega = 0.1

    # %%
    # Create a random matrix with block size m x n with M xN blocks
    A = matrix.random_hankel(m * M, n * N, rng, "fid")
    # print("A:", A)

    print("Generated random Hankel matrix...")
    rank = np.linalg.matrix_rank(A)
    print("Matrix rank sanity check:", rank)
    print("Matrix shape:", A.shape)
    print("Block shape:", "(", m, ",", n, ")")
    print("Block partitions:", "(", M, ",", N, ")\n")

    A_array = matrix.hankel_to_array(A)
    print("Approximate SVD with tolerance :", eps, " and omega:", omega, "\n")

    print("\u0332".join("LAPACK gesdd"))
    # SVD of the whole matrix
    start = time.time()
    U, E, Vh = svd.svd_with_tol(
        A,
        full_matrices=False,
        truncate_tol=eps,
    )
    duration = time.time() - start
    rank = len(E)
    print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U @ np.diag(E) @ Vh), "\n")

    # DISTRIBUTED HASVD
    print("\u0332".join("Distributed HASVD"))
    # Create a tree for the distributed HASVD
    tree1 = trees.tlbd_dist_hasvd_tree(M, N, 1, block_shape=(m, n))

    # Create mapping of nodes to their corresponding matrix blocks
    node_to_block = nodemap_hasvd(
        tree1,
        A_array,
        m,
        n,
        M,
        N,
    )

    # Create a function to get the block for a node
    def get_dist_hasvd_block(node):
        return node_to_block[node]

    # Create mapping of nodes to their corresponding naive error

    # Sum non-leaf nodes
    num_nonleaf_nodes = 0
    for node in tree1.traverse():
        if not node.is_leaf:
            num_nonleaf_nodes += 1

    # Sum of branching nodes
    num_branching_nodes = 0
    for node in tree1.traverse():
        if not node.is_leaf and any(not child.is_leaf for child in node.children):
            num_branching_nodes += 1

    # NAIVE ERROR

    # Create a function to get the naive error for a node
    def get_dist_naive_error(node):
        return errors.naive_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U1, E1, Vh1, rank1n = svd.hasvd(
        tree1, get_dist_hasvd_block, local_eps=get_dist_naive_error, track_ranks=True
    )
    duration = time.time() - start
    print("- Naive Prescription -")
    print("Error:", np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U1 @ np.diag(E1) @ Vh1), "\n")

    svd.rank_analysis(tree1, rank1n)

    # TIGHT ERROR

    # Create a function to get the tight error for a node
    def get_dist_tight_error(node):
        return errors.tight_error(
            node,
            eps,
            omega,
            num_branching_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U1, E1, Vh1, rank1t = svd.hasvd(
        tree1, get_dist_hasvd_block, local_eps=get_dist_tight_error, track_ranks=True
    )
    duration = time.time() - start
    print("\n")
    print("- Tight Prescription -")
    print("Error:", np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U1 @ np.diag(E1) @ Vh1), "\n")
    svd.rank_analysis(tree1, rank1t)

    # INCREMENTAL HASVD
    print("\n")
    print("\u0332".join("Incremental HASVD"))
    # Create a tree for the incremental HASVD
    tree2 = trees.tlbd_inc_hasvd_tree(M, N, 1, 0, (m, n))
    # trees.draw_nxgraph(tree)
    # Create mapping of nodes to their corresponding matrix blocks
    node_to_block = nodemap_hasvd(
        tree2,
        A_array,
        m,
        n,
        M,
        N,
    )

    # Create a function to get the block for a node
    def get_inc_hasvd_block(node):
        return node_to_block[node]

    # Sum non-leaf nodes
    num_nonleaf_nodes = 0
    for node in tree2.traverse():
        if not node.is_leaf:
            num_nonleaf_nodes += 1

    # NAIVE ERROR

    # Create a function to get the naive error for a node
    def get_inc_naive_error(node):
        return errors.naive_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U2, E2, Vh2, rank2n = svd.hasvd(
        tree2, get_inc_hasvd_block, local_eps=get_inc_naive_error, track_ranks=True
    )
    duration = time.time() - start
    print("- Naive Prescription -")
    print("Error:", np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U2 @ np.diag(E2) @ Vh2), "\n")
    svd.rank_analysis(tree2, rank2n)
    print("\n")

    # TIGHT ERROR
    # Create a function to get the tight error for a node
    def get_inc_tight_error(node):
        return errors.tight_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U2, E2, Vh2, rank2t = svd.hasvd(
        tree2, get_inc_hasvd_block, local_eps=get_inc_tight_error, track_ranks=True
    )
    duration = time.time() - start
    print("- Tight Prescription -")
    print("Error:", np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U2 @ np.diag(E2) @ Vh2), "\n")
    svd.rank_analysis(tree2, rank2t)

    # %%
    # Create a random matrix with block size m x n with M xN blocks
    A = matrix.random_matrix(m * M, n * N, rank, rng)
    # print("A:", A)
    print("Generated random general matrix...")
    print("Prescribed rank:", rank)
    print("Matrix rank sanity check:", np.linalg.matrix_rank(A))
    print("Matrix shape:", A.shape)
    print("Block shape:", "(", m, ",", n, ")")
    print("Block partitions:", "(", M, ",", N, ")\n")

    direction = 1
    print("Approximate SVD with tolerance :", eps, " and omega:", omega, "\n")

    print("\u0332".join("LAPACK gesdd"))
    # SVD of the whole matrix
    start = time.time()
    U, E, Vh = svd.svd_with_tol(
        A,
        full_matrices=False,
        truncate_tol=eps,
    )
    duration = time.time() - start
    print("Error:", np.linalg.norm(A - U @ np.diag(E) @ Vh))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U @ np.diag(E) @ Vh), "\n")

    # DISTRIBUTED HASVD
    print("\u0332".join("Distributed HASVD"))
    # Create a tree for the distributed HASVD
    tree1 = trees.tlbd_dist_hasvd_tree(M, N, 1, block_shape=(m, n))

    def node_to_block_map(node: trees.hasvd_Node):
        if direction == 0:
            row_pos = int(node.tag % M) * m
            col_pos = int(node.tag // M) * n
        else:
            col_pos = int(node.tag % N) * n
            row_pos = int((node.tag // N)) * m
        return A[row_pos : row_pos + m, col_pos : col_pos + n]

    # Create mapping of nodes to their corresponding naive error

    # Sum non-leaf nodes
    num_nonleaf_nodes = 0
    for node in tree1.traverse():
        if not node.is_leaf:
            num_nonleaf_nodes += 1

    # Sum of branching nodes
    num_branching_nodes = 0
    for node in tree1.traverse():
        if not node.is_leaf and any(not child.is_leaf for child in node.children):
            num_branching_nodes += 1

    # NAIVE ERROR

    # Create a function to get the naive error for a node
    def get_dist_naive_error(node):
        return errors.naive_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U1, E1, Vh1, rank1n = svd.hasvd(
        tree1, node_to_block_map, local_eps=get_dist_naive_error, track_ranks=True
    )
    duration = time.time() - start
    print("- Naive Prescription -")
    print("Error:", np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U1 @ np.diag(E1) @ Vh1), "\n")

    svd.rank_analysis(tree1, rank1n)

    # TIGHT ERROR

    # Create a function to get the tight error for a node
    def get_dist_tight_error(node):
        return errors.tight_error(
            node,
            eps,
            omega,
            num_branching_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U1, E1, Vh1, rank1t = svd.hasvd(
        tree1, node_to_block_map, local_eps=get_dist_tight_error, track_ranks=True
    )
    duration = time.time() - start
    print("\n")
    print("- Tight Prescription -")
    print("Error:", np.linalg.norm(A - U1 @ np.diag(E1) @ Vh1))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U1 @ np.diag(E1) @ Vh1), "\n")
    svd.rank_analysis(tree1, rank1t)
    print("\n")

    # INCREMENTAL HASVD
    print("\u0332".join("Incremental HASVD"))
    # Create a tree for the incremental HASVD
    tree2 = trees.tlbd_inc_hasvd_tree(M, N, 1, 0, (m, n))
    # trees.draw_nxgraph(tree)

    # Sum non-leaf nodes
    num_nonleaf_nodes = 0
    for node in tree2.traverse():
        if not node.is_leaf:
            num_nonleaf_nodes += 1

    # NAIVE ERROR

    # Create a function to get the naive error for a node
    def get_inc_naive_error(node):
        return errors.naive_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U2, E2, Vh2, rank2n = svd.hasvd(
        tree2, node_to_block_map, local_eps=get_inc_naive_error, track_ranks=True
    )
    duration = time.time() - start
    print("- Naive Prescription -")
    print("Error:", np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U2 @ np.diag(E2) @ Vh2), "\n")
    svd.rank_analysis(tree2, rank2n)
    print("\n")

    # TIGHT ERROR
    # Create a function to get the tight error for a node
    def get_inc_tight_error(node):
        return errors.tight_error(
            node,
            eps,
            omega,
            num_nonleaf_nodes,
        )

    # SVD of the blocks
    start = time.time()
    U2, E2, Vh2, rank2t = svd.hasvd(
        tree2, node_to_block_map, local_eps=get_inc_tight_error, track_ranks=True
    )
    duration = time.time() - start
    print("- Tight Prescription -")
    print("Error:", np.linalg.norm(A - U2 @ np.diag(E2) @ Vh2))
    print("Time of SVD:", duration)
    print("Rank of U*S*Vt:", np.linalg.matrix_rank(U2 @ np.diag(E2) @ Vh2), "\n")
    svd.rank_analysis(tree2, rank2t)
