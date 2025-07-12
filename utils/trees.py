# TREES

from pymor.algorithms.hapod import Node
from hasvd.utils.matrix import array_to_hankel
import numpy as np

from typing import Literal


class hasvd_Node(Node):
    """docstring for hasvd_Node."""

    def __init__(
        self,
        shape: tuple[int, int] = None,
        direction=0,
        tag=None,
        id=None,
        parent=None,
        after=None,
    ):
        super().__init__(tag=tag, parent=parent, after=after)
        self.direction = direction
        self.m = shape[0]
        self.n = shape[1]
        self.root = parent.root if parent else self
        self.id = id

    def add_child(self, direction=2, tag=None, id=None, after=None, **kwargs):
        return hasvd_Node(
            direction=direction, tag=tag, id=id, parent=self, after=after, **kwargs
        )

    @property
    def depth(self):
        return super().depth

    @property
    def is_leaf(self):
        return super().is_leaf

    @property
    def is_root(self):
        return super().is_root

    def traverse(self, return_level=False):
        return super().traverse(return_level)

    def __str__(self):
        lines = []
        for node, level in self.traverse(True):
            line = ""
            if node.parent:
                p = node.parent
                while p.parent:
                    if p.parent.children.index(p) + 1 == len(p.parent.children):
                        line = "  " + line
                    else:
                        line = "| " + line
                    p = p.parent
                match node.direction:
                    case 0:
                        line += "+-c"
                    case 1:
                        line += "+-r"
                    case 2:
                        line += "+-l"
            else:
                match node.direction:
                    case 0:
                        line += "c"
                    case 1:
                        line += "r"
                    case 2:
                        line += "l"
            if node.tag is not None:
                line += f" {node.tag}"

            line += f" [{node.m}x{node.n}]"

            if node.after:
                line += f' (after {",".join(str(a) for a in node.after)})'
            lines.append(line)
        return "\n".join(lines)

    def validity_check(self):
        """Check the validity of the tree."""
        assert_shape_consistency(self)


# One-Level Trees


def inc_hasvd_tree(
    num_slices: int,
    direction: int = 0,
    block_shape: tuple[int, int] = None,
):
    """
    Construct an incremental HASVD tree (serial) in bidirectional style.

    Each new leaf is added with a dependency on the previous snapshot.

    Parameters
    ----------
    num_slices : int
        Number of individual snapshot blocks.
    direction : int, optional
        Aggregation direction (default: 0).
    block_shape : tuple[int, int], optional
        Shape of each leaf block.

    Returns
    -------
    hasvd_Node
        Root of the constructed HASVD tree.
    """

    match direction:
        case 0:
            shape = (block_shape[0], num_slices * block_shape[1])
        case 1:
            shape = (num_slices * block_shape[0], block_shape[1])

    root = hasvd_Node(
        tag="r",
        id="r",
        direction=direction,
        shape=shape,
    )
    total_leaves = num_slices - 1

    leaf_tag = 0
    outer_tag = total_leaves + 1
    parent_node = root

    for outer_idx in range(num_slices):
        leaf_node = parent_node.add_child(
            tag=leaf_tag,
            id=leaf_tag,
            direction=2,  # leaf direction (optional, for consistency)
            shape=block_shape,
        )
        leaf_tag += 1
        if outer_idx < num_slices - 2:
            match direction:
                case 0:
                    shape = (shape[0], shape[1] - block_shape[1])
                case 1:
                    shape = (shape[0] - block_shape[0], shape[1])

            parent_node = parent_node.add_child(
                tag=outer_tag,
                id=outer_tag,
                direction=direction,
                shape=shape,
            )
            outer_tag += 1

    return root


def dist_hasvd_tree(
    num_slices: int,
    direction: int = 0,
    block_shape: tuple[int, int] = None,
):
    """
    Construct a one-level distributed HASVD tree using bidirectional-style coding.

    Parameters
    ----------
    num_slices : int
        Number of leaf slices.
    direction : int, optional
        Aggregation direction (default: 0).
    block_shape : tuple[int, int], optional
        Shape of each block.

    Returns
    -------
    hasvd_Node
        Root of the constructed HASVD tree.
    """
    root = hasvd_Node(
        tag="r",
        id="r",
        direction=direction,
        shape=(num_slices * block_shape[0], block_shape[1]),
    )

    for tag in range(num_slices):
        root.add_child(
            tag=tag,
            id=tag,
            direction=2,  # leaf direction (optional, for consistency)
            shape=(block_shape[0], block_shape[1]),
        )

    return root


def linear_general_ltb_map(A: np.ndarray, m: int, n: int, direction: int):
    """Block-to-leaf map generator connecting general matrices to two-level bidirectional trees.

    Parameters
    ----------
    A : np.ndarray
        Matrix
    partitions : int
        Number of partitions
    m : int
        Row size of block
    n : int
        Column size of block
    direction : int
        Direction of aggregation
    """

    def map(node: hasvd_Node):
        """Block-to-leaf map of general matrix for two-level bidirectional trees.

        Parameters
        ----------
        node : hasvd_Node
            Node in tree.

        Returns
        -------
        np.ndarray
            Block to corresponding node.
        """
        assert node.is_leaf, "Node is not leaf."
        if direction == 0:
            row_pos = 0
            col_pos = int(node.tag) * n
        elif direction == 1:
            row_pos = int(node.tag) * m
            col_pos = 0
        return A[row_pos : row_pos + m, col_pos : col_pos + n]

    return map


def linear_hankelarray_ltb_map(
    A_array: np.ndarray, partitions, m: int, n: int, direction: int
):
    """Block-to-leaf map generator connecting Hankel matrices to two-level bidirectional trees.

    Parameters
    ----------
    A : np.ndarray
        Hankel element sequence
    partitions : int
        Number of partitions
    m : int
        Row size of block
    n : int
        Column size of block
    direction : int
        Direction of aggregation
    """

    def map(node: hasvd_Node):
        """Block-to-leaf map of Hankel matrix for two-level bidirectional trees.

        Parameters
        ----------
        node : hasvd_Node
            Node in tree.

        Returns
        -------
        np.ndarray
            Block to corresponding node.
        """
        assert node.is_leaf, "Node is not leaf."

        if direction == 0:
            M = 1
            N = partitions
            row_pos = 0
            col_pos = int(node.tag) * n
        elif direction == 1:
            M = partitions
            N = 1
            row_pos = int(node.tag) * m
            col_pos = 0
        return array_to_hankel(
            A_array,
            (m * M, n * N),
            (row_pos, col_pos),
            (m, n),
        )

    return map


# Two-Level Bidirectional trees


def tlbd_dist_hasvd_tree(
    num_outer_slices: int,
    num_inner_slices: int,
    outer_direction: int = 0,
    block_shape: tuple[int, int] = None,
    recursion: Literal[None, "hankel"] = None,
):
    """
    Build a two-level hierarchical HASVD tree with full control over slice partitioning and directions.

    Parameters
    ----------
    num_outer_slices : int
        Number of top-level groups (outer partitions).
    num_inner_slices : int
        Number of inner slices per outer group (inner partitions).
    outer_direction : int, optional
        Aggregation direction at the outer level (default: 0).

    Returns
    -------
    hasvd_Node
        Root of the constructed HASVD tree.
    """
    total_m = block_shape[0]
    total_n = block_shape[1]
    if outer_direction == 0:
        total_n *= num_outer_slices
        total_m *= num_inner_slices
    else:
        total_m *= num_outer_slices
        total_n *= num_inner_slices
    total_leaves = num_outer_slices * num_inner_slices
    root = hasvd_Node(
        tag="r",
        id="r",
        direction=outer_direction,
        shape=(total_m, total_n),
    )

    # Inner nodes get tags 0, 1, ..., total_slices-1
    # Outer nodes get tags total_slices, total_slices+1, ...
    outer_tag = total_leaves
    match outer_direction:
        case 0:
            shape = (total_m, block_shape[1])
        case 1:
            shape = (block_shape[0], total_n)
    for outer_idx in range(num_outer_slices):
        outer_node = root.add_child(
            tag=outer_tag,
            id=outer_tag,
            direction=(outer_direction + 1) % 2,
            shape=shape,
        )
        outer_tag += 1
        for inner_idx in range(num_inner_slices):
            inner_tag = outer_idx * num_inner_slices + inner_idx
            if recursion == None:
                inner_id = inner_tag
            elif recursion == "hankel":
                inner_id = outer_idx + inner_idx
            outer_node.add_child(tag=inner_tag, id=inner_id, shape=block_shape)

    return root


def tlbd_inc_hasvd_tree(
    num_outer_slices: int,
    num_inner_slices: int,
    outer_direction: int = 0,
    block_shape: tuple[int, int] = None,
    recursion: Literal[None, "hankel"] = None,
):
    """
    Build a two-level hierarchical HASVD tree with full control over slice partitioning and directions.

    Parameters
    ----------
    num_outer_slices : int
        Number of top-level groups (outer partitions).
    num_inner_slices : int
        Number of inner slices per outer group (inner partitions).
    outer_direction : int, optional
        Aggregation direction at the outer level (default: 0).

    Returns
    -------
    hasvd_Node
        Root of the constructed HASVD tree.
    """
    total_m = block_shape[0]
    total_n = block_shape[1]

    if outer_direction == 0:
        total_n *= num_outer_slices
        total_m *= num_inner_slices
    else:
        total_n *= num_inner_slices
        total_m *= num_outer_slices

    total_leaves = num_outer_slices * num_inner_slices
    root = hasvd_Node(
        tag="r",
        id="r",
        direction=outer_direction,
        shape=(total_m, total_n),
    )

    leaf_tag = 0
    outer_tag = total_leaves + 1
    parent_node = root

    for outer_idx in range(num_outer_slices):
        match outer_direction:
            case 0:
                shape = (total_m, block_shape[1])
            case 1:
                shape = (block_shape[0], total_n)
        outer_node = parent_node.add_child(
            tag=outer_tag,
            id=outer_tag,
            direction=(outer_direction + 1) % 2,
            shape=shape,
        )
        outer_tag += 1
        for inner_idx in range(num_inner_slices):
            if recursion == None:
                inner_id = leaf_tag
            elif recursion == "hankel":
                inner_id = inner_idx + outer_idx
            outer_node.add_child(tag=leaf_tag, id=inner_id, shape=block_shape)
            leaf_tag += 1
        if outer_idx < num_outer_slices - 2:
            match outer_direction:
                case 0:
                    shape = (total_m, total_n - (1 + outer_idx) * block_shape[1])
                case 1:
                    shape = (total_m - (1 + outer_idx) * block_shape[0], total_n)
            merge_node = parent_node.add_child(
                tag=outer_tag,
                id=outer_tag,
                direction=outer_direction,
                shape=shape,
            )
            outer_tag += 1
            parent_node = merge_node
    return root


def tlbd_general_ltb_map(
    A: np.ndarray, M: int, N: int, m: int, n: int, outer_direction: int
):
    """Block-to-leaf map generator connecting general matrices to two-level bidirectional trees.

    Parameters
    ----------
    A : np.ndarray
        Matrix
    M : int
        Number of row blocks
    N : int
        Number of column blocks
    m : int
        Row size of block
    n : int
        Column size of block
    outer_direction : int
        Outer top-level direction of aggregation
    """

    def map(node: hasvd_Node):
        """Block-to-leaf map of general matrix for two-level bidirectional trees.

        Parameters
        ----------
        node : hasvd_Node
            Node in tree.

        Returns
        -------
        np.ndarray
            Block to corresponding node.
        """
        assert node.is_leaf, "Node is not leaf."
        if outer_direction == 0:
            row_pos = int(node.tag % M) * m
            col_pos = int(node.tag // M) * n
        else:
            col_pos = int(node.tag % N) * n
            row_pos = int((node.tag // N)) * m
        return A[row_pos : row_pos + m, col_pos : col_pos + n]

    return map


def tlbd_hankelarray_ltb_map(
    A_array: np.ndarray, M: int, N: int, m: int, n: int, outer_direction: int
):
    """Block-to-leaf map connecting Hankel matrices to two-linear bidirectional trees.

    Parameters
    ----------
    A_array : np.ndarray
        Hankel element sequence
    M : int
        Number of row blocks
    N : int
        Number of column blocks
    m : int
        Row size of block
    n : int
        Column size of block
    outer_direction : int
        Outer top-level direction of aggregation
    """

    def map(node: hasvd_Node):

        if outer_direction == 0:
            row_pos = node.tag % M * m
            col_pos = int(node.tag // M) * n
        else:
            row_pos = int((node.tag // N)) * m
            col_pos = node.tag % N * n

        return array_to_hankel(
            A_array,
            (m * M, n * N),
            (row_pos, col_pos),
            (m, n),
        )

    return map


def tlbd_hankelblockarray_ltb_map(
    block_array: np.ndarray, M: int, N: int, outer_direction: int, transpose=False
):
    """Block-to-leaf map connecting Hankel matrices to two-linear bidirectional trees.

    Parameters
    ----------
    A_array : np.ndarray
        Hankel block sequence
    M : int
        Number of row blocks
    N : int
        Number of column blocks
    m : int
        Row size of block
    n : int
        Column size of block
    outer_direction : int
        Outer top-level direction of aggregation
    """

    def map(node: hasvd_Node):

        if outer_direction == 0:
            row_pos = node.tag % M
            col_pos = int(node.tag // M)
        else:
            row_pos = int((node.tag // N))
            col_pos = node.tag % N

        return block_array[row_pos + col_pos]

    return map


# Alternating Incremental Trees


def alt_inc_tree(
    dblock_lengths: list[int],
    outer_direction: int,
    recursion: Literal[None, "hankel"] = None,
):

    diagonal_num = len(dblock_lengths)
    total_m = sum(dblock_lengths)

    main_rect_idx = 0
    sub_rect_idx = diagonal_num - 1
    diagonal_idx = (diagonal_num - 1) * 2
    outer_idx = diagonal_num + (diagonal_num - 1) * 2

    tree = hasvd_Node(
        tag="r", id="r", direction=outer_direction, shape=(total_m, total_m)
    )

    parent = tree

    for i in range(0, diagonal_num - 1):
        d = dblock_lengths[i]

        if outer_direction == 0:
            skinny_shape = (parent.m, d)
            rect_shape = (skinny_shape[0] - d, d)
            fat_shape = (skinny_shape[0], parent.n - d)
        elif outer_direction == 1:
            skinny_shape = (d, parent.n)
            rect_shape = (d, skinny_shape[1] - d)
            fat_shape = (parent.m - d, skinny_shape[1])

        skinny_part = parent.add_child(
            tag=outer_idx,
            id=outer_idx,
            direction=(outer_direction + 1) % 2,
            shape=skinny_shape,
        )
        outer_idx += 1

        skinny_part.add_child(tag=diagonal_idx, id=diagonal_idx, shape=(d, d))
        diagonal_idx += 1

        skinny_part.add_child(tag=main_rect_idx, id=main_rect_idx, shape=rect_shape)

        fat_part = parent.add_child(
            tag=outer_idx,
            id=outer_idx,
            direction=(outer_direction + 1) % 2,
            shape=fat_shape,
        )
        outer_idx += 1

        if recursion == None:
            idx = sub_rect_idx
        elif recursion == "hankel":
            idx = main_rect_idx

        fat_part.add_child(tag=sub_rect_idx, id=idx, shape=rect_shape[::-1])
        sub_rect_idx += 1
        main_rect_idx += 1

        if i == diagonal_num - 2:
            fat_part.add_child(
                tag=diagonal_idx,
                id=diagonal_idx,
                shape=(dblock_lengths[i + 1], dblock_lengths[i + 1]),
            )
        else:
            parent = fat_part.add_child(
                tag=outer_idx,
                id=outer_idx,
                direction=outer_direction,
                shape=(parent.m - d, parent.n - d),
            )
            outer_idx += 1

    return tree


def regular_alt_inc_tree(
    dblock_length: int,
    matrix_length: int,
    outer_direction: int,
    recursion: Literal[None, "hankel"] = None,
):

    assert (
        matrix_length % dblock_length == 0
    ), "Number of blocks and matrix do not coincide!"

    dblock_num = int(matrix_length / dblock_length)
    dblock_lengths = [dblock_length] * dblock_num

    print(
        f"Generated regular alternating tree with {dblock_num} square diagonal blocks..."
    )

    return alt_inc_tree(dblock_lengths, outer_direction, recursion=recursion)


def alt_inc_general_ltb_map(
    A: np.ndarray, dblock_lengths: list[int], outer_direction: int
):

    diagonal_num = len(dblock_lengths)

    total_length = sum(dblock_lengths)

    assert A.shape == (
        total_length,
        total_length,
    ), f"Shape mismatch: {A.shape} != {(total_length,total_length)}"

    def map(node: hasvd_Node):

        if node.tag < diagonal_num - 1:

            block_type = 0  # identify as main rectangle
            posid = node.tag

        elif node.tag < (diagonal_num - 1) * 2:

            block_type = 1  # identify as sub rectangle
            posid = node.tag - diagonal_num + 1

        elif node.tag < (diagonal_num - 1) * 2 + diagonal_num:

            posid = node.tag - (diagonal_num - 1) * 2

            min_pos = sum(dblock_lengths[:posid])
            max_pos = sum(dblock_lengths[: posid + 1])

            return A[min_pos:max_pos, min_pos:max_pos]
        else:
            raise "Node is not leaf!"

        if outer_direction == 0:

            if block_type == 0:

                row_pos = sum(dblock_lengths[: posid + 1])
                col_pos = sum(dblock_lengths[:posid])

                n = dblock_lengths[posid]

                return A[row_pos:, col_pos : col_pos + n]

            elif block_type == 1:

                row_pos = sum(dblock_lengths[:posid])
                col_pos = sum(dblock_lengths[: posid + 1])

                m = dblock_lengths[posid]

                return A[row_pos : row_pos + m, col_pos:]

        elif outer_direction == 1:

            if block_type == 0:

                row_pos = sum(dblock_lengths[:posid])
                col_pos = sum(dblock_lengths[: posid + 1])

                m = dblock_lengths[posid]

                return A[row_pos : row_pos + m, col_pos:]

            elif block_type == 1:

                row_pos = sum(dblock_lengths[: posid + 1])
                col_pos = sum(dblock_lengths[:posid])

                n = dblock_lengths[posid]

                return A[row_pos:, col_pos : col_pos + n]

    return map


def regular_alt_inc_general_ltb_map(
    A: np.ndarray, dblock_length: int, outer_direction: int
):
    assert A.shape[0] == A.shape[1], "Matrix not square"
    assert (
        A.shape[0] % dblock_length == 0
    ), "Number of blocks and matrix do not coincide!"

    dblock_num = int(A.shape[0] / dblock_length)

    dblock_lengths = [dblock_length] * dblock_num
    return alt_inc_general_ltb_map(A, dblock_lengths, outer_direction)


def find_geometric_sequences(M: int, min_a: int = 10):
    """Find all geometric sequences of the form a, ar, ar^2, ..., ar^(n-1) such that the sum is M.
    The first term 'a' must be greater than or equal to min_a. Good to use for finding sequence of diagonal sizes with geometric growth for alternating incremental HASVD.

    Parameters
    ----------
    M : int
        The target sum of the geometric sequence.
    min_a : int, optional
        The minimum value for the first term 'a', by default 10

    Returns
    -------
    List[List[int]]
        A list of geometric sequences that sum to M.
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


# Graphs and trees

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects
import numpy as np

rng = np.random.default_rng(42)


def random_kary_tree(n, k):
    nxG = nx.Graph()
    node_count = 1
    nxG.add_node(r"$\rho$")
    parent_count = 0
    leaf_nodes = [r"$\rho$"]
    while parent_count < n:
        for leaf_node in leaf_nodes:
            child_count = rng.choice(np.append(0, np.arange(2, k + 1)))
            for i in range(child_count):
                node_count = node_count + 1
                nxG.add_node(node_count)
                nxG.add_edge(node_count, leaf_node)
                leaf_nodes.append(node_count)
            if child_count > 0:
                parent_count = parent_count + 1
                leaf_nodes.remove(leaf_node)
            if parent_count == n:
                break
    pos = nx.nx_agraph.graphviz_layout(
        nxG,
        prog="dot",
        args=f"-Groot={1}",
    )
    return nxG, pos, leaf_nodes


import numpy as np


def random_kary_hasvd_tree(n: int, k: int, block_shape: tuple[int, int]) -> hasvd_Node:
    """
    Build a random k-ary HASVD tree (root tag "r", others 1,2,3…) with:
      - exactly n internal nodes (excluding the root),
      - each internal node has between 2 and k children,
      - internal nodes direction ∈ {0,1},
      - leaf nodes direction = 2,
      - mixed leaf+internal children allowed,
      - shapes propagate consistently per direction.
    """
    rng = np.random.default_rng()
    remaining_internals = n
    next_tag = 1

    # Create root
    root_dir = rng.integers(0, 2)
    root = hasvd_Node(shape=block_shape, direction=root_dir, tag="r")
    queue = [(root, root_dir, block_shape)]

    while remaining_internals > 0 and queue:
        parent, p_dir, p_shape = queue.pop(0)

        # choose number of children (2..k)
        child_count = rng.integers(2, k + 1)
        # ensure we don't exceed internals count
        child_count = min(child_count, remaining_internals + (k - 1))
        remaining_internals -= 1  # this node now counted as internal

        # compute child_shape and update parent.shape
        if p_dir == 0:
            # column-split: same rows, split columns
            r, c = p_shape[0], p_shape[1] // child_count
            parent.m, parent.n = r, c * child_count
            child_shape = (r, c)
        else:
            # row-split: split rows, same columns
            r, c = p_shape[0] // child_count, p_shape[1]
            parent.m, parent.n = r * child_count, c
            child_shape = (r, c)

        # decide how many internals among these children
        max_int = min(remaining_internals, child_count - 1)
        num_int = rng.integers(1, max_int + 1) if max_int >= 1 else 0
        num_leaf = child_count - num_int

        # create internal children
        for _ in range(num_int):
            d = rng.integers(0, 2)
            child = parent.add_child(direction=d, shape=child_shape, tag=next_tag)
            queue.append((child, d, child_shape))
            next_tag += 1
            remaining_internals -= 1

        # create leaf children
        for _ in range(num_leaf):
            parent.add_child(direction=2, shape=child_shape, tag=next_tag)
            next_tag += 1

    # any leftover queued nodes become leaf parents
    for node, _, shape in queue:
        leaf_count = rng.integers(2, k + 1)
        for _ in range(leaf_count):
            node.add_child(direction=2, shape=shape, tag=next_tag)
            next_tag += 1

    return root


def graphviz_for_tree(nxG: nx.Graph):
    return nx.nx_agraph.graphviz_layout(
        nxG,
        prog="dot",
        args=f"-Groot={1}",
    )


# Draw graphs


def draw_nxgraph(root: hasvd_Node, node_size=1000):
    nxG = nx.Graph()

    options = {"node_color": "white", "edgecolors": "black", "node_size": node_size}

    optionsRow = {
        "node_color": "tab:red",
        "edgecolors": "black",
        "node_shape": "s",
        "node_size": node_size,
    }

    optionsCol = {
        "node_color": "tab:blue",
        "edgecolors": "black",
        "node_shape": "D",
        "node_size": node_size,
    }

    leafNodes = []
    columnNodes = []
    rowNodes = []

    # Specify the root node
    root_node = root

    # Add edges (example)
    for node in root.traverse():
        nxG.add_node(node)
        if node.parent and node != root:
            nxG.add_edge(node.parent, node)
        match node.direction:
            case 2:
                leafNodes.append(node)
            case 0:
                rowNodes.append(node)
            case 1:
                columnNodes.append(node)

    # Generate the layout using graphviz_layout
    pos = nx.nx_agraph.graphviz_layout(
        nxG,
        prog="dot",
        root=root_node.id,
    )

    # Draw the graph
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.rcParams["text.usetex"] = True
    nx.draw(nxG, pos)
    nx.draw_networkx_nodes(nxG, pos, nodelist=leafNodes, **options)
    nx.draw_networkx_nodes(nxG, pos, nodelist=rowNodes, **optionsRow)
    nx.draw_networkx_nodes(nxG, pos, nodelist=columnNodes, **optionsCol)
    nx.draw_networkx_edges(nxG, pos, edgelist=nxG.edges())
    nx.draw_networkx_labels(
        nxG,
        pos,
        {n: n.tag for n in columnNodes + rowNodes if n in pos},
        font_size=24,
        font_color="white",
    )
    nx.draw_networkx_labels(
        nxG,
        pos,
        {n: n.tag for n in leafNodes if n in pos},
        font_size=24,
        font_color="black",
    )


# Rank plots


def plot_rank_graph(
    root: hasvd_Node, node_rank_map, cmap="RdYlGn_r", bound=(None, None)
):
    G = nx.DiGraph()
    color_vals = []

    G.add_node(root)
    color_vals.append(node_rank_map.get(root, 0))

    for node in root.traverse():
        for child in node.children:
            G.add_node(child)
            G.add_edge(node, child)
            color_vals.append(node_rank_map.get(child, 0))

    # Generate the layout using graphviz_layout
    pos = nx.nx_agraph.graphviz_layout(
        G,
        prog="dot",
        args=f"-Groot={root.tag}",
    )

    # Create figure and axis manually
    fig, ax = plt.subplots(figsize=(10, 6))

    if bound == (None, None):
        vmin, vmax = min(color_vals), max(color_vals)
    else:
        vmin, vmax = bound

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    # Draw nodes and edges
    # Map color values manually to RGBA colors
    node_colors = [cmap_obj(norm(val)) for val in color_vals]

    # Draw nodes with manually mapped colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, node_size=1000)
    nx.draw_networkx_edges(G, pos, ax=ax)

    for node, (x, y) in pos.items():
        val = node_rank_map.get(node, 0)
        label = f"{val}"
        ax.text(
            x,
            y,
            label,
            fontsize=12,
            ha="center",
            va="center",
            color="white",
            path_effects=[
                path_effects.Stroke(linewidth=1.5, foreground="black"),
                path_effects.Normal(),
            ],
        )

    # Add colorbar manually with mappable

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)

    sm.set_array([])  # Required for matplotlib >= 3.1
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Truncated Rank")

    ax.set_title("HASVD Hierarchical Rank Map")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_rank_graph(node, node_rank_map, indent=0):
    rank = node_rank_map.get(node, 0)
    print("    " * indent + f"{node.tag} (Rank: {rank})")
    for child in node.children:
        print_rank_graph(child, node_rank_map, indent + 1)


# Utils counter


def non_leaf_count(tree: hasvd_Node):
    """
    Count the number of non-leaf nodes in a HASVD tree.

    Parameters
    ----------
    tree : hasvd_Node
        The root node of the HASVD tree.

    Returns
    -------
    int
        The number of non-leaf nodes in the tree.
    """
    return sum(1 for node in tree.traverse() if not node.is_leaf)


def branch_node_count(tree: hasvd_Node):
    """
    Count the number of branching nodes in a HASVD tree.

    A branching node is defined as a non-leaf node that has at least one child that is also a non-leaf node.

    Parameters
    ----------
    tree : hasvd_Node
        The root node of the HASVD tree.

    Returns
    -------
    int
        The number of branching nodes in the tree.
    """
    return sum(
        1
        for node in tree.traverse()
        if not node.is_leaf and any(not child.is_leaf for child in node.children)
    )


def assert_shape_consistency(node: hasvd_Node):
    """
    Recursively asserts that the sum of the children's shapes
    matches the parent's shape along the split direction.

    Raises AssertionError if inconsistency is found.
    """
    if node.is_leaf:
        return

    if node.direction == 1:  # row split
        total = sum(child.m for child in node.children)
        assert total == node.m, (
            f"Row split mismatch at node {node.tag}: "
            f"sum of rows {total} != parent rows {node.m}"
        )
        for child in node.children:
            assert child.n == node.n, (
                f"Column size mismatch at node {child.tag}: "
                f"child n {child.n} != parent n {node.n}"
            )

    elif node.direction == 2:  # column split
        total = sum(child.n for child in node.children)
        assert total == node.n, (
            f"Column split mismatch at node {node.tag}: "
            f"sum of columns {total} != parent columns {node.n}"
        )
        for child in node.children:
            assert child.m == node.m, (
                f"Row size mismatch at node {child.tag}: "
                f"child m {child.m} != parent m {node.m}"
            )

    elif node.direction == 0:
        # If it's direction 0 (e.g., root or no split), you can define:
        # Either no children allowed, or no shape constraint.
        pass

    # Recursively check all children
    for child in node.children:
        assert_shape_consistency(child)
