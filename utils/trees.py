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
        direction=direction,
        shape=(num_slices * block_shape[0], block_shape[1]),
    )

    for tag in range(num_slices):
        root.add_child(
            tag=tag,
            direction=2,  # leaf direction (optional, for consistency)
            shape=(block_shape[0], block_shape[1]),
        )

    return root


def linear_general_btl_map(A: np.ndarray, m: int, n: int, direction: int):
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


def linear_hankelarray_btl_map(
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
            direction=(outer_direction + 1) % 2,
            shape=shape,
        )
        outer_tag += 1
        for inner_idx in range(num_inner_slices):
            outer_node.add_child(tag=leaf_tag, shape=block_shape)
            leaf_tag += 1
        if outer_idx < num_outer_slices - 2:
            match outer_direction:
                case 0:
                    shape = (total_m, total_n - (1 + outer_idx) * block_shape[1])
                case 1:
                    shape = (total_m - (1 + outer_idx) * block_shape[0], total_n)
            merge_node = parent_node.add_child(
                tag=outer_tag,
                direction=outer_direction,
                shape=shape,
            )
            outer_tag += 1
            parent_node = merge_node
    return root


def tlbd_general_btl_map(
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


def tlbd_hankelarray_btl_map(
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
        args=f"-Groot={root_node}",
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
        {n: n for n in columnNodes + rowNodes if n in pos},
        font_size=24,
        font_color="white",
    )
    nx.draw_networkx_labels(
        nxG,
        pos,
        {n: n for n in leafNodes if n in pos},
        font_size=24,
        font_color="black",
    )


def plot_rank_graph(root: hasvd_Node, node_rank_map, cmap="RdYlGn_r"):
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

    # Draw nodes and edges
    nx.draw_networkx_nodes(
        G, pos, node_color=color_vals, cmap=plt.get_cmap(cmap), ax=ax, node_size=1000
    )
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
    sm = ScalarMappable(
        norm=Normalize(vmin=min(color_vals), vmax=max(color_vals)), cmap=cmap
    )
    sm.set_array([])  # Required for matplotlib >= 3.1
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Truncated Rank")

    ax.set_title("HASVD Hierarchical Rank Map")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


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
