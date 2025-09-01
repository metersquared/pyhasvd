from hasvd.utils.trees import hasvd_Node
import numpy as np


def naive_error(
    node: hasvd_Node,
    eps_star: float,
    omega: float,
    num_nonleaf_nodes: int,
):
    """
    Calculate the naive error for a node in the HASVD tree.

    Parameters
    ----------
    node : hasvd_Node
        The node for which to calculate the naive error.
    eps_star : float
        The epsilon star value.
    omega : float
        The omega value.
    num_nonleaf_nodes : int
        The number of non-leaf nodes in the tree.

    Returns
    -------
    error : float
        The naive error for the node.
    """
    if node.is_leaf:
        num_neighbor_nodes = 0
        sum_shapes = 0
        for child in node.parent.children:
            if child.is_leaf:
                num_neighbor_nodes += 1
                sum_shapes += child.m * child.n
        return (
            (1 - omega)
            * eps_star
            / np.sqrt(num_neighbor_nodes)
            * ((sum_shapes) / (node.parent.m * node.parent.n))
            * (1 / num_nonleaf_nodes)
        )
    elif node.is_root:
        return omega * eps_star
    else:
        return (
            (1 - omega)
            * eps_star
            * ((node.m * node.n) / (node.parent.m * node.parent.n))
            * (1 / num_nonleaf_nodes)
        )


def total_naive_error(root: hasvd_Node, error_func):
    total_error = 0
    for node in root.traverse():
        if not node.is_leaf:
            nodal_error = error_func(node)
            leaf_nodal_error = 0
            for child in node.children:
                if child.is_leaf:
                    leaf_nodal_error += error_func(child) ** 2
            total_error += nodal_error + np.sqrt(leaf_nodal_error)
    return total_error


def tight_error(
    node: hasvd_Node,
    eps_star: float,
    omega: float,
    num_branching_nodes: int,
):
    """
    Calculate the naive error for a node in the HASVD tree.

    Parameters
    ----------
    node : utils.hasvd_Node
        The node for which to calculate the naive error.
    eps_star : float
        The epsilon star value.
    omega : float
        The omega value.
    num_branching_nodes : int
        The number of branching nodes in the tree : Non-leaf nodes with non-leaf children.

    Returns
    -------
    error : float
        The naive error for the node.
    """
    if node.is_leaf:
        return (
            (1 - omega)
            * eps_star
            * np.sqrt((node.m * node.n) / (node.root.m * node.root.n))
            * (1 / (num_branching_nodes + 1))
        )
    elif node.is_root:
        return omega * eps_star
    else:
        sum_shapes = 0
        for child in node.parent.children:
            if not child.is_leaf:
                sum_shapes += child.m * child.n
        return (
            (1 - omega)
            * eps_star
            * ((node.m * node.n) / (sum_shapes))
            * (1 / (num_branching_nodes + 1))
        )


def total_tight_error(root: hasvd_Node, error_func: callable):
    """Calculate the total tight error for a HASVD tree.

    Parameters
    ----------
    root : utils.hasvd_Node
        The root node of the HASVD tree.
    error_func : callable
        A function that takes a node and returns the error for that node.
    Returns
    -------
    total_error : float
        The total tight error for the HASVD tree.
    """
    total_error = 0
    leaf_total_error = 0
    for node in root.traverse():
        if not node.is_leaf:
            nodal_error = error_func(node)
            total_error += nodal_error
        else:
            leaf_total_error += error_func(node) ** 2
    total_error += np.sqrt(leaf_total_error)
    return total_error
