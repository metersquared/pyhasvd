# %%
import hasvd.utils.trees as trees
import matplotlib.pyplot as plt

diagonal_num = 5
tree = trees.hasvd_Node(tag="r", direction=1, shape=(3, 3))
parent = tree
for i in range(1, diagonal_num):
    column_part = parent.add_child(tag="$c_" + str(i) + "$", direction=0, shape=(3, 3))
    column_part.add_child(tag="$d_" + str(i) + "$", shape=(3, 3))
    column_part.add_child(tag="$" + str(i) + "^T$", shape=(3, 3))
    agg_part = parent.add_child(tag="$a_" + str(i) + "$", direction=0, shape=(3, 3))
    agg_part.add_child(tag="$" + str(i) + "$", shape=(3, 3))
    if i == diagonal_num - 1:
        agg_part.add_child(tag="$d_" + str(i + 1) + "$", shape=(3, 3))
    else:
        parent = agg_part.add_child(tag="$r_" + str(i) + "$", direction=1, shape=(3, 3))


trees.draw_nxgraph(tree, node_size=1500)
plt.savefig("transpostional-recur-tree.png", format="png", dpi=600, transparent=True)
# %%
