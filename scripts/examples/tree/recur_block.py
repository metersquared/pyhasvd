# %%
import hasvd.utils.trees as trees
import matplotlib.pyplot as plt

tree = trees.hasvd_Node(tag="r", direction=1, shape=(3, 3))
first_layer = [None] * 5

for i in range(5):
    first_layer[i] = tree.add_child(
        tag="$p_" + str(i + 1) + "$", direction=0, shape=(3, 3)
    )

first_layer[0].add_child(tag="$\ell_1$", shape=(3, 3))
first_layer[0].add_child(tag="$1_1$", shape=(3, 3))
first_layer[0].add_child(tag="$2_1$", shape=(3, 3))

first_layer[1].add_child(tag="$1_2$", shape=(3, 3))
first_layer[1].add_child(tag="$2_2$", shape=(3, 3))
first_layer[1].add_child(tag="$\ell_2$", shape=(3, 3))

first_layer[2].add_child(tag="$\ell_3$", shape=(3, 3))
first_layer[2].add_child(tag="$2_3$", shape=(3, 3))
first_layer[2].add_child(tag="$3_1$", shape=(3, 3))

first_layer[3].add_child(tag="$2_4$", shape=(3, 3))
first_layer[3].add_child(tag="$3_2$", shape=(3, 3))
first_layer[3].add_child(tag="$\ell_4$", shape=(3, 3))

first_layer[4].add_child(tag="$\ell_5$", shape=(3, 3))
first_layer[4].add_child(tag="$3_3$", shape=(3, 3))
first_layer[4].add_child(tag="$\ell_6$", shape=(3, 3))

trees.draw_nxgraph(tree, node_size=900)
plt.savefig("recur-tree.png", format="png", dpi=600, transparent=True)
# %%
