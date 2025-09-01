# %%
import utils
import matplotlib.pyplot as plt

# %%
tree = utils.inc_hasvd_tree(4, 1, block_shape=(2, 2))
utils.draw_nxgraph(tree)
plt.savefig("inc-tree.png", format="png", dpi=600, transparent=True)
# %%
tree = utils.dist_hasvd_tree(4, 0, block_shape=(2, 2))
utils.draw_nxgraph(tree)
plt.savefig("dist-tree.png", format="png", dpi=600, transparent=True)

# %%
tree = utils.random_kary_hasvd_tree(8, 4, block_shape=(1000, 1000))
utils.draw_nxgraph(tree)
plt.savefig("random-kary.png", format="png", dpi=600, transparent=True)

# %%
