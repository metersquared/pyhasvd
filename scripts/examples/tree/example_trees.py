import utils

tree1 = utils.inc_hasvd_tree(3)
print(tree1)


tree2 = utils.dist_hasvd_tree(3)
print(tree2)

tree3 = utils.hasvd_Node()

a = tree3.add_child(direction=1, tag="a")
a.add_child(tag="a1")
a.add_child(tag="a2")
a.add_child(direction=1, tag="a3")

b = tree3.add_child(tag="b")
b1 = b.add_child(tag="b1")
b.add_child(tag="b2")
b.add_child(direction=1, tag="b3")

print(tree2.children[1].tag)
print(tree3)
