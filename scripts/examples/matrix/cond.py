import hasvd.utils.matrix as matrix
import numpy as np

rng = np.random.Generator(np.random.MT19937(42))

A = matrix.random_matrix(100, 200, 10, rng, 1e3)


print(np.linalg.cond(A))
