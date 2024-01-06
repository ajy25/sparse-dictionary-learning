import numpy as np

# Assuming A is your matrix
A = np.arange(12).reshape((4, 3)).astype(float)

A /= np.linalg.norm(A, axis=0)

print(np.linalg.norm(A, axis=0))
print(A)

