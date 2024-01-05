import numpy as np

# Assuming A is your matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, S, Vt = np.linalg.svd(A)

print(A)

print(U, Vt)
print(U @ np.diag(S) @ Vt)

U_normalized = U / np.linalg.norm(U, axis=0)
print("U norm:", np.linalg.norm(U, axis=0))
Vt_normalized = Vt * np.linalg.norm(U, axis=0)


print(U_normalized, Vt_normalized)
print(U_normalized @ np.diag(S) @ Vt)

