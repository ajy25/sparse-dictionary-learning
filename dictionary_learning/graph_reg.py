import numpy as np
from scipy import linalg as la
from sklearn.linear_model import orthogonal_mp_gram

def sparse_projection(X_plus_U: np.ndarray, T: int):
    """
    Computes the sparse projection operator T, only the T largest entries of 
    each column of input X + U are kept. 

    @args
    - X_plus_U: np.array ~ (n_atoms, n_samples)
    - T: int << target sparsity, i.e., sparsity constraint
    """
    Z = np.zeros_like(X_plus_U)
    for j in range(Z.shape[1]):
        ind = np.argsort(np.abs(X_plus_U[:, j]))[::-1]
        Z[ind[:T], j] = X_plus_U[ind[:T], j]
    return Z

def grsc_admm(Y: np.ndarray, D: np.ndarray, Lc: np.ndarray, beta: float, 
              T: int, X: np.ndarray = None, iternum: int = 5, 
              rho: float = 1.0):
    """
    Implementation of Algorithm 2 from the paper 
    https://ieeexplore.ieee.org/document/7559727

    @args
    - Y: np.array ~ (n_features, n_samples) << signal set
    - D: np.array ~ (n_features, n_atoms) << dictionary
    - Lc: np.array ~ (n_features, n_features) << manifold graph Laplacian
    - beta: float << regularization coefficient
    - T: int << target sparsity, i.e., sparsity constraint
    - X: np.array ~ (n_atoms, n_samples) << initial sparse coding
    - iternum: int << number of ADMM iterations
    - rho: float << ADMM step size parameter

    @returns
    - X: np.array ~ (n_atoms, n_samples) << sparse coding
    """
    n_samples = Y.shape[1]
    n_atoms = D.shape[1]
    D_gram = D.T @ D
    if X is None:
        X = orthogonal_mp_gram(D_gram, D.T @ Y, n_nonzero_coefs=T)
    Z = X
    U = np.zeros((n_atoms, n_samples))
    for i in range(iternum):
        X = la.solve_sylvester(D_gram + rho * np.eye(n_atoms), beta * Lc, 
                               D.T @ Y + rho * (Z - U))
        Z = sparse_projection(X + U)
        U = U + X - Z
    X = Z
    for j in range(n_samples):
        suppInd = np.where(X[:, j] != 0)[0]
        X[suppInd, j] = la.pinv(D[:, suppInd]) @ Y[:, j]
    return X
        


# def graphDL(Y: np.ndarray, D_0: np.ndarray, L: np.ndarray, T: int = None, 
#                max_iter: int = 10, tol: float = 1e-6, 
#                prune_interval: int = None, verbose: bool = False):
#     """
#     Implementation of Algorithm 1 from the paper 
#     https://ieeexplore.ieee.org/document/7559727

#     @args
#     - Y: np.array ~ (n_features, n_samples) << signal set
#     - D_0: np.array ~ (n_features, n_atoms) << initial dictionary
#     - L: np.array ~ (n_features, n_features) << graph Laplacian
#     - K: int << target sparsity
#     - max_iter: int << maximum number of iterations
#     - tol: float << error tolerance (terminates if error below tol)
#     - prune_interval: int << prunes dictionary every prune_interval iterations
#     - verbose: bool << if True, prints learning progress

#     @returns
#     - D: np.array ~ (n_features, n_atoms) << dictionary
#     - X: np.array ~ (n_atoms, n_samples) << sparse representations matrix
#     """
#     D = D_0
#     n_features, n_atoms = D.shape
#     error_check = np.nan
#     for n in range(max_iter):
#         X = batch_omp(D, Y, K=T, catch_warnings=False)
#         error_check = np.linalg.norm(Y - D @ X)
#         if verbose:
#             print(f"\rgraphDL iteration {n+1} of {max_iter}. " + \
#                   f"Error: {np.round(error_check, 2)}", end="")
#         if error_check < tol:
#             break
#         for j in range(n_atoms):
#             I = X[j, :] > 0
#             if np.sum(I) == 0:
#                 continue
#             D[:, j] = np.zeros(n_features)
#             E = Y[:, I] - D @ X[:, I]
#     if error_check < tol:
#         print(f"\rgraphDL complete. Error of {error_check}.")
#     else:
#         print("\rgraphDL maximum iterations reached. Error of", 
#               f"{error_check}.", 
#               "Please consider reselecting hyperparameters.")
#     return D, X

class GraphRegularizedDictionaryLearner():
    """
    Graph regularized dictionary learning algorithm for learning sparse 
    signal representations.
    """

    def __init__(self, n_atoms: int = 10, max_n_nonzero: int = None,
                 algorithm: str = "ksvd", 
                 initialization: str = "random", max_iter: int = 10, 
                 tol: float = 1e-6, verbose: bool = True):
        """
        Graph regularized dictionary learning algorithm for learning sparse 
        signal representations.

        @args
        - n_atoms: int << desired size of dictionary
        - max_n_nonzero: int << if None, auto; sparsity target
        - algorithm: str << one of {ksvd, approx-ksvd}
        - initialization: str << one of {random, pca, k-means}
        - max_iter: int << maximum number of iterations
        - tol: float << error tolerance for early stopping
        - verbose: bool << if True, prints learning progress

        @returns
        - None
        """
        pass

