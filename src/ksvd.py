import numpy as np
from sklearn.linear_model import orthogonal_mp_gram, orthogonal_mp
from sklearn.cluster import KMeans

def batch_omp(D: np.ndarray, X: np.ndarray, method: str = "gram", 
              K: int = None):
    """
    Solves the sparsity-constrained sparse coding problem

    @args
    - D: np.array ~ (n_features, n_atoms) << dictionary estimate
    - X: np.array ~ (n_features, n_samples) << sparse representations matrix
    - method: str << if gram, uses only the Gram matrix for OMP
    - K: int << sparsity target; if None, set to 10% of K

    @returns
    - Gamma: np.array ~ (n_atoms, n_samples)
    """
    if method == "gram":
        return orthogonal_mp_gram(D.T @ D, D.T @ X, 
            n_nonzero_coefs=K)
    else:
        return orthogonal_mp(D, X, n_nonzero_coefs=K)

def approximate_ksvd():
    pass

def exact_ksvd(X: np.ndarray, D_0: np.ndarray, K: int = None, 
               max_iter: int = 10, tol: float = 1e-6, verbose: bool = False):
    """
    Implementation of Algorithm 4 from the paper 
    https://csaws.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    @args
    - X: np.array ~ (n_features, n_samples) << signal set
    - D_0: np.array ~ (n_features, n_atoms) << initial dictionary
    - K: int << target sparsity
    - max_iter: int << maximum number of iterations
    - tol: float << error tolerance (terminates if error below tol)
    - verbose: bool << if True, prints learning progress

    @returns
    - D: np.array ~ (n_features, n_atoms) << dictionary
    - Gamma: np.array ~ (n_atoms, n_samples) << sparse representations matrix
    """
    D = D_0
    n_features, n_atoms = D.shape
    error_check = np.nan
    for n in range(max_iter):
        Gamma = batch_omp(D, X, K=K)
        error_check = np.linalg.norm(X - D @ Gamma)
        if verbose:
            print(f"\rExact K-SVD iteration {n+1} of {max_iter}. " + \
                  f"Error: {np.round(error_check, 2)}", end="")
        if error_check < tol:
            break
        for j in range(n_atoms):
            D[:, j] = np.zeros(n_features)
            I = Gamma[j, :] > 0
            if np.sum(I) == 0:
                continue
            E = X[:, I] - D @ Gamma[:, I]
            U, _ , Vt = np.linalg.svd(E, full_matrices=True) # U, Vt orthonormal
            D[:, j] = U[:, 0]
            Gamma[j, I] = Vt[:, 0].T
    print(f"\rExact K-SVD complete. Error of {error_check}.")
    return D, Gamma

class KSVD_Learner():
    """
    K-SVD algorithm for learning sparse signal representations.
    """

    def __init__(self, n_atoms: int = 10, max_n_nonzero: int = None,
                 algorithm: str = "exact-ksvd", 
                 initialization: str = "random", max_iter: int = 10, 
                 tol: float = 1e-6):
        """
        K-SVD algorithm for learning sparse signal representations.

        @args
        - n_atoms: int << desired size of dictionary
        - max_n_nonzero: int << if None, auto; sparsity target
        - algorithm: str << one of {exact-ksvd, approx-ksvd}
        - initialization: str << one of {random, pca}
        - max_iter: int << maximum number of iterations
        - tol: float << error tolerance for early stopping

        @returns
        - None
        """
        self.n_atoms = n_atoms
        self.max_n_nonzero = max_n_nonzero
        self.algorithm = algorithm
        self.initialization = initialization
        self.max_iter = max_iter
        self.tol = tol
        self.X = None
        self.D = None
        self.Gamma = None

    def fit(self, X: np.ndarray, n_atoms: int = None, 
            max_n_nonzero: int = None, algorithm: str = None, 
            initialization: str = None, max_iter: int = None, 
            tol: float = None):
        """
        Learns the sparse representation of X.

        @args
        - X: np.array ~ (n_features, n_samples)
        - n_atoms: int << desired size of dictionary
        - max_n_nonzero: int << if None, auto; sparsity target
        - algorithm: str << one of {exact-ksvd, approx-ksvd}
        - initialization: str << one of {random, pca}
        - max_iter: int << maximum number of iterations
        - tol: float << error tolerance for early stopping

        @returns
        - None
        """
        self.X = X

        if n_atoms is not None:
            self.n_atoms = n_atoms
        if max_n_nonzero is not None:
            self.max_n_nonzero = max_n_nonzero
        if algorithm is not None:
            self.algorithm = algorithm
        if initialization is not None:
            self.initialization = initialization
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol

        self._initialize_D(verbose = True)

        if self.algorithm == "exact-ksvd":
            self.D, self.Gamma = exact_ksvd(
                X = self.X,
                D_0 = self.D,
                K = self.max_n_nonzero,
                max_iter = self.max_iter,
                tol = self.tol,
                verbose = True
            ) 
        elif self.algorithm == "approx-ksvd":
            self.D, self.Gamma = approximate_ksvd(
                X = self.X,
                D_0 = self.D,
                K = self.max_n_nonzero,
                max_iter = self.max_iter,
                tol = self.tol,
                verbose = True
            )

    def transform(self, X: np.ndarray):
        """
        @args
        - X: np.array ~ (n_features, n_samples)

        @returns
        - Gamma: np.array ~ (n_atoms, n_samples) << sparse representation matrix
        """
        return batch_omp(self.D, X, self.max_n_nonzero)

    def _initialize_D(self, verbose: bool = False):
        """
        Resets D with known X.

        @args
        - verbose: bool << if True, prints updates
        """
        if self.X is None:
            raise ValueError("X has not yet been assigned a value.")
        if verbose:
            print(f"Initializing D using method '{self.initialization}'.")
        n_features = self.X.shape[0]
        if self.initialization == "random":
            self.D = np.random.randn(n_features, self.n_atoms)
        elif self.initialization == "pca":
            if self.n_atoms> n_features:
                self.initialization = "random"
                self._initialize_D(verbose)
                return 
            U, _, _ = np.linalg.svd(self.X, full_matrices=True)
            self.D = U[:, :self.n_atoms]
        elif self.initialization == "k-means":
            kmeans = KMeans(n_clusters=self.n_atoms, n_init='auto')
            kmeans.fit(self.X.T)
            self.D = kmeans.cluster_centers_.T
        self.D /= np.linalg.norm(self.D, axis=0)



        



