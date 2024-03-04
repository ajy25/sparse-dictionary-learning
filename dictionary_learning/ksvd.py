import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram, orthogonal_mp
import warnings

def prune_dictionary(D: np.ndarray, threshold: float = 0.8, 
                     verbose: bool = False):
    """
    Prunes the dictionary, removing too-close atoms from the dictionary 
    based on their absolute inner product exceeding some threshold. Replaces 
    removed atoms randomly. 

    @args
    - D: np.array ~ (n_features, n_atoms) << dictionary
    - threshold: float
    - verbose: bool << if True, prints progress

    @returns 
    - D: np.array ~ (n_features, n_atoms) << dictionary
    """
    n_features = D.shape[0]
    inner_products = np.abs(D.T @ D)
    np.fill_diagonal(inner_products, 0)
    close_elements_idx = np.unique(np.where(inner_products > threshold)[0])
    n_to_replace = len(close_elements_idx)
    if n_to_replace == 0:
        Dnew = np.random.uniform(size=D.shape)
        return Dnew / np.linalg.norm(Dnew, axis=0)
    if verbose:
        print(f"{n_to_replace} similar atoms identified at indices",
              f"{close_elements_idx}.")
    Dj = np.random.uniform(size=(n_features, n_to_replace))
    D[:, close_elements_idx] = Dj
    D = D / np.linalg.norm(D, axis=0)
    return D

def batch_omp(D: np.ndarray, X: np.ndarray, method: str = "gram", 
              K: int = None, catch_warnings: bool = True):
    """
    Solves the sparsity-constrained sparse coding problem

    @args
    - D: np.array ~ (n_features, n_atoms) << dictionary estimate
    - X: np.array ~ (n_features, n_samples) << signal set
    - method: str << if gram, uses only the Gram matrix for OMP
    - K: int << sparsity target; if None, set to 10% of K
    - catch_warning: bool << if True, returns None if Warning

    @returns
    - Gamma: np.array ~ (n_atoms, n_samples)
    """
    assert np.sum(np.isnan(D)) == 0
    if method == "gram":
        with warnings.catch_warnings(record=True) as warning_list:
            Gamma = orthogonal_mp_gram(D.T @ D, D.T @ X, 
                n_nonzero_coefs=K)
            if warning_list and catch_warnings:
                return None
    else:
        with warnings.catch_warnings(record=True) as warning_list:
            Gamma = orthogonal_mp(D, X, n_nonzero_coefs=K)
            if warning_list and catch_warnings:
                return None
    assert np.sum(np.isnan(Gamma)) == 0
    return Gamma

def approximate_ksvd(X: np.ndarray, D_0: np.ndarray, K: int = None, 
               max_iter: int = 10, tol: float = 1e-6, 
               prune_interval: int = None, verbose: bool = False):
    """
    Implementation of Algorithm 5 from the paper 
    https://csaws.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    @args
    - X: np.array ~ (n_features, n_samples) << signal set
    - D_0: np.array ~ (n_features, n_atoms) << initial dictionary
    - K: int << target sparsity
    - max_iter: int << maximum number of iterations
    - tol: float << error tolerance (terminates if error below tol)
    - prune_interval: int << prunes dictionary every prune_interval iterations
    - verbose: bool << if True, prints learning progress

    @returns
    - D: np.array ~ (n_features, n_atoms) << dictionary
    - Gamma: np.array ~ (n_atoms, n_samples) << sparse representations matrix
    """
    D = D_0
    n_features, n_atoms = D.shape
    error_check = np.nan
    for n in range(max_iter):
        if prune_interval is not None:
            if (n+1) % prune_interval == 0:
                D = prune_dictionary(D, verbose=True)
        Gamma = batch_omp(D, X, K=K, catch_warnings=True)
        if Gamma is None:
            print("\rWarning in OMP. Attempting to prune dictionary.")
            D = prune_dictionary(D, verbose=verbose)
            Gamma = batch_omp(D, X, K=K, catch_warnings=False)
        error_check = np.linalg.norm(X - D @ Gamma)
        if verbose:
            print(f"\rApproximate K-SVD iteration {n+1} of {max_iter}. " + \
                  f"Error: {np.round(error_check, 2)}", end="")
        if error_check < tol:
            break
        for j in range(n_atoms):
            I = Gamma[j, :] > 0
            if np.sum(I) == 0:
                continue
            D[:, j] = np.zeros(n_features)
            g = Gamma[j, I].T
            d = X[:, I] @ g - D @ Gamma[:, I] @ g
            d = d / np.linalg.norm(d)
            g = X[:, I].T @ d - (D @ Gamma[:, I]).T @ d
            D[:, j] = d 
            Gamma[j, I] = g.T
    if error_check < tol:
        print(f"\rApproximate K-SVD complete. Error of {error_check}.")
    else:
        print("\rApproximate K-SVD maximum iterations reached. Error of", 
              f"{error_check}.", 
              "Please consider reselecting hyperparameters.")
    return D, Gamma

def ksvd(X: np.ndarray, D_0: np.ndarray, K: int = None, max_iter: int = 10, 
         tol: float = 1e-6, prune_interval: int = None, verbose: bool = False):
    """
    Implementation of Algorithm 4 from the paper 
    https://csaws.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    @args
    - X: np.array ~ (n_features, n_samples) << signal set
    - D_0: np.array ~ (n_features, n_atoms) << initial dictionary
    - K: int << target sparsity
    - max_iter: int << maximum number of iterations
    - tol: float << error tolerance (terminates if error below tol)
    - prune_interval: int << prunes dictionary every prune_interval iterations
    - verbose: bool << if True, prints learning progress

    @returns
    - D: np.array ~ (n_features, n_atoms) << dictionary
    - Gamma: np.array ~ (n_atoms, n_samples) << sparse representations matrix
    """
    D = D_0
    n_features, n_atoms = D.shape
    error_check = np.nan
    for n in range(max_iter):
        if prune_interval is not None:
            if (n+1) % prune_interval == 0:
                D = prune_dictionary(D, verbose=False)
        Gamma = batch_omp(D, X, K=K, catch_warnings=True)
        if Gamma is None:
            print("\rWarning in Orthogonal Matching Pursuit.",
                  "Attempting to prune dictionary.")
            D = prune_dictionary(D, verbose=verbose)
            Gamma = batch_omp(D, X, K=K, catch_warnings=False)   
        error_check = np.linalg.norm(X - D @ Gamma)
        if verbose:
            print(f"\rK-SVD iteration {n+1} of {max_iter}. " + \
                  f"Error: {np.round(error_check, 2)}", end="")
        if error_check < tol:
            break
        for j in range(n_atoms):
            I = Gamma[j, :] > 0
            if np.sum(I) == 0:
                continue
            D[:, j] = np.zeros(n_features)
            E = X[:, I] - D @ Gamma[:, I]
            U, _ , Vt = np.linalg.svd(E, full_matrices=True) # U, Vt orthonormal
            D[:, j] = U[:, 0]
            Gamma[j, I] = Vt[:, 0].T
    if error_check < tol:
        print(f"\rK-SVD complete. Error of {error_check}.")
    else:
        print(" " * 20, end = "")
        print("\rK-SVD maximum iterations reached. Error of", 
              f"{error_check}.", 
              "Please consider reselecting hyperparameters.")
    return D, Gamma

class KSVDDictionaryLearner():
    """
    K-SVD algorithm for learning sparse signal representations.
    """

    def __init__(self, n_atoms: int = 10, max_n_nonzero: int = None,
                 algorithm: str = "ksvd", 
                 initialization: str = "random", max_iter: int = 10, 
                 tol: float = 1e-6, verbose: bool = True):
        """
        K-SVD algorithm for learning sparse signal representations.

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
        self.n_atoms = n_atoms
        self.max_n_nonzero = max_n_nonzero
        self.algorithm = algorithm
        self.initialization = initialization
        self.max_iter = max_iter
        self.tol = tol
        self.X = None
        self.D = None
        self.Gamma = None
        self.verbose = verbose

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
        - algorithm: str << one of {ksvd, approx-ksvd}
        - initialization: str << one of {random, pca, k-means}
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

        self._initialize_D()

        if self.algorithm == "ksvd":
            self.D, self.Gamma = ksvd(
                X = self.X,
                D_0 = self.D,
                K = self.max_n_nonzero,
                max_iter = self.max_iter,
                tol = self.tol,
                verbose = self.verbose
            ) 
        elif self.algorithm == "approx-ksvd":
            self.D, self.Gamma = approximate_ksvd(
                X = self.X,
                D_0 = self.D,
                K = self.max_n_nonzero,
                max_iter = self.max_iter,
                tol = self.tol,
                verbose = self.verbose
            )

    def transform(self, X: np.ndarray):
        """
        @args
        - X: np.array ~ (n_features, n_samples)

        @returns
        - Gamma: np.array ~ (n_atoms, n_samples) << sparse representation matrix
        """
        return batch_omp(self.D, X, self.max_n_nonzero)

    def _initialize_D(self):
        """
        Resets D with known X.

        @args
        - verbose: bool << if True, prints updates
        """
        if self.X is None:
            raise ValueError("X has not yet been assigned a value.")
        if self.verbose:
            print(f"Initializing D using method '{self.initialization}'.")
        n_features, n_samples = self.X.shape
        if self.initialization == "random":
            self.D = np.random.randn(n_features, self.n_atoms)
        elif self.initialization == "pca":
            pca = PCA()
            pca.fit(self.X)
            prncpl_components = pca.components_
            contribution_per_sample = np.linalg.norm(prncpl_components, axis=0)
            top_sample_indices = np.argsort(contribution_per_sample)\
                [-self.n_atoms:][::-1]
            self.D = self.X[:, top_sample_indices]
        elif self.initialization == "k-means":
            kmeans = KMeans(n_clusters=self.n_atoms, n_init='auto')
            kmeans.fit(self.X.T)
            self.D = kmeans.cluster_centers_.T
        assert self.D.shape[0] == n_features
        assert self.D.shape[1] == self.n_atoms
        self.D /= np.linalg.norm(self.D, axis=0)
        



        



