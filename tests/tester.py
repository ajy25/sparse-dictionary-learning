import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import KSVD_Learner
import numpy as np

def test_KSVD_Learner():
    n_features = 10
    n_samples = 1000
    X = np.random.randn(n_features, n_samples) * 0.2 + 5
    learner = KSVD_Learner(algorithm="ksvd", initialization="k-means", 
                           max_iter=100, max_n_nonzero=6)
    learner.fit(X, n_atoms = 60)

if __name__ == "__main__":
    test_KSVD_Learner()