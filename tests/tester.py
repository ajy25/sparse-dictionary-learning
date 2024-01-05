import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import KSVD_Learner
import numpy as np

def test_KSVD_Learner():
    n_features = 10
    n_samples = 1000
    X = np.random.randn(n_features, n_samples) * 5 + 1
    learner = KSVD_Learner(algorithm="exact-ksvd", initialization="random", 
                           max_iter=20, max_n_nonzero=20)
    learner.fit(X, n_atoms = 50)

if __name__ == "__main__":
    test_KSVD_Learner()

