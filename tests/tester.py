import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dictionary_learning import KSVDDictionaryLearner
import numpy as np

def test_KSVD_Learner():
    n_features = 10
    n_samples = 1000
    X = np.random.randn(n_features, n_samples) * 0.2 + 5
    learner = KSVDDictionaryLearner(algorithm="ksvd", initialization="k-means", 
                           max_iter=100, max_n_nonzero=10)
    learner.fit(X, n_atoms = 100)

if __name__ == "__main__":
    test_KSVD_Learner()


    