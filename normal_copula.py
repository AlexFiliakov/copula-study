import numpy as np

def normal_copula(correlation, n_claims=1000, rng=None):
    rng = np.random.default_rng(rng)  # Use the provided random number generator or create a new one
    # Create a correlation matrix of size (n_claims, n_claims)
    corr_matrix = np.full((2, 2), correlation)
    np.fill_diagonal(corr_matrix, 1)

    # Sample from a multivariate normal with the above correlation structure.
    z = np.random.multivariate_normal(mean=np.zeros(2), cov=corr_matrix, size=n_claims)
    return z[:,0], z[:,1]
