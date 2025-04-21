from scipy.stats import t as t_dist
from scipy.stats import norm

def t_copula(ρ, df, n_claims=1000, rng=None):
    rng = np.random.default_rng(rng)

    # Step 1: Gaussian layer with correlation rho
    z = rng.multivariate_normal(mean=[0, 0],
                                cov=[[1, ρ], [ρ, 1]],
                                size=n_claims)
    # Step 2: Gamma/χ² layer
    g = rng.chisquare(df, size=n_claims)
    t_vals = z / np.sqrt(g[:, None] / df)          # shape (n,2)

    # Step 3: push through univariate t‑CDF → uniforms
    U = t_dist.cdf(t_vals, df)

    Z1 = U[:, 0]
    Z2 = U[:, 1]

    # Step 4:  Convert to standard normal marginals
    Z1 = norm.ppf(Z1)
    Z2 = norm.ppf(Z2)
    return Z1, Z2
