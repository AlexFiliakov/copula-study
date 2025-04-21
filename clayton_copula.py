def clayton_copula(θ, n_claims=1000, rng=None):
    # >0  ⇒ lower-tail positive dependence
    rng = np.random.default_rng(rng)

    # Generate uniform random variables
    U  = rng.uniform(size=n_claims)
    V  = rng.uniform(size=n_claims)

    # Apply the Clayton copula transformation
    Z1 = U # for frequency
    Z2 = (U**(-θ) * (V**(-θ/(θ+1)) - 1) + 1)**(-1/θ) # for severity

    Z1, Z2 = norm.ppf(Z1), norm.ppf(Z2) # Convert to normal marginals
    return Z1, Z2
