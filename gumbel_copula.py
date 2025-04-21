def gumbel_copula(θ, n_claims=1000, rng=None):
    if θ < 1:
        raise ValueError("Gumbel θ must be ≥ 1")
    rng = np.random.default_rng(rng)

    # Degenerate case: θ = 1
    if θ == 1:
        Z1 = rng.normal(size=n_claims)
        Z2 = rng.normal(size=n_claims)
        return Z1, Z2

    # for θ>1
    # Marshall–Olkin algorithm for Archimedean copulas
    alpha = 1.0 / θ

    # Draw S ~ positive α–stable, α = 1/θ, skewness β = 1
    # S is strictly positive with this parameterisation
    S = levy_stable.rvs(alpha=alpha, beta=1, size=n_claims, random_state=rng)
    E1 = rng.exponential(scale=1.0, size=n_claims)
    E2 = rng.exponential(scale=1.0, size=n_claims)
    
    Z1 = np.exp(-E1 / S)
    Z2 = np.exp(-E2 / S)

    # Ensure rounding doesn't send Z1, Z2 outside (0,1)
    # Use a wider clipping range to avoid numerical issues with the normal inverse CDF
    # Avoid values too close to 0 or 1 which would result in extreme normal values
    epsilon  = np.finfo(float).eps # Avoid boundaries due to imprecision
    Z1 = np.clip(Z1, epsilon, 1 - epsilon)
    Z2 = np.clip(Z2, epsilon, 1 - epsilon)
    # Convert to standard normal marginals
    Z1 = norm.ppf(Z1)
    Z2 = norm.ppf(Z2)
    return Z1, Z2
