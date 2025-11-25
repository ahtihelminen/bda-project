import math
import numpy as np

def log_nb_pmf_scalar(y: int, mu: float, phi: float) -> float:
    """Negative Binomial pmf in log-space, parameterised by mean mu and alpha=phi."""
    y = int(y)
    r = float(phi)
    if r <= 0.0 or mu <= 0.0:
        return -np.inf

    p = r / (r + float(mu))  # success prob
    return (
        math.lgamma(y + r)
        - math.lgamma(r)
        - math.lgamma(y + 1.0)
        + r * math.log(p)
        + y * math.log(1.0 - p)
    )


def posterior_prob_drone(
    y: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    phi0: float,
    phi1: float,
    prior_drone: float,
) -> np.ndarray:
    """
    Compute P(z=1 | y) elementwise using NB likelihoods + Bernoulli prior.

    y, mu0, mu1 must be same shape.
    phi0, phi1, prior_drone are scalars.
    """
    eps = 1e-12
    pi1 = float(np.clip(prior_drone, eps, 1.0 - eps))
    pi0 = 1.0 - pi1

    y_flat = y.astype(np.int64, copy=False).ravel()
    mu0_flat = mu0.ravel()
    mu1_flat = mu1.ravel()

    log_p0 = np.empty_like(mu0_flat, dtype=float)
    log_p1 = np.empty_like(mu1_flat, dtype=float)

    for i, (yy, m0, m1) in enumerate(zip(y_flat, mu0_flat, mu1_flat, strict=False)):
        log_p0[i] = log_nb_pmf_scalar(int(yy), float(m0), phi0) + math.log(pi0)
        log_p1[i] = log_nb_pmf_scalar(int(yy), float(m1), phi1) + math.log(pi1)

    m = np.maximum(log_p0, log_p1)
    log_denom = m + np.log(np.exp(log_p0 - m) + np.exp(log_p1 - m))
    p1 = np.exp(log_p1 - log_denom)
    return p1.reshape(y.shape)
