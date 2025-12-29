import numpy as np
import pytest 
from nig_model import NormalInverseGammaState


@pytest.fixture
def prior_params():
    """Prior safe for start"""
    mu0 = 0.0
    kappa0 = 1.0
    alpha0 = 2.0
    beta0 = 1.0
    return (mu0, kappa0, alpha0, beta0)

@pytest.fixture
def state(prior_params):
    """Instancie NIG"""
    mu0, kappa0, alpha0, beta0 = prior_params
    s = NormalInverseGammaState(mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0)
    return s

@pytest.fixture
def rng():
    """RNG déterministe"""
    return np.random.default_rng(seed=123)

@pytest.fixture
def samples(rng, prior_params):
    """Generated a little x vector"""
    mu0, _, _, _ = prior_params
    n = 200
    sigma = 1.0
    return rng.normal(loc=mu0, scale=sigma, size=n).astype(np.float64)

@pytest.fixture
def simul_params():
    mu0 = 0.0
    sigma0 = 1.0
    mu1 = 5.0
    sigma1 = 1.0

    T = 1000
    tau = 400

    tau_min = 200
    tau_max = 800

    seed = 123
    return mu0, sigma0, mu1, sigma1, T, tau, tau_min, tau_max, seed