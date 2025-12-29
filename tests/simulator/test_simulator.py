from simulator import GaussianChangepointSimulator
import numpy as np
import pytest


def test_sample_reproducible_with_fixed_seed(simul_params):
    mu0, sigma0, mu1, sigma1, T, _, tau_min, tau_max, seed = simul_params
    simulator = GaussianChangepointSimulator(mu0, sigma0, mu1, sigma1)

    x0, tau0 = simulator.sample_random_tau(T, tau_min, tau_max, seed, return_regime=False)
    x1, tau1 = simulator.sample_random_tau(T, tau_min, tau_max, seed, return_regime=False)

    assert tau0 == tau1
    assert np.array_equal(x0, x1)


def test_sample_shape_and_tau(simul_params):
    mu0, sigma0, mu1, sigma1, T, tau, _, _, seed = simul_params
    simulator = GaussianChangepointSimulator(mu0, sigma0, mu1, sigma1)

    x, tau_return, regime = simulator.sample(T, tau, seed, return_regime=True)

    assert x.shape == (T,)
    assert tau_return == tau
    assert regime.shape == (T,)
    assert np.all(regime[:tau] == 0)
    assert np.all(regime[tau:] == 1)


def test_sigma_positive_assertions(simul_params):
    mu0, _, mu1, _, _, _, _, _, _ = simul_params

    sigma0 = -0.5
    sigma1 = -1.0

    with pytest.raises(AssertionError):
        GaussianChangepointSimulator(mu0, sigma0, mu1, sigma1)


def test_means_are_approximately_correct_statistically(simul_params):
    mu0, sigma0, mu1, sigma1, _, _, _, _, seed = simul_params
    simulator = GaussianChangepointSimulator(mu0, sigma0, mu1, sigma1)

    T = 5000
    tau = 2500
    x, tau_return = simulator.sample(T, tau, seed, return_regime=False)
    assert tau_return == tau

    mean_pre = x[:tau].mean()
    mean_post = x[tau:].mean()

    tol0 = 5 * sigma0 / np.sqrt(tau)
    tol1 = 5 * sigma1 / np.sqrt(T - tau)

    assert abs(mean_pre - mu0) < tol0
    assert abs(mean_post - mu1) < tol1



    
