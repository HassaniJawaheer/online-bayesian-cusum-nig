import pytest
from numpy.testing import assert_allclose
import numpy as np
from nig_model import NormalInverseGammaState

def test_posterior_params_are_finite_and_positive_after_updates(state: NormalInverseGammaState, samples):
    for xi in samples:
        state.update(float(xi))

    post = state.posterior_params()

    assert post.kappa > 0.0 and np.isfinite(post.kappa)
    assert post.alpha > 0.0 and np.isfinite(post.alpha)
    assert post.beta > 0.0 and np.isfinite(post.beta)
    assert np.isfinite(post.mu)

def test_posterior_kappa_alpha_monotonicity(state: NormalInverseGammaState, samples):
    i = 0
    post0 = state.posterior_params()
    assert post0.kappa == pytest.approx(state.kappa0 + i, abs=0.0)
    assert post0.alpha == pytest.approx(state.alpha0 + 0.5 * i, abs=0.0)

    for xi in samples:
        i += 1
        state.update(float(xi))
        post = state.posterior_params()

        assert post.kappa == pytest.approx(state.kappa0 + i, abs=0.0)
        assert post.alpha == pytest.approx(state.alpha0 + 0.5 * i, abs=0.0)


def test_batch_vs_online_posterior_allclose(state: NormalInverseGammaState, samples):
    # Online
    for xi in samples:
        state.update(float(xi))
    post_online = state.posterior_params()

    # Batch stats
    x = samples.astype(np.float64)
    n = int(x.size)
    mean = float(np.mean(x))
    sse = float(np.sum((x - mean) ** 2))

    # "Formules papier"
    kappa_n = state.kappa0 + n
    mu_n = (state.kappa0 * state.mu0 + n * mean) / kappa_n
    alpha_n = state.alpha0 + 0.5 * n
    beta_n = (
        state.beta0
        + 0.5 * sse
        + (state.kappa0 * n / (2.0 * kappa_n)) * ((mean - state.mu0) ** 2)
    )

    # Compare
    assert_allclose(post_online.kappa, kappa_n, rtol=1e-12, atol=1e-12)
    assert_allclose(post_online.mu, mu_n, rtol=1e-12, atol=1e-12)
    assert_allclose(post_online.alpha, alpha_n, rtol=1e-12, atol=1e-12)
    assert_allclose(post_online.beta, beta_n, rtol=1e-10, atol=1e-10)