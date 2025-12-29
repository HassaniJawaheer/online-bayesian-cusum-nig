import numpy as np
from nig_model import NormalInverseGammaState

def test_reset_initializes_minimal_state(state: NormalInverseGammaState):
    state.reset()
    assert state.n == 0
    assert state.sum_x == 0.0
    assert state.sum_x2 == 0.0

def test_prior_posterior_params_are_valid(state: NormalInverseGammaState):
    state.reset()
    post = state.posterior_params()

    # Champs attendus sur la dataclass: n, mu, kappa, alpha, beta
    assert post.kappa > 0.0
    assert post.alpha > 0.0
    assert post.beta > 0.0

    assert np.isfinite(post.mu)
    assert np.isfinite(post.kappa)
    assert np.isfinite(post.alpha)
    assert np.isfinite(post.beta)

def test_prior_predictive_params_are_valid(state: NormalInverseGammaState):
    state.reset()
    df, loc, scale = state.predictive_params()

    assert df > 0.0
    assert scale > 0.0

    assert np.isfinite(df)
    assert np.isfinite(loc)
    assert np.isfinite(scale)

def test_prior_log_predictive_isfinite_on_resonnable_values(state: NormalInverseGammaState, prior_params):
    """On vérifie que log_predictive est finie sur quelques valeurs "raisonnables"""
    state.reset()
    mu0, _, _, _ = prior_params

    _, _, scale = state.predictive_params()
    x_center = mu0
    x_near = mu0 + 1.0 * scale
    x_far = mu0 + 10.0 * scale

    logp_center = state.log_predictive(x_center)
    logp_near = state.log_predictive(x_near)
    logp_far = state.log_predictive(x_far)

    assert np.isfinite(logp_center)
    assert np.isfinite(logp_near)
    assert np.isfinite(logp_far)

    # Check d'ordre : au centre doit etres plus probable qu'un point trés loin
    assert logp_center > logp_far
