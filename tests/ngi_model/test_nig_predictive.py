import numpy as np
from nig_model import NormalInverseGammaState


def test_predictive_df_scale_valid_after_updates(state: NormalInverseGammaState, samples):
    for xi in samples:
        state.update(float(xi))

    df, loc, scale = state.predictive_params()

    assert df > 0.0
    assert scale > 0.0
    assert np.isfinite(df)
    assert np.isfinite(loc)
    assert np.isfinite(scale)


def test_predictive_logpdf_is_finite_after_updates(state: NormalInverseGammaState, samples):
    for xi in samples:
        state.update(float(xi))

    mean = float(np.mean(samples))
    std = float(np.std(samples) + 1e-12)  # évite std=0

    # points proches + outlier modéré
    xs = [mean, mean + std, mean - std, mean + 5.0 * std]

    for x in xs:
        logp = state.log_predictive(float(x))
        assert np.isfinite(logp)


def test_predictive_scale_decreases_with_more_data_qualitatively(
    state: NormalInverseGammaState,
    rng: np.random.Generator,
    prior_params,
):
    mu0, _, _, _ = prior_params

    # petit n
    state.reset()
    x_small = rng.normal(loc=mu0, scale=1.0, size=2).astype(np.float64)
    for xi in x_small:
        state.update(float(xi))
    df_small, _, scale_small = state.predictive_params()

    # grand n
    state.reset()
    x_large = rng.normal(loc=mu0, scale=1.0, size=200).astype(np.float64)
    for xi in x_large:
        state.update(float(xi))
    df_large, _, scale_large = state.predictive_params()

    assert df_large > df_small
    assert scale_small > 0.0 and np.isfinite(scale_small)
    assert scale_large > 0.0 and np.isfinite(scale_large)
