import numpy as np

from cusum_detector import BayesianCUSUMDetector
from nig_model import NormalInverseGammaState


def _make_detector(prior_params, warmup_samples, threshold_h=25.0, stop_on_alert=True):
    mu0, kappa0, alpha0, beta0 = prior_params

    model0 = NormalInverseGammaState(mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0)
    model1 = NormalInverseGammaState(mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0)

    model0.reset()
    model1.reset()

    warmup_n = min(50, len(warmup_samples))
    for x in warmup_samples[:warmup_n]:
        model0.update(float(x))

    detector = BayesianCUSUMDetector(
        model0=model0,
        model1=model1,
        threshold_h=threshold_h,
        stop_on_alert=stop_on_alert,
        store_logs=False,
    )
    return detector


def test_detects_strong_mean_shift_after_tau(prior_params, rng):
    mu0, _, _, _ = prior_params
    sigma = 1.0

    n1 = 200
    n2 = 200
    tau = n1
    delta = 5.0

    x_pre = rng.normal(loc=mu0, scale=sigma, size=n1).astype(np.float64)
    x_post = rng.normal(loc=mu0 + delta, scale=sigma, size=n2).astype(np.float64)
    x_full = np.concatenate([x_pre, x_post])

    detector = _make_detector(prior_params, warmup_samples=x_pre, threshold_h=25.0, stop_on_alert=True)
    result = detector.run(x_full)

    assert result.alert_index is not None
    assert result.alert_index >= tau
    assert result.alert_index <= tau + 50


def test_detects_strong_variance_increase_after_tau(prior_params, rng):
    mu0, _, _, _ = prior_params

    n1 = 200
    n2 = 200
    tau = n1

    sigma_pre = 1.0
    sigma_post = 3.0 

    x_pre = rng.normal(loc=mu0, scale=sigma_pre, size=n1).astype(np.float64)
    x_post = rng.normal(loc=mu0, scale=sigma_post, size=n2).astype(np.float64)
    x_full = np.concatenate([x_pre, x_post])

    detector = _make_detector(prior_params, warmup_samples=x_pre, threshold_h=25.0, stop_on_alert=True)
    result = detector.run(x_full)

    assert result.alert_index is not None
    assert result.alert_index >= tau
    assert result.alert_index <= tau + 80

