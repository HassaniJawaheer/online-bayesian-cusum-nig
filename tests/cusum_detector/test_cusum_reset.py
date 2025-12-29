import numpy as np

from cusum_detector import BayesianCUSUMDetector
from nig_model import NormalInverseGammaState


def _make_detector(prior_params, warmup_samples, threshold_h=1e6):
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
        stop_on_alert=False,   # important: on veut continuer
        store_logs=False,
    )
    return detector


def test_model1_resets_when_score_hits_zero(prior_params, rng):
    mu0, _, _, _ = prior_params

    # Séquence "monte puis redescend"
    x_up = rng.normal(loc=mu0 + 5.0, scale=1.0, size=20).astype(np.float64)
    x_down = rng.normal(loc=mu0, scale=1.0, size=200).astype(np.float64)
    x_full = np.concatenate([x_up, x_down])

    detector = _make_detector(prior_params, warmup_samples=x_down, threshold_h=1e6)

    for x in x_full:
        step_result = detector.step(float(x))
        if step_result.S == 0.0:
            assert detector.model1.n == 0
