from math import isfinite

from cusum_detector import BayesianCUSUMDetector
from nig_model import NormalInverseGammaState


def _make_detector(prior_params, samples, threshold_h=1e6):
    mu0, kappa0, alpha0, beta0 = prior_params

    model0 = NormalInverseGammaState(mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0)
    model1 = NormalInverseGammaState(mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0)

    # initialiser l’état interne
    model0.reset()
    model1.reset()

    # warm-up de M0 sur quelques points
    warmup_n = min(50, len(samples))
    for x in samples[:warmup_n]:
        model0.update(float(x))

    detector = BayesianCUSUMDetector(model0=model0, model1=model1, threshold_h=threshold_h)
    return detector


def test_step_invariant_score_non_negative(prior_params, samples):
    detector = _make_detector(prior_params, samples)

    for x in samples:
        step_result = detector.step(float(x))
        assert step_result.S >= 0.0
        assert isfinite(step_result.S)


def test_step_returns_finite_increment_and_logs(prior_params, samples):
    detector = _make_detector(prior_params, samples)

    for x in samples:
        step_result = detector.step(float(x))
        assert isfinite(step_result.s)
        assert isfinite(step_result.log[0])
        assert isfinite(step_result.log[1])
