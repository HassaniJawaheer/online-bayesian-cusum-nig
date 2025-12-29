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

    return BayesianCUSUMDetector(model0=model0, model1=model1, threshold_h=threshold_h)


def test_run_no_alert_with_large_threshold_on_nominal_short_sequence(prior_params, samples):
    detector = _make_detector(prior_params, warmup_samples=samples, threshold_h=1e6)
    detection_result = detector.run(samples)

    assert detection_result.alert_index is None
    assert detection_result.stopped_early is False
    assert not any(detection_result.alerts)
