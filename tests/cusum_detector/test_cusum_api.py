from cusum_detector import BayesianCUSUMDetector
from nig_model import NormalInverseGammaState


def _make_detector(prior_params, warmup_samples, threshold_h=1e6, stop_on_alert=False, store_logs=True):
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
        store_logs=store_logs,
    )
    return detector


def test_run_returns_lists_same_length_as_input_when_no_stop(prior_params, samples):
    detector = _make_detector(
        prior_params,
        warmup_samples=samples,
        threshold_h=1e6,
        stop_on_alert=False,
        store_logs=True,
    )
    result = detector.run(samples)

    assert len(result.scores) == len(samples)
    assert len(result.increments) == len(samples)
    assert len(result.alerts) == len(samples)
    assert result.logs is not None
    assert len(result.logs) == len(samples)
