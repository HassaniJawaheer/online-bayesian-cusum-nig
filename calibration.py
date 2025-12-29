from cusum_detector import BayesianCUSUMDetector
from nig_model import NormalInverseGammaState
from simulator import GaussianChangepointSimulator
from config import NIGParams, SIMParams
import numpy as np
from tqdm.auto import tqdm


def detector_factory(
    h: float,
    warmup_samples: np.ndarray,
    store_logs: bool = False,
    stop_on_alert: bool = True,
) -> BayesianCUSUMDetector:
    model0 = NormalInverseGammaState(NIGParams.mu.value, NIGParams.kappa.value, NIGParams.alpha.value, NIGParams.beta.value)
    model1 = NormalInverseGammaState(NIGParams.mu.value, NIGParams.kappa.value, NIGParams.alpha.value, NIGParams.beta.value)

    model0.reset()
    model1.reset()

    # warm-up:
    for x in warmup_samples:
        model0.update(float(x))

    return BayesianCUSUMDetector(
        model0=model0,
        model1=model1,
        threshold_h=h,
        store_logs=store_logs,
        stop_on_alert=stop_on_alert,
    )

def estimate_arl0(h: float, T_max: int, n_runs: int, seed: int, warmup_n: int = 50):
    times = []

    # simulateur nominal: mu1=mu0, sigma1=sigma0
    simulator = GaussianChangepointSimulator(
        mu0=SIMParams.mu0.value,
        sigma0=SIMParams.sigma0.value,
        mu1=SIMParams.mu0.value,
        sigma1=SIMParams.sigma0.value,
    )

    for run in tqdm(range(n_runs), total=n_runs, desc=f"estimate_arl0(h={h:.3g})", leave=False):
        x, _ = simulator.sample_random_tau(
            T=T_max,
            tau_min=int(T_max / 4),
            tau_max=int(3 * T_max / 4),
            seed=seed + run,              
            return_regime=False,
        )

        # warmup
        w = min(warmup_n, len(x) - 1)
        x_warmup = x[:w]
        x_monitor = x[w:]
        T_monitor = len(x_monitor)

        det = detector_factory(
            h,
            warmup_samples=x_warmup,
            store_logs=False,
            stop_on_alert=True,
        )
        res = det.run(x_monitor)

        if res.alert_index is None:
            t = T_monitor
        else:
            t = res.alert_index + 1

        assert np.isfinite(t)
        assert 1 <= t <= T_monitor
        times.append(t)

    times = np.asarray(times, dtype=np.float64)

    arl0_mean = float(times.mean())
    arl0_sem = float(times.std(ddof=1) / np.sqrt(len(times))) if len(times) > 1 else 0.0
    alert_rate = float((times < T_monitor).mean())

    return arl0_mean, arl0_sem, alert_rate

def calibrate_threshold(
    target_arl0: float,
    h_min: float,
    h_max: float,
    T_max: int,
    n_runs: int,
    seed: int,
    tol_arl0: float = 0.1,
    max_iter: int = 20,
    factor: float = 2.0,
):
    # bracket: ARL0(h_min) <= target <= ARL0(h_max)
    arl_min, _, _ = estimate_arl0(h_min, T_max, n_runs, seed)
    arl_max, _, _ = estimate_arl0(h_max, T_max, n_runs, seed)

    expand = 0
    while arl_min > target_arl0 and expand < 100:
        h_min = h_min / factor
        arl_min, _, _ = estimate_arl0(h_min, T_max, n_runs, seed)
        expand += 1

    expand = 0
    while arl_max < target_arl0 and expand < 100:
        h_max = h_max * factor
        arl_max, _, _ = estimate_arl0(h_max, T_max, n_runs, seed)
        expand += 1

    # si le bracketer echoue
    if arl_min > target_arl0 or arl_max < target_arl0:
        raise ValueError(
            "Failed to bracket target ARL0. "
            f"Got ARL0(h_min={h_min})={arl_min:.3f}, ARL0(h_max={h_max})={arl_max:.3f}, "
            f"target={target_arl0:.3f}. "
            "Try increasing T_max, widening initial (h_min,h_max), or checking detector logic."
        )

    # dichotomie
    h_mid = 0.5 * (h_min + h_max)
    arl_mid = None

    for _ in range(max_iter):
        h_mid = 0.5 * (h_min + h_max)
        arl_mid, _, _ = estimate_arl0(h_mid, T_max, n_runs, seed)

        if abs(arl_mid - target_arl0) <= tol_arl0:
            return h_mid, arl_mid

        if arl_mid < target_arl0:
            h_min = h_mid
        else:
            h_max = h_mid

    return h_mid, arl_mid
