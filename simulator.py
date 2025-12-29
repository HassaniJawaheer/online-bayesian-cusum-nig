import numpy as np


class GaussianChangepointSimulator:
    def __init__(self, mu0: float, sigma0: float, mu1: float, sigma1: float):
        assert sigma0 > 0.0
        assert sigma1 > 0.0
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.mu1 = mu1
        self.sigma1 = sigma1

    def sample(self, T: int, tau: int, seed: int | None = None, return_regime: bool = False):
        assert T > 0
        assert 0 < tau < T

        rng = np.random.default_rng(seed)

        n_pre = tau
        n_post = T - tau

        x_pre = rng.normal(loc=self.mu0, scale=self.sigma0, size=n_pre).astype(np.float64)
        x_post = rng.normal(loc=self.mu1, scale=self.sigma1, size=n_post).astype(np.float64)
        x = np.concatenate([x_pre, x_post])

        assert x.shape == (T,)

        if return_regime:
            regime = np.concatenate([
                np.zeros(n_pre, dtype=np.int8),
                np.ones(n_post, dtype=np.int8),
            ])
            assert regime.shape == (T,)
            assert np.all(regime[:tau] == 0)
            assert np.all(regime[tau:] == 1)
            return x, tau, regime

        return x, tau

    def sample_random_tau(
        self,
        T: int,
        tau_min: int,
        tau_max: int,
        seed: int | None = None,
        return_regime: bool = False,
    ):
        assert T > 0
        assert 1 <= tau_min <= tau_max <= T - 1

        rng = np.random.default_rng(seed)
        tau = int(rng.integers(low=tau_min, high=tau_max + 1))

        n_pre = tau
        n_post = T - tau

        x_pre = rng.normal(loc=self.mu0, scale=self.sigma0, size=n_pre).astype(np.float64)
        x_post = rng.normal(loc=self.mu1, scale=self.sigma1, size=n_post).astype(np.float64)
        x = np.concatenate([x_pre, x_post])

        assert x.shape == (T,)

        if return_regime:
            regime = np.concatenate([
                np.zeros(n_pre, dtype=np.int8),
                np.ones(n_post, dtype=np.int8),
            ])
            assert regime.shape == (T,)
            assert np.all(regime[:tau] == 0)
            assert np.all(regime[tau:] == 1)
            return x, tau, regime

        return x, tau

