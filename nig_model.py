from math import sqrt, isfinite
from scipy.stats import t
from dataclasses import dataclass

@dataclass
class NIGPosteriorParams:
    """Hyperparamètres du posterior NIG (après n observations)."""
    n: int
    mu: float
    alpha: float
    kappa: float
    beta: float

class NormalInverseGammaState:
    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        # hyperparams prior
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # sanity checks
        self._sanity_prior()
        
        self.reset()
    
    def _sanity_prior(self) -> None:
        assert self.kappa0 > 0.0
        assert self.alpha0 > 0.0
        assert self.beta0 > 0.0

    def _sanity_posterior(self, post: NIGPosteriorParams) -> None:
        assert post.kappa > 0.0
        assert post.alpha > 0.0
        assert post.beta > 0.0
    
    def _sanity_predictive(self, df: float, scale: float) -> None:
        assert df > 0.0
        assert scale > 0.0
        assert isfinite(df) and isfinite(scale)

    def reset(self):
        """Remettre l'état comme si on n'avait vu aucune donnée (prior seul)"""
        self.n = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def update(self, x: float):
        """Intégrer une observation dans l'état courant."""
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x**2

    def mean(self) -> float:
        return 0.0 if self.n == 0 else self.sum_x / self.n
    
    def sse(self) -> float:
        """Sum of squared errors autour de la moyenne empirique."""
        if self.n == 0:
            return 0.0
        m = self.mean()
        return self.sum_x2 - self.n * (m * m)
    
    def posterior_params(self):
        """Retourner les paramètres du posterior NIG."""
        n = self.n
        m = self.mean()
        sse = self.sse()

        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * m) / kappa_n
        alpha_n = self.alpha0 + 0.5* n
        beta_n = self.beta0 + 0.5 * sse + (self.kappa0 * n / (2.0 * kappa_n)) * ((m - self.mu0)**2)
        
        post = NIGPosteriorParams(n=n, mu=mu_n, kappa=kappa_n, alpha=alpha_n, beta=beta_n)
        self._sanity_posterior(post)

        return post
    
    def predictive_params(self):
        """Paramètres df/loc/scale de la Student-t prédictive."""
        post = self.posterior_params()
        loc = post.mu
        df = 2 * post.alpha
        scale2 = post.beta * (post.kappa + 1.0) / (post.alpha * post.kappa)
        scale = sqrt(scale2)

        self._sanity_predictive(df, scale)

        return df, loc, scale

    def log_predictive(self, x: float):
        """Log-densité prédictive de x sous l'état courant."""
        df, loc, scale = self.predictive_params()
        logp = float(t.logpdf(x, df=df, loc=loc, scale=scale))
        assert isfinite(logp)
        return logp