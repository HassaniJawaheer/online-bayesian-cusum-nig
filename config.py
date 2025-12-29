from enum import Enum

class NIGParams(Enum):
    mu: float = 0.0
    kappa: float = 1.0
    alpha: float = 2.0
    beta: float = 3.0

class SIMParams(Enum):
    mu0: float = 0.0
    sigma0: float = 1.0
