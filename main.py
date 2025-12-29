from nig_model import NormalInverseGammaState
from simulator import GaussianChangepointSimulator
from cusum_detector import BayesianCUSUMDetector
from datetime import datetime
from pathlib import Path
import json
from dataclasses import asdict

# Paramètres
date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path("examples/results")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"detection_{date_tag}.json"

cfg = {
  "simulator": {
    "mu0": 1.0,
    "sigma0": 2.0,
    "delta_mu": None,          
    "mu1": None,            
    "sigma1": 2.0,
    "T": 150,
    "warmup_n": 50,
    "tau_min": 60,
    "tau_max": 120,
    "seed": 42,
    "return_regime": True
  },
  "nig_model0": {
    "mu0": None,               
    "kappa0": 1.0,
    "alpha0": 2.0,
    "beta0": None             
  },
  "nig_model1": {

    "mu0": None,              
    "kappa0": 0.1,
    "alpha0": 2.0,
    "beta0": None          
  },
  "detector": {
    "h_star": 1.6875,
    "store_logs": True,
    "stop_on_alert": False
  }
}

# Remplissage
sigma0 = float(cfg["simulator"]["sigma0"])
cfg["simulator"]["delta_mu"] = 5.0 * sigma0
cfg["simulator"]["mu1"] = float(cfg["simulator"]["mu0"] + cfg["simulator"]["delta_mu"])

# prior variance approx via E[sigma^2] = beta/(alpha-1) (si alpha>1)
alpha0 = float(cfg["nig_model0"]["alpha0"])
beta0 = (alpha0 - 1.0) * (sigma0 ** 2)

cfg["nig_model0"]["mu0"] = float(cfg["simulator"]["mu0"])
cfg["nig_model0"]["beta0"] = float(beta0)

cfg["nig_model1"]["mu0"] = float(cfg["simulator"]["mu0"])
cfg["nig_model1"]["beta0"] = float(beta0)

# Simulator
simulator = GaussianChangepointSimulator(
    mu0=cfg["simulator"]["mu0"],
    sigma0=cfg["simulator"]["sigma0"],
    mu1=cfg["simulator"]["mu1"],
    sigma1=cfg["simulator"]["sigma1"],
)

samples, tau, regime = simulator.sample_random_tau(
    T=cfg["simulator"]["T"],
    tau_min=cfg["simulator"]["tau_min"],
    tau_max=cfg["simulator"]["tau_max"],
    seed=cfg["simulator"]["seed"],
    return_regime=cfg["simulator"]["return_regime"],
)

# M0 et M1
model0 = NormalInverseGammaState(
    mu0=cfg["nig_model0"]["mu0"],
    kappa0=cfg["nig_model0"]["kappa0"],
    alpha0=cfg["nig_model0"]["alpha0"],
    beta0=cfg["nig_model0"]["beta0"],
)

model1 = NormalInverseGammaState(
    mu0=cfg["nig_model1"]["mu0"],
    kappa0=cfg["nig_model1"]["kappa0"],
    alpha0=cfg["nig_model1"]["alpha0"],
    beta0=cfg["nig_model1"]["beta0"],
)

# Warm-up de M0, puis on fige M0
warmup_n = int(cfg["simulator"]["warmup_n"])
warmup_n = min(warmup_n, len(samples) - 1)

model0.reset()
for x in samples[:warmup_n]:
    model0.update(float(x))

model1.reset()

# Detector
detector = BayesianCUSUMDetector(
    model0=model0,
    model1=model1,
    threshold_h=cfg["detector"]["h_star"],
    store_logs=cfg["detector"]["store_logs"],
    stop_on_alert=cfg["detector"]["stop_on_alert"],
)

if __name__ == "__main__":
    x_monitor = samples[warmup_n:]
    result = detector.run(samples=x_monitor)

    first_alert_local = next((i for i, a in enumerate(result.alerts) if a), None)
    first_alert_global = (first_alert_local + warmup_n) if first_alert_local is not None else None

    cfg["tau"] = int(tau)
    cfg["warmup_n"] = warmup_n
    cfg["tau_in_monitor"] = int(tau - warmup_n)
    cfg["first_alert_index_local"] = first_alert_local
    cfg["first_alert_index_global"] = first_alert_global
    cfg["result"] = asdict(result)
    cfg["regime"] = regime.tolist()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

