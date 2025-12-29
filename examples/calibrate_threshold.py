from calibration import calibrate_threshold
from datetime import datetime
from pathlib import Path
import json

# Parameters
target_arl0 = 50
T_max = int(3 * target_arl0)
n_runs = 50            
seed = 0
h_min = 0.5              
h_max = 10.0        
factor = 2.0
tol_arl0 = 2.0
max_iter = 15 

def run():
    date_iso = datetime.now().isoformat(timespec="seconds")
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    h_star, arl0_est = calibrate_threshold(
        target_arl0=target_arl0,
        h_min=h_min,
        h_max=h_max,
        T_max=T_max,
        n_runs=n_runs,
        seed=seed,
        tol_arl0=tol_arl0,
        max_iter=max_iter,
        factor=factor,
    )

    data = {
        "timestamp": date_iso,
        "target_arl0": target_arl0,
        "tol_arlo": tol_arl0,
        "T_max": T_max,
        "n_runs": n_runs,
        "seed": seed,
        "h_min": h_min,
        "h_max": h_max,
        "max_iter": max_iter,
        "factor": factor,
        "h_star": float(h_star),
        "arl0_est": float(arl0_est),
    }

    out_dir = Path("examples/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibrate_{date_tag}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

if __name__ == "__main__":
    run()



    



