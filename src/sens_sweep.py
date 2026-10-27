"""PART A: one impairment at a time, swept over magnitude, for each system configuration and 3 device draws.
For each point: rate of model-based (ideal-model) gradient design vs oracle-model design, on the true device, 20 test channels.
Usage: python3 sens_sweep.py K M L"""
import sys, json, numpy as np, torch
torch.set_num_threads(1)
from impair_lib import System, Dev, NONE, design_rate

SWEEP = {
    "psi":      [0.05, 0.1, 0.2, 0.32, 0.5],
    "weak":     [0.95, 0.85, 0.7, 0.5],
    "spacing":  [0.005, 0.01, 0.03, 0.05],
    "offset":   [0.02, 0.05, 0.1, 0.2],
    "csi_db":   [-20, -15, -10, -5],
    "amp_elem": [0.05, 0.1, 0.2],
    "stuck":    [0.01, 0.03, 0.05],
}

def main():
    K, M, L = map(int, sys.argv[1:4])
    S = System(K, M, L)
    print(f"config K={K} M={M} L={L}: geometry check {S.check():.1e}", flush=True)
    H = S.channels([5000, 5001], 20)
    base_ideal = Dev(S, S.nominal())
    r0 = design_rate(base_ideal, base_ideal, H, H).mean().item()
    print(f"no impairment: {r0:.3f}", flush=True)
    res = {"no_impairment": r0}
    for key, vals in SWEEP.items():
        for v in vals:
            ideal, oracle = [], []
            for ds in (1, 2, 3):
                lv = dict(NONE); lv[key] = v
                th, ae, st = S.sample(lv, 100 * ds + 7)
                true = Dev(S, th, ae, st)
                Hh = S.estimate(H, lv["csi_db"], 50 + ds)
                ideal.append(design_rate(Dev(S, S.nominal()), true, H, Hh).mean().item())
                oracle.append(design_rate(true, true, H, H if key == "csi_db" else Hh).mean().item())
            res[f"{key}={v}"] = dict(ideal=ideal, oracle=oracle)
            print(f"{key:9s}={v:<6} ideal-model {np.mean(ideal):.3f}±{np.std(ideal):.3f}  oracle {np.mean(oracle):.3f}  retained {100*np.mean(ideal)/np.mean(oracle):5.1f}%", flush=True)
    json.dump(res, open(f"sens_{K}_{M}_{L}.json", "w"))
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
