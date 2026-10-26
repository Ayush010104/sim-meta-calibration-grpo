"""Stage 1: reset-initialization rules (meta vs leader vs attn), paired seeds.
Scenario: rho=0.7, v=1.4 m/s, T_s=0.5, N=10, stagger=50, 1000 steps -- the
regime with the most reset events (~1900/run), where the reset rule matters
most. seq-WARM reference taken from the robustness sweep (same seed/channel).
Checkpointed per (variant, seed).
"""
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)

from run_robustness_sweep import generate_channel_window, BASE_CFG, T_S
from ensemble_reset_variants import run_ensemble_variants

RHO, V, N_STEPS = 0.7, 1.4, 1000
SEEDS = [12345, 23456, 34567]
VARIANTS = ["meta", "leader", "attn"]
LOG, OUT = "run_stage1.log", "stage1_results.npy"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    res = np.load(OUT, allow_pickle=True).item() if os.path.exists(OUT) else {}
    if not res:
        open(LOG, "w").close()
    sweep = np.load("robustness_sweep_results.npy", allow_pickle=True).item()
    log(f"STAGE 1: rho={RHO} v={V} steps={N_STEPS} seeds={SEEDS} variants={VARIANTS} [{len(res)} done]")
    for seed in SEEDS:
        H = generate_channel_window(seed, RHO, V, T_S, N_STEPS)
        sw = float(np.mean(sweep[(seed, RHO, V)]["seq_warm"]))
        for var in VARIANTS:
            if (var, seed) in res:
                continue
            t0 = time.time()
            minrs, rl, diag = run_ensemble_variants(H, seed, RHO, 1, BASE_CFG, V, T_S, n_copies=10,
                                                     stagger=50, reset_init=var)
            m = float(np.mean(minrs))
            extra = ""
            if var == "attn" and diag["retrieval_age"]:
                extra = (f"  retrieved-age mean={np.mean(diag['retrieval_age']):.1f} "
                         f"latest-frac={np.mean(diag['retrieved_is_latest']):.2f}")
            res[(var, seed)] = dict(minrs=minrs, reset_log=rl, diag=diag, seq_warm=sw)
            np.save(OUT, np.array(res, dtype=object))
            log(f"[{var:6s} seed={seed}] mean={m:.4f}  {100*m/sw:.1f}% of seq-WARM  resets={len(rl)}  "
                f"{time.time()-t0:.0f}s{extra}")
    log("\nSTAGE 1 COMPLETE")
    for var in VARIANTS:
        p = [100 * np.mean(res[(var, s)]["minrs"]) / res[(var, s)]["seq_warm"] for s in SEEDS]
        log(f"{var:7s} {np.mean(p):6.1f}% +/- {np.std(p):4.1f}   per-seed: {[round(x, 1) for x in p]}")
