"""Does channel-AWARE policy input help the full ENSEMBLE-META pipeline?
rho=0.7, v=1.4, N=10, 1000 steps, paired seeds. Blind reference = stage-1 META (same code path, blind policy)."""
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)

from run_robustness_sweep import generate_channel_window, BASE_CFG, T_S
from ensemble_reset_variants import run_ensemble_variants

OUT, LOG = "aware_meta_results.npy", "run_aware_meta.log"


def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")


def main():
    res = np.load(OUT, allow_pickle=True).item() if os.path.exists(OUT) else {}
    s1 = np.load("stage1_results.npy", allow_pickle=True).item()
    for seed in (12345, 23456):
        if seed in res:
            continue
        H = generate_channel_window(seed, 0.7, 1.4, T_S, 1000)
        t0 = time.time()
        m, rl, d = run_ensemble_variants(H, seed, 0.7, 1, BASE_CFG, 1.4, T_S, n_copies=10, stagger=50,
                                         reset_init="meta", aware=True)
        res[seed] = dict(minrs=m, reset_log=rl)
        np.save(OUT, np.array(res, dtype=object))
        sw = s1[("meta", seed)]["seq_warm"]
        blind = float(np.mean(s1[("meta", seed)]["minrs"]))
        log(f"[AWARE META seed={seed}] mean={np.mean(m):.4f} ({100*np.mean(m)/sw:.1f}% of seq-WARM) "
            f"vs BLIND META {blind:.4f} ({100*blind/sw:.1f}%)  {time.time()-t0:.0f}s")
    log("DONE")


if __name__ == "__main__":
    main()
