"""Block 1 ensemble queue (checkpointed per run). Key = (rho, seed, mode, tau, polish).
P1 aware-vs-blind META (with polish) at rho 0.99 and 0.7, 3 seeds; deployed phases logged for delay analysis.
P2 no-polish (RL-only) META, rho 0.7, seeds 12345/23456, blind+aware.
P3 reset-threshold sweep, AWARE policy, rho 0.99, seeds 12345/23456."""
import os, time
import numpy as np, torch
torch.set_num_threads(1)
from run_robustness_sweep import generate_channel_window, BASE_CFG, T_S
from ensemble_reset_variants import run_ensemble_variants

OUT, LOG, V = "block1_results.npy", "run_block1.log", 1.4
SEEDS = [12345, 23456, 34567]

def queue():
    """TRIMMED (was 24 runs). Order = priority. The first entry is already done (reused from checkpoint)."""
    q = [
        (0.99, 12345, "aware", 0.8, True),    # done
        (0.99, 12345, "blind", 0.8, True),    # blind pair for it
        (0.7, 34567, "aware", 0.8, True),     # third seed at rho=0.7 (blind ref = stage-1 META)
        (0.7, 12345, "blind", 0.8, False),    # RL-only (no polish) attribution
        (0.7, 12345, "aware", 0.8, False),
        (0.99, 23456, "aware", 0.8, True),    # second seed at rho=0.99
        (0.99, 23456, "blind", 0.8, True),
    ]
    for tau in (0.6, 0.7, 0.9, 0.95):         # reset-threshold sweep, single seed, aware policy
        q.append((0.99, 12345, "aware", tau, True))
    return q

def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

def main():
    res = np.load(OUT, allow_pickle=True).item() if os.path.exists(OUT) else {}
    Q = queue(); log(f"BLOCK 1 ensemble queue: {len(Q)} runs, {len(res)} already done")
    cache = {}
    for key in Q:
        if key in res: continue
        rho, seed, mode, tau, polish = key
        if (rho, seed) not in cache: cache = {(rho, seed): generate_channel_window(seed, rho, V, T_S, 1000)}
        t0 = time.time()
        m, rl, d = run_ensemble_variants(cache[(rho, seed)], seed, rho, 1, BASE_CFG, V, T_S, n_copies=10, stagger=50,
                                         reset_init="meta", sim_threshold=tau, log_phi=True, aware=(mode == "aware"), polish=polish)
        bits = (np.round(np.stack(d["phi"]) / np.pi) % 2).astype(np.int8)
        res[key] = dict(minrs=m, bits=bits, resets=len(rl))
        np.save(OUT, np.array(res, dtype=object))
        log(f"[rho={rho} seed={seed} {mode:5s} tau={tau} polish={polish}] mean={np.mean(m):.4f} resets={len(rl)} {time.time()-t0:.0f}s  (done {len(res)}/{len(Q)})")
    log("BLOCK 1 ENSEMBLE DONE")

if __name__ == "__main__":
    main()
