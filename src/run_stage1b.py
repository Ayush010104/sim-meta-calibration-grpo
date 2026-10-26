"""Stage 1b + Phase-0 logging.
 (1) logged META run (rho=0.99, v=1.4, seed 12345): saves deployed phase config per step
 (2) tune soft-attention temperature on a HELD-OUT seed (45678) at rho=0.7, v=1.4
 (3) run the best tau on the 3 paired test seeds; compare with stage-1 META (stage1_results.npy)
Checkpointed per run."""
import os, time
import numpy as np, torch
torch.set_num_threads(1)
from run_robustness_sweep import generate_channel_window, BASE_CFG, T_S
from ensemble_reset_variants import run_ensemble_variants

OUT, LOG = "stage1b_results.npy", "run_stage1b.log"
TEST_SEEDS, TUNE_SEED = [12345, 23456, 34567], 45678
TAUS = [0.03, 0.1, 0.3]

def log(m):
    print(m, flush=True); open(LOG, "a").write(m + "\n")

def run(H, seed, rho, v, **kw):
    t0 = time.time()
    m, rl, d = run_ensemble_variants(H, seed, rho, 1, BASE_CFG, v, T_S, n_copies=10, stagger=50, **kw)
    return float(np.mean(m)), m, rl, d, time.time() - t0

if __name__ == "__main__":
    res = np.load(OUT, allow_pickle=True).item() if os.path.exists(OUT) else {}
    if not res: open(LOG, "w").close()
    log(f"STAGE 1b [{len(res)} done]")

    if "phi_log" not in res:
        H = generate_channel_window(12345, 0.99, 1.4, T_S, 1000)
        mean, m, rl, d, dt = run(H, 12345, 0.99, 1.4, reset_init="meta", log_phi=True)
        res["phi_log"] = dict(phi=np.stack(d["phi"]), minrs=m, mean=mean)
        np.save(OUT, np.array(res, dtype=object)); log(f"[phi_log meta rho=0.99 seed=12345] mean={mean:.4f} {dt:.0f}s")

    H = generate_channel_window(TUNE_SEED, 0.7, 1.4, T_S, 1000)
    for name, kw in [("meta", dict(reset_init="meta"))] + [(f"soft_tau{t}", dict(reset_init="soft", soft_tau=t)) for t in TAUS]:
        key = ("tune", name)
        if key in res: continue
        mean, m, rl, d, dt = run(H, TUNE_SEED, 0.7, 1.4, **kw)
        res[key] = dict(mean=mean, minrs=m, eff_n=float(np.mean(d["soft_eff_n"])) if d["soft_eff_n"] else None)
        np.save(OUT, np.array(res, dtype=object))
        log(f"[tune {name}] mean={mean:.4f} resets={len(rl)} eff_n={res[key]['eff_n']} {dt:.0f}s")

    best_tau = max(TAUS, key=lambda t: res[("tune", f"soft_tau{t}")]["mean"])
    log(f"best tau on held-out seed: {best_tau}")
    res["best_tau"] = best_tau
    s1 = np.load("stage1_results.npy", allow_pickle=True).item()
    for seed in TEST_SEEDS:
        key = ("soft", seed)
        if key in res: continue
        H = generate_channel_window(seed, 0.7, 1.4, T_S, 1000)
        mean, m, rl, d, dt = run(H, seed, 0.7, 1.4, reset_init="soft", soft_tau=best_tau)
        sw = s1[("meta", seed)]["seq_warm"]
        res[key] = dict(minrs=m, seq_warm=sw, mean=mean, eff_n=float(np.mean(d["soft_eff_n"])))
        np.save(OUT, np.array(res, dtype=object))
        meta_pct = 100 * np.mean(s1[("meta", seed)]["minrs"]) / sw
        log(f"[soft seed={seed}] {100*mean/sw:.1f}% of seq-WARM (META {meta_pct:.1f}%)  eff_n={res[key]['eff_n']:.1f}  {dt:.0f}s")
    log("\nSTAGE 1b COMPLETE")
    p = [100 * res[("soft", s)]["mean"] / res[("soft", s)]["seq_warm"] for s in TEST_SEEDS]
    q = [100 * np.mean(s1[("meta", s)]["minrs"]) / s1[("meta", s)]["seq_warm"] for s in TEST_SEEDS]
    log(f"soft(tau={best_tau}) {np.mean(p):.1f}% +/- {np.std(p):.1f} {np.round(p,1).tolist()}   vs META {np.mean(q):.1f}% {np.round(q,1).tolist()}")
