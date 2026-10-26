"""GRPO adaptation trajectories over 250 random held-out channel realizations: COLD (random init) vs OFFLINE-REPTILE META init.
250 realizations = 5 channels from each of 50 UNSEEN trajectories (seeds 6000-6049; training used 1000-1299), rho=0.99, v=1.4.
Per realization and per arm: 50 inner GRPO updates; record min-rate of the current quantized configuration at every update
(pre-polish), best-so-far, and 2-/20-sweep bit-flip polish at selected updates. Same inner-loop settings for both arms.
Usage: python3 adapt250.py run [N] [n_seeds] | plot
"""
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

torch.set_num_threads(1)
POL = (0, 1, 2, 5, 10, 20, 30, 40, 50)


def curve(args):
    theta, H, seed, N = args
    from meta_offline import new_agent, ctx_state, quant_cfg, minr, polish
    from grpo_ris import update_quantized_best
    a = new_agent(seed, theta)
    s = ctx_state(H)
    pre, post2, post20 = [], {}, {}
    for u in range(N + 1):
        if u > 0:
            update_quantized_best(a, [s] * 8, [H] * 8, n_bits=1, reparam_coef=1.0)
        phi = quant_cfg(a, s)
        pre.append(minr(phi, H))
        if u in POL and u <= N:
            post2[u], post20[u] = polish(phi, H, 2), polish(phi, H, 20)
    return pre, post2, post20


def run():
    from meta_offline import make_tasks
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    te = make_tasks([6000 + i for i in range(nseeds)], [(0.99, 1.4)], n_steps=400, stride=80)
    print(f"{len(te)} held-out realizations, {N} inner updates", flush=True)
    theta_meta = torch.load("meta_theta.pt")
    ex = ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn"))
    res, t0 = {}, time.time()
    for name, th in (("cold", None), ("meta", theta_meta)):
        out = list(ex.map(curve, [(th, H, 777 + i, N) for i, H in enumerate(te)]))
        res[name] = dict(pre=np.array([o[0] for o in out]),                     # (R, N+1)
                         post2={u: np.array([o[1][u] for o in out]) for u in out[0][1]},
                         post20={u: np.array([o[2][u] for o in out]) for u in out[0][2]})
        print(f"{name}: pre u=0/10/{N}: {res[name]['pre'][:, 0].mean():.3f}/{res[name]['pre'][:, min(10, N)].mean():.3f}/"
              f"{res[name]['pre'][:, N].mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    np.save("adapt250.npy", np.array(res, dtype=object))
    print("DONE", flush=True)


def boot_ci(d, n=4000, seed=0):
    r = np.random.default_rng(seed)
    m = [d[r.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight"})
    res = np.load("adapt250.npy", allow_pickle=True).item()
    cold, meta = res["cold"], res["meta"]
    R, N1 = cold["pre"].shape
    n = np.arange(N1)
    C, MC = "#7f8c8d", "#C0392B"
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    rng = np.random.default_rng(3)
    for r in rng.choice(R, min(25, R), replace=False):
        a.plot(n, cold["pre"][r], color=C, alpha=0.10, lw=0.7)
        a.plot(n, meta["pre"][r], color=MC, alpha=0.10, lw=0.7)
    for arr, c, lab in ((cold["pre"], C, "cold (random) init"), (meta["pre"], MC, "offline-Reptile meta-init")):
        m, se = arr.mean(0), arr.std(0) / np.sqrt(R)
        lo, hi = np.percentile(arr, [10, 90], axis=0)
        a.fill_between(n, lo, hi, color=c, alpha=0.12)
        a.plot(n, m, color=c, lw=2.2, label=f"{lab} (mean, 10-90% band)")
    a.set_xlabel("inner GRPO updates on the new channel"); a.set_ylabel("min-rate, current 1-bit configuration")
    a.set_title(f"Trajectories over {R} random realizations (25 shown faint)", fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8)

    a = ax[0, 1]
    for arr, c, lab in ((cold["pre"], C, "cold"), (meta["pre"], MC, "meta-init")):
        b = np.maximum.accumulate(arr, axis=1)
        m, se = b.mean(0), b.std(0) / np.sqrt(R)
        a.fill_between(n, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.2)
        a.plot(n, m, color=c, lw=2.0, label=lab)
    a.set_xlabel("inner GRPO updates"); a.set_ylabel("best min-rate so far (no polish)")
    a.set_title("Best-so-far, mean with 95% CI", fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8)

    a = ax[1, 0]
    us = sorted(cold["post20"])
    for key, ls, lab in (("post20", "-", "20-sweep polish"), ("post2", "--", "2-sweep polish")):
        for arr, c, nm in ((cold, C, "cold"), (meta, MC, "meta-init")):
            m = np.array([arr[key][u].mean() for u in us]); se = np.array([arr[key][u].std() / np.sqrt(R) for u in us])
            a.errorbar(us, m, yerr=1.96 * se, color=c, ls=ls, marker="o", ms=4, capsize=2, label=f"{nm}, {lab}")
    a.set_xlabel("inner GRPO updates"); a.set_ylabel("min-rate after polish")
    a.set_title("Same configurations after bit-flip polish (95% CI)", fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=7.5)

    a = ax[1, 1]
    for u, col in [(u, c) for u, c in ((10, "#E67E22"), (50, "#2C3E50")) if u < N1]:
        d = meta["pre"][:, u] - cold["pre"][:, u]
        a.hist(d, bins=30, alpha=0.55, color=col, label=f"u={u}: {100*np.mean(d>0):.0f}% of realizations improved")
    a.axvline(0, color="k", lw=0.8)
    a.set_xlabel("meta-init minus cold, min-rate (pre-polish), per realization"); a.set_ylabel("count")
    a.set_title("Paired per-realization gain", fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("paper_figures/Fig14_meta_vs_cold_250realizations.png")
    print("saved paper_figures/Fig14_meta_vs_cold_250realizations.png\n")
    print(f"{'u':>3s} | {'cold':>7s} {'meta':>7s} {'gain':>8s} {'95% CI':>17s} {'improved':>9s}   (pre-polish, mean over {R} realizations)")
    for u in (0, 1, 2, 5, 10, 20, 30, 40, 50):
        if u < N1:
            d = meta["pre"][:, u] - cold["pre"][:, u]; lo, hi = boot_ci(d)
            print(f"{u:>3d} | {cold['pre'][:, u].mean():7.3f} {meta['pre'][:, u].mean():7.3f} {d.mean():+8.3f} [{lo:+.3f},{hi:+.3f}] {100*np.mean(d>0):8.0f}%")
    for key, lab in (("post2", "2-sweep polish"), ("post20", "20-sweep polish")):
        print(f"\n{lab}:")
        for u in sorted(cold[key]):
            d = meta[key][u] - cold[key][u]; lo, hi = boot_ci(d)
            print(f"{u:>3d} | {cold[key][u].mean():7.3f} {meta[key][u].mean():7.3f} {d.mean():+8.3f} [{lo:+.3f},{hi:+.3f}] {100*np.mean(d>0):8.0f}%")


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else plot()
