"""GRPO for 500 inner updates on 100 held-out channels (5 unseen trajectories, rho=.99 v=1.4): cold vs offline-Reptile meta-init,
against the relaxed +-1 gradient baseline (random init, 200 and 500 steps). Rates scored with phases in {0,pi}.
Per update: min-rate of current quantized config (pre-polish), best-so-far; 20-sweep polish at selected updates.
Usage: python3 adapt500.py run [N] | plot"""
import multiprocessing as mp, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
torch.set_num_threads(1)
POL = (0, 10, 50, 100, 200, 300, 400, 500)

def curve(args):
    theta, H, seed, N = args
    from meta_offline import new_agent, ctx_state, quant_cfg, minr, polish
    from grpo_ris import update_quantized_best
    a = new_agent(seed, theta); s = ctx_state(H)
    pre, post = [], {}
    for u in range(N + 1):
        if u > 0:
            update_quantized_best(a, [s] * 8, [H] * 8, n_bits=1, reparam_coef=1.0)
        phi = quant_cfg(a, s)
        pre.append(minr(phi, H))
        if u in POL and u <= N:
            post[u] = polish(phi, H, 20)
    return pre, post

def relax_ref(Hte, N):
    from unrolled_meta import Phys, softmin, L, M
    from diag_relax import rates_x
    phys = Phys(); B = Hte.shape[0]
    w = torch.randn(B, L * M, generator=torch.Generator().manual_seed(1)) * 0.5
    out = []
    for n in range(N + 1):
        with torch.no_grad():
            out.append(rates_x(phys, torch.where(w < 0, -1.0, 1.0), Hte).min(-1).values.numpy())
        if n == N: break
        beta = 1.0 + 4.0 * n / N
        with torch.enable_grad():
            t = w.detach().requires_grad_(True)
            g, = torch.autograd.grad(softmin(rates_x(phys, torch.tanh(beta * t), Hte)).sum(), t)
        w = w + 0.1 * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    return np.array(out)

def run():
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    from meta_offline import make_tasks
    te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:100]
    Hte = torch.tensor(np.stack(te), dtype=torch.complex64)
    res = {"relax200": relax_ref(Hte, 200), "relax500": relax_ref(Hte, N)}
    print("relax: 200-step final %.3f, %d-step final %.3f" % (res["relax200"][-1].mean(), N, res["relax500"][-1].mean()), flush=True)
    theta = torch.load("meta_theta.pt")
    ex = ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn"))
    t0 = time.time()
    for name, th in (("meta", theta), ("cold", None)):
        out = list(ex.map(curve, [(th, H, 777 + i, N) for i, H in enumerate(te)]))
        pre = np.array([o[0] for o in out]).T
        res[name] = dict(pre=pre, post={u: np.array([o[1][u] for o in out]) for u in out[0][1]})
        print(f"{name}: u=50 {pre[50].mean():.3f} u=200 {pre[min(200,N)].mean():.3f} u={N} {pre[N].mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
        np.save("adapt500.npy", np.array(res, dtype=object))
    print("DONE", flush=True)

def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = np.load("adapt500.npy", allow_pickle=True).item()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, c, lab in (("cold", "#7f8c8d", "GRPO cold"), ("meta", "#C0392B", "GRPO offline-Reptile meta-init")):
        p = res[k]["pre"]; n = np.arange(len(p))
        ax[0].plot(n, p.mean(1), color=c, label=lab)
        ax[0].plot(n, np.maximum.accumulate(p, 0).mean(1), color=c, ls=":", label=lab + " (best-so-far)")
    for k, c, lab in (("relax200", "#2980B9", "relaxed gradient, 200 steps"), ("relax500", "#1ABC9C", "relaxed gradient, 500 steps")):
        r = res[k]; ax[0].plot(np.arange(len(r)), r.mean(1), color=c, ls="--", label=lab)
    ax[0].set_xlabel("inner updates / gradient steps"); ax[0].set_ylabel("min-rate, 1-bit {0,pi}, no polish"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    for k, c in (("cold", "#7f8c8d"), ("meta", "#C0392B")):
        us = sorted(res[k]["post"]); ax[1].plot(us, [res[k]["post"][u].mean() for u in us], marker="o", color=c, label=k + " + 20-sweep polish")
    ax[1].axhline(res["relax200"][-1].mean(), color="#2980B9", ls="--", label="relaxed gradient 200 (no polish)")
    ax[1].set_xlabel("GRPO updates"); ax[1].set_ylabel("min-rate after polish"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("paper_figures/Fig15_grpo_500.png", dpi=200)
    for k in ("cold", "meta"):
        p = res[k]["pre"]
        print(k, {u: round(float(p[u].mean()), 3) for u in (0, 10, 50, 100, 200, 300, 400, 500) if u < len(p)},
              "best-so-far@N", round(float(np.maximum.accumulate(p, 0)[-1].mean()), 3))
        print("  polish", {u: round(float(v.mean()), 3) for u, v in res[k]["post"].items()})

if __name__ == "__main__":
    run() if sys.argv[1] == "run" else plot()
