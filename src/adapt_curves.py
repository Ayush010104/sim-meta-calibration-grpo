"""50-step adaptation curves on HELD-OUT channels (100 channels, 5 unseen trajectories, rho=0.99, v=1.4).
Arms (min-rate of the current 1-bit configuration after n inner steps, before any polish; plus 20-sweep polish at marked n):
  GRPO  random init                         (RL baseline)
  GRPO  offline Reptile meta-init           (RL + offline meta-training, meta_theta.pt)
  Relax random init + constant step         (classical relaxed-gradient baseline)
  Relax encoder init + constant step        (value of the meta-learned init alone)
  Relax META (encoder + learned schedule)   (proposed meta-learned unrolled optimizer, unrolled_relax.pt)
Usage: python3 adapt_curves.py run [N] [n_channels] | plot
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

torch.set_num_threads(1)

N_DEFAULT = 50
POL = (0, 5, 10, 20, 30, 50)


def grpo_curve(args):
    theta, H, seed, N = args
    from meta_offline import new_agent, ctx_state, quant_cfg, minr, polish
    from grpo_ris import update_quantized_best
    a = new_agent(seed, theta)
    s = ctx_state(H)
    pre, post = [], {}
    for u in range(N + 1):
        if u > 0:
            update_quantized_best(a, [s] * 8, [H] * 8, n_bits=1, reparam_coef=1.0)
        phi = quant_cfg(a, s)
        pre.append(minr(phi, H))
        if u in POL and u <= N:
            post[u] = polish(phi, H)
    return pre, post


def relax_curves(model, phys, Hte, N):
    from unrolled_relax import hard_rate, plain_relax
    from unrolled_meta import softmin, L, M
    from diag_relax import rates_x
    B = Hte.shape[0]
    gen = torch.Generator().manual_seed(1)
    w_rand = torch.randn(B, L * M, generator=gen) * 0.5
    with torch.no_grad():
        w_enc = model.init_w(Hte)
    out = {}

    def run_plain(w0):
        w, ws = w0.clone(), [w0.clone()]
        for n in range(N):
            beta = 1.0 + 4.0 * n / N
            with torch.enable_grad():
                t = w.detach().requires_grad_(True)
                g, = torch.autograd.grad(softmin(rates_x(phys, torch.tanh(beta * t), Hte)).sum(), t)
            w = w + 0.1 * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
            ws.append(w.clone())
        return ws

    ws_meta = model.refine(w_enc, Hte, N, phys)
    arms = {"Relax random init + const step": run_plain(w_rand),
            "Relax encoder init + const step": run_plain(w_enc),
            "Relax META (encoder + learned schedule)": [w.detach() for w in ws_meta]}
    for name, ws in arms.items():
        pre = np.array([hard_rate(phys, w, Hte).numpy() for w in ws])          # (N+1, B)
        out[name] = pre
    return out, arms


def main():
    mode = sys.argv[1]
    if mode == "run":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT
        nch = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        from meta_offline import make_tasks
        from unrolled_relax import RelaxUnrolled
        from unrolled_meta import Phys
        from meta_offline import polish as np_polish
        te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:nch]
        res, t0 = {}, time.time()
        theta_meta = torch.load("meta_theta.pt")
        ex = ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn"))
        for name, th in (("GRPO random init", None), ("GRPO offline Reptile meta-init", theta_meta)):
            out = list(ex.map(grpo_curve, [(th, H, 777 + i, N) for i, H in enumerate(te)]))
            pre = np.array([o[0] for o in out]).T                                  # (N+1, B)
            post = {u: float(np.mean([o[1][u] for o in out])) for u in out[0][1]}
            res[name] = dict(pre=pre, post=post)
            print(f"{name}: pre n=0/10/50 = {pre[0].mean():.3f}/{pre[min(10, N)].mean():.3f}/{pre[N].mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
        phys = Phys()
        model = RelaxUnrolled()
        if os.path.exists("unrolled_relax.pt"):
            model.load_state_dict(torch.load("unrolled_relax.pt"))
        Hte = torch.tensor(np.stack(te), dtype=torch.complex64)
        curves, arms = relax_curves(model, phys, Hte, N)
        for name, pre in curves.items():
            post = {}
            for u in POL:
                if u <= N:
                    phi = (arms[name][u] < 0).float().numpy() * np.pi
                    post[u] = float(np.mean([np_polish(phi[i], Hte[i].numpy().astype(complex), 20) for i in range(len(te))]))
            res[name] = dict(pre=pre, post=post)
            print(f"{name}: pre n=0/10/50 = {pre[0].mean():.3f}/{pre[min(10, N)].mean():.3f}/{pre[N].mean():.3f}", flush=True)
        # per-step cost (single channel, one core)
        from unrolled_meta import softmin
        from diag_relax import rates_x
        w = torch.randn(1, 144) * 0.5
        t = time.time()
        for _ in range(50):
            with torch.enable_grad():
                tt = w.detach().requires_grad_(True)
                g, = torch.autograd.grad(softmin(rates_x(phys, torch.tanh(2.0 * tt), Hte[:1])).sum(), tt)
            w = w + 0.1 * g
        res["_cost"] = dict(relax_ms_per_step=(time.time() - t) / 50 * 1e3)
        from meta_offline import new_agent, ctx_state
        from grpo_ris import update_quantized_best
        a = new_agent(0)
        s = ctx_state(te[0])
        t = time.time()
        for _ in range(10):
            update_quantized_best(a, [s] * 8, [te[0]] * 8, n_bits=1, reparam_coef=1.0)
        res["_cost"]["grpo_ms_per_update"] = (time.time() - t) / 10 * 1e3
        np.save("adapt_curves.npy", np.array(res, dtype=object))
        print("cost:", res["_cost"], "DONE", flush=True)
    else:
        plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight"})
    res = np.load("adapt_curves.npy", allow_pickle=True).item()
    style = {"GRPO random init": ("#999999", "--", "GRPO, random init"),
             "GRPO offline Reptile meta-init": ("#C0392B", "-", "GRPO, offline-Reptile meta-init"),
             "Relax random init + const step": ("#2980B9", "--", "Relaxed gradient, random init"),
             "Relax encoder init + const step": ("#8E44AD", "-.", "Relaxed gradient, meta-learned init only"),
             "Relax META (encoder + learned schedule)": ("#000000", "-", "Meta-learned unrolled optimizer (proposed)")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    for k, (c, ls, lab) in style.items():
        pre = res[k]["pre"]
        n = np.arange(pre.shape[0])
        m, sd = pre.mean(1), pre.std(1) / np.sqrt(pre.shape[1])
        ax.plot(n, m, color=c, ls=ls, lw=2.0 if "proposed" in lab else 1.5, label=lab)
        ax.fill_between(n, m - 2 * sd, m + 2 * sd, color=c, alpha=0.12)
    ax.set_xlabel("inner steps on a new channel")
    ax.set_ylabel("min-rate of current 1-bit configuration (no polish)")
    ax.set_title("Adaptation on held-out channels, before polish", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    for k, (c, ls, lab) in style.items():
        post = res[k]["post"]
        us = sorted(post)
        ax.plot(us, [post[u] for u in us], color=c, ls=ls, marker="o", ms=4, lw=2.0 if "proposed" in lab else 1.5, label=lab)
    ax.set_xlabel("inner steps on a new channel")
    ax.set_ylabel("min-rate after 20-sweep bit-flip polish")
    ax.set_title("Same configurations after polish", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("paper_figures/Fig13_adaptation_50steps.png")
    print("saved paper_figures/Fig13_adaptation_50steps.png")
    ns = [n for n in (0, 1, 2, 5, 10, 20, 30, 50) if n < res["GRPO random init"]["pre"].shape[0]]
    print(f"{'arm':44s} " + " ".join(f"n={n:>2d}  " for n in ns))
    for k, (_, _, lab) in style.items():
        pre = res[k]["pre"]
        print(f"{lab:44s} " + " ".join(f"{pre[n].mean():.3f}  " for n in ns))
    print("after 20-sweep polish:")
    for k, (_, _, lab) in style.items():
        post = res[k]["post"]
        print(f"{lab:44s} " + " ".join(f"{post[n]:.3f}  " if n in post else "  -    " for n in ns))
    print("cost:", res.get("_cost"))


if __name__ == "__main__":
    main()
