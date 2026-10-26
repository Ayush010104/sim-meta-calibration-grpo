"""Meta-learned unrolled optimizer on the REAL-VALUED +-1 RELAXATION of the 1-bit problem.

1-bit phase  <=>  real coefficient d = +-1.  Relax d = tanh(beta*w); binarize with sign(w).
  w_0       = f_psi(H)                         channel encoder (meta-learned amortized initialization)
  w_{k+1}   = w_k + alpha_k (.) ghat_k         ghat_k = unit-RMS exact gradient of the smooth-min rate at tanh(beta_k w_k)
  deploy      sign(w_K) in {+1,-1}^144
Meta-learned: encoder, per-step/per-layer step sizes alpha_k, per-step anneal beta_k. First-order meta-gradients,
deep supervision over all K stages, trained over a DISTRIBUTION of channels (all rho x v cells), evaluated on HELD-OUT
trajectories and with more steps than trained.  Baseline: the same relaxation from RANDOM init with constant step size.
Usage: python3 unrolled_relax.py smoke | train
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from unrolled_meta import Phys, softmin, feats, L, M, K
from diag_relax import rates_x
from meta_offline import make_tasks, polish as np_polish


def hard_rate(phys, w, H):
    with torch.no_grad():
        return rates_x(phys, torch.where(w < 0, -1.0, 1.0), H).min(-1).values


class RelaxUnrolled(nn.Module):
    def __init__(self, k_train=10, hidden=512, a0=0.1, b0=(1.0, 5.0)):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2 * M * K, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, L * M))
        self.log_alpha = nn.Parameter(torch.full((k_train, L), math.log(a0)))
        self.log_beta = nn.Parameter(torch.log(torch.linspace(b0[0], b0[1], k_train + 1)))
        self.k_train = k_train

    def init_w(self, H):
        return self.enc(feats(H))

    def beta(self, k):
        return self.log_beta.exp()[min(k, self.k_train)]

    def refine(self, w, H, steps, phys):
        outs = [w]
        for k in range(steps):
            with torch.enable_grad():
                t = w.detach().requires_grad_(True)
                g, = torch.autograd.grad(softmin(rates_x(phys, torch.tanh(self.beta(k) * t), H)).sum(), t)
            g = g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
            w = w + self.log_alpha.exp()[min(k, self.k_train - 1)].repeat_interleave(M) * g
            outs.append(w)
        return outs


def plain_relax(phys, w0, H, steps, alpha=0.1, beta_end=5.0):
    w = w0.clone()
    for n in range(steps):
        beta = 1.0 + (beta_end - 1.0) * n / max(steps, 1)
        with torch.enable_grad():
            t = w.detach().requires_grad_(True)
            g, = torch.autograd.grad(softmin(rates_x(phys, torch.tanh(beta * t), H)).sum(), t)
        w = w + alpha * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    return w


def evaluate(model, phys, Hte, tag, Ks=(0, 1, 2, 3, 5, 10, 20, 50), polish_at=(0, 5, 10)):
    B = Hte.shape[0]
    gen = torch.Generator().manual_seed(1)
    w_rand = torch.randn(B, L * M, generator=gen) * 0.5
    with torch.no_grad():
        w_enc = model.init_w(Hte)
    rows = {}
    outs_meta = model.refine(w_enc, Hte, max(Ks), phys)
    outs_rand_learned = model.refine(w_rand, Hte, max(Ks), phys)
    arms = {"META: encoder + learned schedule": lambda k: outs_meta[k].detach(),
            "random init + learned schedule": lambda k: outs_rand_learned[k].detach(),
            "BASELINE random init + const alpha=0.1": lambda k: plain_relax(phys, w_rand, Hte, k).detach()}
    for name, get in arms.items():
        row = {}
        for k in Ks:
            w = get(k)
            row[k] = dict(pre=float(hard_rate(phys, w, Hte).mean()))
            if k in polish_at:
                phi = (w < 0).float().numpy() * np.pi
                row[k]["post2"] = float(np.mean([np_polish(phi[i], Hte[i].numpy().astype(complex), 2) for i in range(B)]))
                row[k]["post20"] = float(np.mean([np_polish(phi[i], Hte[i].numpy().astype(complex), 20) for i in range(B)]))
        rows[name] = row
        print(f"  [{tag}] {name:40s} " + " ".join(f"K={k}:{row[k]['pre']:.3f}" for k in Ks), flush=True)
        print(f"        {'':40s} polish: " + " ".join(f"K={k}: p2={row[k]['post2']:.3f} p20={row[k]['post20']:.3f}" for k in polish_at), flush=True)
    return rows


def main():
    smoke = sys.argv[1] == "smoke"
    phys = Phys()
    CELLS = [(r, v) for r in (0.7, 0.9, 0.99) for v in (0.5, 1.4, 3.0)]
    tr = make_tasks(list(range(1000, 1000 + (12 if smoke else 300))), CELLS)
    te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:100]
    Htr = torch.tensor(np.stack(tr), dtype=torch.complex64)
    Hte = torch.tensor(np.stack(te), dtype=torch.complex64)
    print(f"train channels {len(tr)}, held-out {len(te)}", flush=True)
    model = RelaxUnrolled()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    iters, Bsz = (20, 16) if smoke else (3000, 64)
    wts = torch.arange(1, model.k_train + 2, dtype=torch.float32)
    wts = wts / wts.sum()
    t0 = time.time()
    for it in range(1, iters + 1):
        H = Htr[torch.randint(0, len(tr), (Bsz,))]
        outs = model.refine(model.init_w(H), H, model.k_train, phys)
        loss = -sum(wk * softmin(rates_x(phys, torch.tanh(model.beta(k) * o), H)).mean() for k, (wk, o) in enumerate(zip(wts, outs)))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if it % (10 if smoke else 250) == 0:
            with torch.no_grad():
                o = model.refine(model.init_w(Hte), Hte, 10, phys)
            print(f"iter {it}/{iters} loss={loss.item():.3f}  held-out hard min-rate: K0={hard_rate(phys, o[0], Hte).mean():.3f} "
                  f"K5={hard_rate(phys, o[5], Hte).mean():.3f} K10={hard_rate(phys, o[10], Hte).mean():.3f}  "
                  f"alpha0={model.log_alpha.exp()[0].detach().numpy().round(2).tolist()}  {(time.time()-t0)/60:.1f} min", flush=True)
    torch.save(model.state_dict(), "unrolled_relax.pt")
    res = evaluate(model, phys, Hte, "FINAL held-out")
    np.save("unrolled_relax_results.npy", np.array(res, dtype=object))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
