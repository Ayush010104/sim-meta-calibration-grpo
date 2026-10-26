"""Meta-learned UNROLLED optimizer for 1-bit SIM max-min rate (optimization-based meta-learning).

  theta_0 = f_w(H)                                   amortized initialization (channel encoder, meta-learned)
  theta_{k+1} = theta_k + alpha_k (.) ghat_k          K unrolled steps; ghat_k = unit-RMS exact physical gradient of the
                                                     smooth-min rate of the STE-quantized configuration; alpha_k learned per step & per layer
  deploy      q(theta_K) in {0, pi}^144

Meta-training (outer loop) over a DISTRIBUTION of channels (all rho x v cells, many trajectories), first-order
(the inner physical gradients are treated as constants), deep supervision over all K stages. Evaluation on HELD-OUT
trajectories, also with MORE gradient steps than trained (learned step sizes extrapolate).

Ablations in the same script: encoder-only (K=0), random init + learned alpha (no encoder), encoder + constant alpha
(no learned step sizes).
Usage: python3 unrolled_meta.py smoke | train
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from meta_offline import make_tasks, env as get_env, minr as np_minr, polish as np_polish, SCALE
from grpo_ris import quantize_phase_ste, quantize_phase_hard_torch

L, M, K = 4, 36, 5
TAU = 8.0


class Phys:
    def __init__(self):
        e = get_env()
        self.Ws = [torch.tensor(w, dtype=torch.complex64) for w in e.Ws]
        self.P = torch.tensor(e.P, dtype=torch.float32)
        self.var = float(e.var)

    def rates(self, theta, H):
        B = theta.shape[0]
        d = [torch.polar(torch.ones_like(theta[:, l * M:(l + 1) * M]), theta[:, l * M:(l + 1) * M]) for l in range(L)]
        Gv = torch.eye(M, dtype=torch.complex64).unsqueeze(0).expand(B, M, M)
        for idx in range(L - 1, 0, -1):
            Gv = torch.matmul(Gv, d[idx].unsqueeze(-1) * self.Ws[idx].unsqueeze(0))
        Gv = Gv * d[0].unsqueeze(1)
        Full = torch.matmul(torch.matmul(H.conj().transpose(1, 2), Gv), self.Ws[0])
        power = Full.abs() ** 2 * self.P.view(1, 1, K)
        sig = torch.diagonal(power, dim1=-2, dim2=-1)
        return torch.log2(1 + sig / (power.sum(-1) - sig + self.var))


def softmin(r):
    return -(1.0 / TAU) * torch.logsumexp(-TAU * r, dim=-1)


def feats(H):
    return torch.cat([H.real.flatten(1), H.imag.flatten(1)], dim=1) * SCALE


class Unrolled(nn.Module):
    def __init__(self, k_train=3, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2 * M * K, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, L * M))
        self.log_alpha = nn.Parameter(torch.full((k_train, L), math.log(0.6)))
        self.k_train = k_train

    def init_theta(self, H):
        return self.enc(feats(H)) * math.pi

    def alpha(self, k, const=None):
        if const is not None:
            return torch.full((L * M,), const)
        return self.log_alpha.exp()[min(k, self.k_train - 1)].repeat_interleave(M)

    def refine(self, theta, H, steps, phys, const_alpha=None):
        outs = [theta]
        for k in range(steps):
            with torch.enable_grad():
                th = theta.detach().requires_grad_(True)
                g, = torch.autograd.grad(softmin(phys.rates(quantize_phase_ste(th, 1), H)).sum(), th)
            g = g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
            theta = theta + self.alpha(k, const_alpha) * g
            outs.append(theta)
        return outs


def hard_minrate(phys, theta, H):
    with torch.no_grad():
        return phys.rates(quantize_phase_hard_torch(theta, 1), H).min(-1).values


def evaluate(model, phys, Hte, tag, K_list=(0, 1, 2, 3, 5, 10), polish_at=(0, 3, 10)):
    B = Hte.shape[0]
    rng = torch.Generator().manual_seed(1)
    th_rand = (torch.rand(B, L * M, generator=rng) * 2 - 1) * math.pi
    out = {}
    with torch.no_grad():
        th_enc = model.init_theta(Hte)
    arms = {"meta (encoder + learned alpha)": (th_enc, None), "random init + learned alpha": (th_rand, None),
            "encoder + constant alpha=0.6": (th_enc, 0.6)}
    for name, (th0, const) in arms.items():
        outs = model.refine(th0, Hte, max(K_list), phys, const)
        row = {}
        for kk in K_list:
            th = outs[kk].detach()
            pre = hard_minrate(phys, th, Hte)
            row[kk] = dict(pre=float(pre.mean()))
            if kk in polish_at:
                bits = quantize_phase_hard_torch(th, 1).numpy()
                row[kk]["post2"] = float(np.mean([np_polish(bits[i], Hte[i].numpy().astype(complex), 2) for i in range(B)]))
                row[kk]["post20"] = float(np.mean([np_polish(bits[i], Hte[i].numpy().astype(complex), 20) for i in range(B)]))
        out[name] = row
        print(f"  [{tag}] {name:34s} " + " | ".join(
            f"K={kk}: pre={row[kk]['pre']:.3f}" + (f" p2={row[kk]['post2']:.3f} p20={row[kk]['post20']:.3f}" if 'post2' in row[kk] else "")
            for kk in K_list), flush=True)
    return out


def main():
    smoke = sys.argv[1] == "smoke"
    phys = Phys()
    # validate differentiable physics against the numpy evaluator used everywhere else
    tests = make_tasks([5000], [(0.99, 1.4)], n_steps=40, stride=10)
    Ht = torch.tensor(np.stack(tests), dtype=torch.complex64)
    th = (torch.rand(len(tests), L * M) * 2 - 1) * math.pi
    q = quantize_phase_hard_torch(th, 1)
    ref = np.array([np_minr(q[i].numpy().astype(float), tests[i]) for i in range(len(tests))])
    mine = phys.rates(q, Ht).min(-1).values.numpy()
    print(f"physics check: max |torch - numpy evaluator| = {np.abs(ref - mine).max():.2e} (min-rates ~{ref.mean():.3f})", flush=True)
    assert np.abs(ref - mine).max() < 1e-3

    CELLS = [(r, v) for r in (0.7, 0.9, 0.99) for v in (0.5, 1.4, 3.0)]
    n_tr = 12 if smoke else 300
    tr = make_tasks(list(range(1000, 1000 + n_tr)), CELLS)
    te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:100]
    Htr = torch.tensor(np.stack(tr), dtype=torch.complex64)
    Hte = torch.tensor(np.stack(te), dtype=torch.complex64)
    print(f"train channels {len(tr)}, held-out {len(te)}", flush=True)

    model = Unrolled()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    iters, Bsz = (30, 16) if smoke else (4000, 64)
    t0 = time.time()
    evaluate(model, phys, Hte[:20] if smoke else Hte, "untrained", K_list=(0, 3), polish_at=(0,)) if smoke else None
    for it in range(1, iters + 1):
        idx = torch.randint(0, len(tr), (Bsz,))
        H = Htr[idx]
        outs = model.refine(model.init_theta(H), H, model.k_train, phys)
        loss = -sum(softmin(phys.rates(quantize_phase_ste(o, 1), H)).mean() for o in outs) / len(outs)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if it % (10 if smoke else 250) == 0:
            with torch.no_grad():
                o = model.refine(model.init_theta(Hte), Hte, 3, phys)
            print(f"iter {it}/{iters} loss={loss.item():.4f} heldout hard min-rate: K0={hard_minrate(phys, o[0], Hte).mean():.3f} "
                  f"K3={hard_minrate(phys, o[3], Hte).mean():.3f}  alpha={model.log_alpha.exp().detach().numpy().round(2).tolist()}  {(time.time()-t0)/60:.1f} min", flush=True)
    torch.save(model.state_dict(), "unrolled_meta.pt")
    res = evaluate(model, phys, Hte, "FINAL held-out")
    np.save("unrolled_meta_results.npy", np.array(res, dtype=object))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
