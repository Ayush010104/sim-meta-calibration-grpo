"""Hybrid GRPO on the scaled-up system (K=10, M=64, L=6, 384 bits), optionally with unknown phase offsets (device mismatch).
Policy over the relaxed START w0 ~ N(mu, sigma^2) per channel; each sample is refined by n_ref exact-gradient steps on the
NOMINAL model, rounded to {0,pi} (x=+-1) and scored by the TRUE device (black-box reward, one query per sample).
Methods (same sampling, same refinement, same query budget G per update):
  grpo     group-relative advantage policy gradient on (mu, log sigma)
  cem      elite refit (top 25%): mu <- elite mean, sigma <- elite std      (black-box evolutionary baseline)
  restart  mu fixed at 0, sigma fixed: random restarts                       (multi-start baseline)
Reported per update: best-so-far over all queried samples (what a black-box user would deploy) and the rate of the
deterministic refine(mu) configuration.   Usage: python3 hybrid_grpo.py <mismatch_sigma> <method> <lr> <sigma0> <updates> <dev|test>"""
import sys, time, numpy as np, torch
torch.set_num_threads(4)
from big_sys import make_env, channels, Sys, softmin, sgn, rate, polish_true

K, M, L = 10, 64, 6
D = L * M
G, NREF = 16, 10


def refine(model, H, w, n=NREF):
    for i in range(n):
        beta = 1.0 + 4.0 * i / n
        with torch.enable_grad():
            t = w.detach().requires_grad_(True)
            g, = torch.autograd.grad(softmin(model.rates(torch.tanh(beta * t), H)).sum(), t)
        w = w + 0.1 * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    return w


def get_channels(which):
    seeds = [4000, 4001] if which == "dev" else [5000, 5001, 5002, 5003]
    n = 10 if which == "dev" else 10
    return channels(K, M, L, seeds, n_steps=100, stride=10)[: (10 if which == "dev" else 40)]


def main():
    sig_m, method, lr, sigma0, U, which = float(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    env = make_env(K, M, L, 0)
    nom = Sys(env)
    psi = torch.randn(L, M, generator=torch.Generator().manual_seed(7)) * sig_m if sig_m > 0 else None
    true = Sys(env, psi)
    H = get_channels(which); B = H.shape[0]
    Hrep = H.unsqueeze(1).expand(B, G, K, M).reshape(B * G, M, K) if False else H.unsqueeze(1).expand(B, G, *H.shape[1:]).reshape(B * G, *H.shape[1:])
    torch.manual_seed(0)
    mu = torch.zeros(B, D, requires_grad=(method == "grpo"))
    ls = torch.full((B, 1), float(np.log(sigma0)), requires_grad=(method == "grpo"))
    opt = torch.optim.Adam([mu, ls], lr=lr) if method == "grpo" else None
    best = torch.zeros(B); curve_best, curve_det = [], []
    t0 = time.time()
    for u in range(1, U + 1):
        sd = ls.exp().detach()
        eps = torch.randn(B, G, D)
        w0 = (mu.detach().unsqueeze(1) + sd.unsqueeze(1) * eps)
        wf = refine(nom, Hrep, w0.reshape(B * G, D))
        r = rate(true, sgn(wf), Hrep).reshape(B, G)
        best = torch.maximum(best, r.max(1).values)
        if method == "grpo":
            A = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
            m, s = mu.unsqueeze(1), ls.exp().unsqueeze(1)
            logp = (-0.5 * ((w0 - m) / s) ** 2 - ls.unsqueeze(1)).sum(-1)       # (B,G)
            loss = -(A * logp).mean() / D
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                ls.clamp_(np.log(0.15), np.log(2.0))
        elif method == "cem":
            idx = r.topk(G // 4, dim=1).indices
            el = torch.gather(w0, 1, idx.unsqueeze(-1).expand(-1, -1, D))
            mu = (0.5 * mu.detach() + 0.5 * el.mean(1))
            ls = torch.log((0.7 * ls.exp() + 0.3 * el.std(1).mean(-1, keepdim=True)).clamp(0.15, 2.0))
        curve_best.append(best.mean().item())
        if u % 5 == 0 or u == 1:
            det = rate(true, sgn(refine(nom, H, mu.detach().clone())), H).mean().item()
            curve_det.append((u, det))
            print(f"u={u:3d} queries/ch={u*G:5d} best-so-far={curve_best[-1]:.3f} det(mu)={det:.3f} sigma={ls.exp().mean().item():.2f} ({time.time()-t0:.0f}s)", flush=True)
    tag = f"hyb_{which}_m{sig_m}_{method}_lr{lr}_s{sigma0}"
    np.save(tag + ".npy", np.array(dict(best=curve_best, det=curve_det), dtype=object))
    # final deployed configuration: best queried sample is not stored; report polish on det(mu) refinement for reference
    x = sgn(refine(nom, H, mu.detach().clone()))
    print("final det(mu) + black-box polish(10):", polish_true(true, x, H, 10)[1].mean().item(), flush=True)

if __name__ == "__main__":
    main()
