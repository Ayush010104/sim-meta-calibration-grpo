import numpy as np, torch
torch.set_num_threads(4)
from big_sys import make_env, channels, Sys, sgn, rate, relax
K, M, L = 10, 64, 6; D = L*M
env = make_env(K, M, L, 0)
psi = torch.randn(L, M, generator=torch.Generator().manual_seed(7)) * 0.3
true = Sys(env, psi)
pool = channels(K, M, L, list(range(1000, 1020)), n_steps=100, stride=10)
def R(psih, Bc, nref, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(200, generator=g)[:Bc]; Hb = pool[idx]; w0 = torch.randn(Bc, D, generator=g) * 0.5
    return rate(true, sgn(relax(Sys(env, psih), Hb, w0, nref)), Hb).mean().item()
for nref in (20, 50):
    print(f"-- nref={nref}, Bc=6, mean over 6 independent (channels,starts) draws")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = [R(a * psi, 6, nref, s) for s in range(6)]
        print(f"  psi_hat = {a:.2f} * psi_true : reward {np.mean(v):.3f}  (std over draws {np.std(v):.3f})")
# local noise: perturbations of psi_true by N(0,s^2) with CRN
for s in (0.05, 0.15):
    r0 = np.mean([R(psi, 6, 20, k) for k in range(3)])
    rs = [np.mean([R(psi + s*torch.randn(L, M), 6, 20, k) for k in range(3)]) for _ in range(8)]
    print(f"around truth, perturbation s={s}: reward at truth {r0:.3f}, perturbed mean {np.mean(rs):.3f} std {np.std(rs):.3f}")
