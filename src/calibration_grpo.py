"""Calibration GRPO: learn the device's unknown phase offsets psi (L x M = 384 numbers) from MEASURED RATES ONLY.
Policy: psi_hat ~ N(m, s^2 I).  Reward of a candidate psi_hat (black box, non-differentiable):
   mean over Bc fresh training channels of the TRUE-device min-rate of  round( relaxed-gradient-solution under model psi_hat ).
(common random numbers: all G candidates share the same channels and the same random starts; antithetic psi_hat pairs.)
Group-relative advantage, Adam on (m, log s). No gradient ever flows through the true device.
Deploy: relaxed gradient (200 steps) under the model m, round to {0,pi}, evaluate on the true device.
Usage: python3 calibration_grpo.py <sigma_true> <updates> <G> <Bc> <nref> <lr> <s0> <tag>"""
import sys, time, numpy as np, torch
torch.set_num_threads(4)
from big_sys import make_env, channels, Sys, sgn, rate, polish_true, relax
K, M, L = 10, 64, 6
D = L * M


def sys_batch(env, psi):            # psi: (L, M)
    return Sys(env, psi)


def solve(model, H, w0, n):
    return sgn(relax(model, H, w0, N=n))


def circ_rmse(a, b):                 # compare after removing per-layer common phase
    d = torch.angle(torch.exp(1j * (a - b)))
    d = torch.angle(torch.exp(1j * (d - torch.angle(torch.exp(1j * d).mean(1, keepdim=True)))))
    return d.pow(2).mean().sqrt().item()


def main():
    sig, U, G, Bc, nref, lr, s0, tag = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
    env = make_env(K, M, L, 0)
    nom = Sys(env)
    psi_true = torch.randn(L, M, generator=torch.Generator().manual_seed(7)) * sig
    true = Sys(env, psi_true)
    pool = channels(K, M, L, list(range(1000, 1020)), n_steps=100, stride=10)          # 200 training channels
    dev = channels(K, M, L, [4000, 4001], n_steps=100, stride=10)[:10]
    w0d = torch.randn(dev.shape[0], D, generator=torch.Generator().manual_seed(1)) * 0.5
    print(f"pool {pool.shape[0]} dev {dev.shape[0]}; sigma_true={sig}; ||psi||rms={psi_true.pow(2).mean().sqrt():.3f}", flush=True)
    print(f"dev reference: nominal-model gradient {rate(true, solve(nom, dev, w0d, 50), dev).mean():.3f} | oracle-model gradient {rate(true, solve(true, dev, w0d, 50), dev).mean():.3f}", flush=True)
    torch.manual_seed(0)
    m = torch.zeros(L, M, requires_grad=True)
    fixed = len(sys.argv) > 9 and sys.argv[9] == 'fixed'
    ls = torch.tensor(float(np.log(s0)), requires_grad=not fixed)
    opt = torch.optim.Adam([m] if fixed else [m, ls], lr=lr)
    t0 = time.time(); log = []
    for u in range(0, U + 1):
        if u % 20 == 0:
            with torch.no_grad():
                md = Sys(env, m.detach())
                dr = rate(true, solve(md, dev, w0d, 50), dev).mean().item()
            rm = circ_rmse(m.detach(), psi_true)
            log.append((u, dr, rm))
            print(f"u={u:4d} dev deploy-rate={dr:.3f}  rmse(m,psi)={rm:.3f} (prior rmse {circ_rmse(torch.zeros(L,M), psi_true):.3f})  s={ls.exp().item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        if u == U: break
        idx = torch.randperm(pool.shape[0])[:Bc]
        Hb = pool[idx]
        w0 = torch.randn(Bc, D) * 0.5
        eps = torch.randn(G // 2, L, M); eps = torch.cat([eps, -eps], 0)
        cand = m.detach().unsqueeze(0) + ls.exp().detach() * eps                              # (G, L, M)
        R = torch.zeros(G)
        for g in range(G):
            R[g] = rate(true, solve(Sys(env, cand[g]), Hb, w0, nref), Hb).mean()
        A = (R - R.mean()) / (R.std() + 1e-6)
        logp = (-0.5 * ((cand - m.unsqueeze(0)) / ls.exp()) ** 2 - ls).sum((1, 2))
        loss = -(A * logp).mean() / D
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if not fixed: ls.clamp_(np.log(0.03), np.log(0.4))
    torch.save(dict(m=m.detach(), psi=psi_true, log=log), f"calibration_{tag}.pt")
    print("saved", flush=True)


if __name__ == "__main__":
    main()
