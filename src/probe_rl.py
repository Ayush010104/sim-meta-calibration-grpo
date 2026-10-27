"""Fast-changing channels + imperfect CSI: can RL make adaptation faster?
K=10, M=64, L=6, moderate device (calibrated model = modeled-class truth). Each time step: CSI = previous-step channel
(aging) + estimation noise (-10 dB); the BS may probe S configurations on the true channel (5% noisy rate reports),
deploys the best probed. Each candidate = relaxed-gradient design (NREF steps) on the estimate from some start w0.
  A  blind          : 1 design, no probes                     O  oracle-CSI : 1 design on true channel, no probes
  B  restart-S      : S random starts                          D  warm-S     : S starts around the last deployed solution
  E  RL-start policy: a channel-conditioned policy network pi(w0 | H_hat) proposes S/2 starts (other S/2 = warm, as D);
                      trained with group-relative, advantage-weighted policy updates from the PROBED rates
                      (GRPO group = the S probes of a step).   E-pre: policy pre-trained (meta-trained) on other trajectories in a
                      simulator, then keeps adapting online;  E-frozen: pre-trained, no online updates;  E-scratch: online only.
Usage: python3 probe_rl.py <rho> <v> <S> <T_test> <n_train_traj> <T_train> <nref>"""
import sys, time, copy, numpy as np, torch, torch.nn as nn
torch.set_num_threads(3)
from big_sys import make_env, sgn, relax
from impair_lib import System, Dev, LEVELS, minrate
K, M, L = 10, 64, 6; D = L * M


def traj(seed, T, rho, v):
    e = make_env(K, M, L, seed, rho, v)
    H = [e.H_true.copy()]
    for _ in range(T + 1):
        e.advance_channel(); H.append(e.H_true.copy())
    return torch.tensor(np.stack(H), dtype=torch.complex64)


def noisy(H, db, g):
    eps = float(np.sqrt(10 ** (db / 10)))
    nrm = H.abs().pow(2).sum(0, keepdim=True).sqrt()
    n = (torch.randn(H.shape, generator=g) + 1j * torch.randn(H.shape, generator=g)) / np.sqrt(2)
    return H + n.to(torch.complex64) * eps * nrm / np.sqrt(H.shape[0])


def feat(H):
    x = torch.cat([H.real.flatten(), H.imag.flatten()])
    return x / (x.abs().mean() + 1e-12)


class StartPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * M * K, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, D))
        self.log_sd = nn.Parameter(torch.tensor(float(np.log(0.4))))

    def forward(self, f):
        return self.net(f)


def norm_w(w):
    return 0.5 * w / (w.pow(2).mean(-1, keepdim=True).sqrt() + 1e-9)


def step(method, pol, opt, model, true, Hh, Ht, w_prev, S, nref, g, learn):
    if method in ("A", "O"):
        Hd = Ht if method == "O" else Hh
        w = relax(model, Hd[None], torch.randn(1, D, generator=g) * 0.5, nref)
        return minrate(true, sgn(w), Ht[None]).item(), w[0].detach(), None
    wn = norm_w(w_prev)
    if method == "B":
        w0 = torch.randn(S, D, generator=g) * 0.5
    elif method == "D":
        w0 = wn.unsqueeze(0) + 0.3 * torch.randn(S, D, generator=g)
    else:
        f = feat(Hh)
        mu = pol(f)
        sd = pol.log_sd.exp()
        h = S // 2
        z = torch.randn(h, D, generator=g)
        wpol = (mu.detach() + sd.detach() * z)
        w0 = torch.cat([wpol, wn.unsqueeze(0) + 0.3 * torch.randn(S - h, D, generator=g)], 0)
    w = relax(model, Hh.unsqueeze(0).expand(S, -1, -1), w0, nref)
    r_true = minrate(true, sgn(w), Ht[None].expand(S, -1, -1))
    r_meas = r_true * (1 + 0.05 * torch.randn(S, generator=g))
    b = int(r_meas.argmax())
    info = None
    if method == "E":
        info = int(b < S // 2)
        if learn:
            A = (r_meas - r_meas.mean()) / (r_meas.std() + 1e-6)
            mu = pol(f); sd = pol.log_sd.exp()
            # (i) GRPO term on the policy's own proposals
            lp = (-0.5 * ((w0[:h] - mu) / sd) ** 2 - pol.log_sd).sum(-1)
            pg = -(A[:h] * lp).mean() / D
            # (ii) advantage-weighted regression toward the refined solutions of the whole group (AWR, temperature 1)
            wt = torch.softmax(A, 0).detach()
            tgt = norm_w(w.detach())
            awr = (wt.unsqueeze(1) * (mu.unsqueeze(0) - tgt) ** 2).sum(0).mean()
            loss = pg + awr
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
            with torch.no_grad(): pol.log_sd.clamp_(np.log(0.1), np.log(0.8))
    return r_true[b].item(), w[b].detach(), info


def run_traj(method, pol, opt, model, true, Hs, S, T, nref, seed, learn=True):
    g = torch.Generator().manual_seed(seed)
    w_prev = torch.randn(D, generator=g) * 0.5
    rates, picks = [], []
    for t in range(1, T + 1):
        Ht = Hs[t + 1]; Hh = noisy(Hs[t], -10, g)
        r, w_prev, info = step(method, pol, opt, model, true, Hh, Ht, w_prev, S, nref, g, learn)
        rates.append(r)
        if info is not None: picks.append(info)
    return np.array(rates), (np.mean(picks) if picks else None)


def main():
    rho, v, S, T, ntr, Ttr, nref = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    tag = f"rho{rho}_v{v}_S{S}_n{nref}"
    Sy = System(K, M, L)
    th, _, _ = Sy.sample(LEVELS["moderate"], 7)
    true = Dev(Sy, th); model = true
    test = [traj(sd, T, rho, v) for sd in (5000, 5001, 5002, 5003)]
    ag = np.mean([float(((tr[1:] - tr[:-1]).abs().pow(2).sum() / tr[1:].abs().pow(2).sum())) for tr in test])
    print(f"[{tag}] channel change per step NMSE {10*np.log10(ag):.1f} dB | CSI = previous step + -10 dB noise | S={S} probes, {nref} gradient steps per candidate", flush=True)
    torch.manual_seed(0)
    pol = StartPolicy(); opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    pol_scratch = copy.deepcopy(pol)
    t0 = time.time()
    for i, sd in enumerate(range(1000, 1000 + ntr)):                       # offline pre-training in the simulator
        r, pk = run_traj("E", pol, opt, model, true, traj(sd, Ttr, rho, v), S, Ttr, nref, 100 + i)
        if (i + 1) % 5 == 0:
            print(f"  pretrain traj {i+1}/{ntr}: mean rate {r.mean():.3f}, policy-proposal chosen {100*pk:.0f}% ({(time.time()-t0)/60:.1f} min)", flush=True)
    pre_state = copy.deepcopy(pol.state_dict())
    res = {}
    arms = [("O", None, False), ("A", None, False), ("B", None, False), ("D", None, False),
            ("E-scratch", "scratch", True), ("E-frozen", "pre", False), ("E-pre", "pre", True)]
    for name, init, learn in arms:
        out, picks = [], []
        for i, Hs in enumerate(test):
            p = StartPolicy()
            p.load_state_dict(pre_state if init == "pre" else pol_scratch.state_dict())
            o = torch.optim.Adam(p.parameters(), lr=3e-4)
            r, pk = run_traj(name[0] if name[0] != "E" else "E", p, o, model, true, Hs, S, T, nref, 11 + i, learn)
            out.append(r); picks.append(pk)
        R = np.stack(out); res[name] = R
        q = T // 4
        print(f"{name:10s} mean {R.mean():.3f} | quarters {[round(R[:, j*q:(j+1)*q].mean(), 3) for j in range(4)]} | per-traj {np.round(R.mean(1), 3).tolist()}"
              + (f" | policy pick {100*np.mean(picks):.0f}%" if picks[0] is not None else ""), flush=True)
    for a, b in (("E-pre", "D"), ("E-pre", "B"), ("E-pre", "E-scratch"), ("E-pre", "E-frozen"), ("D", "B"), ("D", "A")):
        d = (res[a] - res[b]).mean(1)
        print(f"{a} - {b}: per-traj {np.round(d, 3).tolist()} overall {d.mean():+.3f}", flush=True)
    np.save(f"probe_rl_{tag}.npy", np.array(res, dtype=object))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
