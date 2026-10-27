"""Iteration 2 of the fast-channel RL test (rho=0.7, high speed, stale + noisy CSI, S probes per step).
New ingredients (both learned by group-relative policy updates from PROBED rates, pre-trained offline in simulation):
  * channel-prediction knob alpha: each candidate is designed on  H_use = m_t + alpha*(H_hat_t - m_t),  m_t = running mean of past
    estimates (slow, predictable part). alpha=1 -> trust the stale estimate fully; alpha<1 -> shrink toward the predictable part.
    alpha ~ N(mu_alpha, 0.2^2) per candidate, mu_alpha learned (GRPO).
  * temporal policy network: pi(w0 | H_hat_t, H_hat_{t-1}, m_t) proposes 6 of the 8 starts (2 = warm starts).
Arms (equal probes):  O oracle-CSI | A blind | D warm-S | D-alpha (warm-S + learned alpha) | E (iteration-1 policy) | E2 (new)
Usage: python3 probe_rl2.py <rho> <v> <S> <T_test> <n_train_traj> <T_train> <nref>"""
import sys, os, time, copy, numpy as np, torch, torch.nn as nn
torch.set_num_threads(3)
from big_sys import sgn, relax
from impair_lib import System, Dev, LEVELS, minrate
from probe_rl import traj, noisy, feat, norm_w, StartPolicy, K, M, L, D


class TemporalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * 2 * M * K, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, D))
        self.log_sd = nn.Parameter(torch.tensor(float(np.log(0.4))))
        self.mu_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, f):
        return self.net(f)


class AlphaOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_alpha = nn.Parameter(torch.tensor(1.0))


def run_traj(arm, pol, opt, model, true, Hs, S, T, nref, seed, learn=True):
    g = torch.Generator().manual_seed(seed)
    w_prev = torch.randn(D, generator=g) * 0.5
    Hh_prev = noisy(Hs[0], -10, g); m = Hh_prev.clone()
    rates, alphas, picks = [], [], []
    for t in range(1, T + 1):
        Ht = Hs[t + 1]; Hh = noisy(Hs[t], -10, g)
        m = 0.7 * m + 0.3 * Hh
        if arm in ("O", "A"):
            Hd = Ht if arm == "O" else Hh
            w = relax(model, Hd[None], torch.randn(1, D, generator=g) * 0.5, nref)
            rates.append(minrate(true, sgn(w), Ht[None]).item()); Hh_prev = Hh; continue
        wn = norm_w(w_prev)
        n_pol = 0 if arm in ("D", "D-alpha") else (S // 2 if arm == "E" else S - 2)
        warm = wn.unsqueeze(0) + 0.3 * torch.randn(S - n_pol, D, generator=g)
        if n_pol:
            f = feat(Hh) if arm == "E" else torch.cat([feat(Hh), feat(Hh_prev), feat(m)])
            mu = pol(f); sd = pol.log_sd.exp()
            wpol = mu.detach() + sd.detach() * torch.randn(n_pol, D, generator=g)
            w0 = torch.cat([wpol, warm], 0)
        else:
            w0 = warm
        if arm in ("D-alpha", "E2"):
            ea = 0.2 * torch.randn(S, generator=g)
            alpha = pol.mu_alpha.detach() + ea
        else:
            alpha = torch.ones(S)
        Huse = m.unsqueeze(0) + alpha.view(S, 1, 1) * (Hh - m).unsqueeze(0)
        w = relax(model, Huse, w0, nref)
        r_true = minrate(true, sgn(w), Ht[None].expand(S, -1, -1))
        r_meas = r_true * (1 + 0.05 * torch.randn(S, generator=g))
        b = int(r_meas.argmax())
        rates.append(r_true[b].item()); w_prev = w[b].detach(); Hh_prev = Hh
        if n_pol: picks.append(int(b < n_pol))
        if arm in ("D-alpha", "E2"): alphas.append(float(pol.mu_alpha.detach()))
        if learn and arm != "D":
            A = (r_meas - r_meas.mean()) / (r_meas.std() + 1e-6)
            loss = 0
            if arm in ("D-alpha", "E2"):
                loss = loss - (A * (-0.5 * ((alpha - pol.mu_alpha) / 0.2) ** 2)).mean()
            if n_pol:
                mu = pol(f); sd = pol.log_sd.exp()
                lp = (-0.5 * ((w0[:n_pol] - mu) / sd) ** 2 - pol.log_sd).sum(-1)
                loss = loss - (A[:n_pol] * lp).mean() / D
                wt = torch.softmax(A, 0).detach()
                loss = loss + (wt.unsqueeze(1) * (mu.unsqueeze(0) - norm_w(w.detach())) ** 2).sum(0).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
            if n_pol:
                with torch.no_grad(): pol.log_sd.clamp_(np.log(0.1), np.log(0.8))
    return np.array(rates), (np.mean(alphas[-10:]) if alphas else None), (np.mean(picks) if picks else None)


def make(arm):
    torch.manual_seed(0)
    p = {"E": StartPolicy, "E2": TemporalPolicy, "D-alpha": AlphaOnly}.get(arm, AlphaOnly)()
    groups = [dict(params=[q for n, q in p.named_parameters() if n != "mu_alpha"], lr=3e-4)]
    if hasattr(p, "mu_alpha"): groups.append(dict(params=[p.mu_alpha], lr=0.02))
    return p, torch.optim.Adam(groups)


def main():
    rho, v, S, T, ntr, Ttr, nref = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    tag = f"rho{rho}_v{v}_S{S}"
    Sy = System(K, M, L)
    th, _, _ = Sy.sample(LEVELS["moderate"], 7)
    true = Dev(Sy, th); model = true
    test = [traj(sd, T, rho, v) for sd in (5000, 5001, 5002, 5003)]
    train = [traj(sd, Ttr, rho, v) for sd in range(1000, 1000 + ntr)]
    print(f"[{tag}] iteration 2 | {ntr} pre-training trajectories x {Ttr} steps | test 4 x {T}", flush=True)
    pre = {}
    t0 = time.time()
    for arm in ("D-alpha", "E", "E2"):                                  # offline pre-training in the simulator (resumable)
        ck = f"pre_{tag}_n{ntr}_{arm}.pt"
        if os.path.exists(ck):
            pre[arm] = torch.load(ck); print(f"  loaded pre-trained {arm} from {ck}", flush=True); continue
        p, o = make(arm)
        for i, Hs in enumerate(train):
            r, a, pk = run_traj(arm, p, o, model, true, Hs, S, Ttr, nref, 100 + i)
        pre[arm] = copy.deepcopy(p.state_dict()); torch.save(pre[arm], ck)
        print(f"  pre-trained {arm}: last-trajectory rate {r.mean():.3f}" + (f", alpha {a:.2f}" if a is not None else "")
              + (f", policy pick {100*pk:.0f}%" if pk is not None else "") + f" ({(time.time()-t0)/60:.1f} min)", flush=True)
    rf = f"probe_rl2_{tag}_n{ntr}_partial.npy"
    res = np.load(rf, allow_pickle=True).item() if os.path.exists(rf) else {}
    for arm in ("O", "A", "D", "D-alpha", "E", "E2"):
        if arm in res:
            R = res[arm]; print(f"{arm:8s} (resumed) mean {R.mean():.3f}", flush=True); continue
        out, al, pks = [], [], []
        for i, Hs in enumerate(test):
            p, o = make(arm)
            if arm in pre: p.load_state_dict(pre[arm])
            r, a, pk = run_traj(arm, p, o, model, true, Hs, S, T, nref, 11 + i, learn=True)
            out.append(r); al.append(a); pks.append(pk)
        R = np.stack(out); res[arm] = R; q = T // 4; np.save(rf, np.array(res, dtype=object))
        print(f"{arm:8s} mean {R.mean():.3f} | quarters {[round(float(R[:, j*q:(j+1)*q].mean()), 3) for j in range(4)]} | per-traj {np.round(R.mean(1), 3).tolist()}"
              + (f" | alpha {np.round(al, 2).tolist()}" if al[0] is not None else "") + (f" | policy pick {100*np.mean(pks):.0f}%" if pks[0] is not None else ""), flush=True)
    for a, b in (("E2", "D"), ("E2", "E"), ("E2", "D-alpha"), ("D-alpha", "D"), ("E", "D"), ("D", "A")):
        d = (res[a] - res[b]).mean(1)
        print(f"{a} - {b}: per-traj {np.round(d, 3).tolist()} overall {d.mean():+.3f} ({100*d.mean()/res[b].mean():+.1f}%)", flush=True)
    np.save(f"probe_rl2_{tag}.npy", np.array(res, dtype=object))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
