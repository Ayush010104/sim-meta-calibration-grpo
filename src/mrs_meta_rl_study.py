"""Offline META-RL study (this session's file; prefix mrs_ to avoid clashing with other sessions in this folder).
Fast-changing channels with stale/noisy CSI (rho=0.7, high speed), calibrated K=10/M=64/L=6 SIM.
Every step: 8 probed candidates (policy starts + warm starts), relaxed-gradient design on the stale estimate, best measured is
deployed; score = true worst-user rate. Learning signal = GRPO (group-relative advantage over the 8 probes) + AWR.
Arms
  D       warm start, no learning (8 warm candidates)
  MT      multi-task offline pre-training of pi(w0|H_hat) (iteration-1 method), online fine-tuning (all weights)
  REP     Reptile offline meta-RL over trajectory segments (inner: 10 online steps), online fine-tuning (all weights)
  CTX     CAVIA-style context meta-RL: pi(w0|H_hat, z); offline: shared weights + per-trajectory context z; test: ONLY z adapts
  FC      (novel) Feedback-Conditioned in-context policy: pi(w0 | H_hat_t, H_hat_{t-1}, w_{t-1}, measured-rate signal);
          adapts in context, no test-time gradients
  FCX     (novel, full) FC + fast context z (CAVIA) -> in-context + low-dimensional test-time adaptation
Paired evaluation: CSI noise and measurement noise are identical across arms (seeded per trajectory & step).
Usage: python3 mrs_meta_rl_study.py train <arm> <rho> <v> | test <arm> <rho> <v> | report <rho> <v>"""
import sys, os, time, copy, numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
from big_sys import sgn, relax
from impair_lib import System, Dev, LEVELS, minrate
from probe_rl import traj, feat, norm_w, K, M, L, D

S, NPOL, NREF, ZD = 8, 4, 50, 32
NTR, TTR, TTE = 30, 50, 100
TEST_SEEDS = list(range(5000, 5008))


def noisy_seeded(H, seed):
    g = torch.Generator().manual_seed(seed)
    eps = float(np.sqrt(0.1))                                            # -10 dB
    nrm = H.abs().pow(2).sum(0, keepdim=True).sqrt()
    n = (torch.randn(H.shape, generator=g) + 1j * torch.randn(H.shape, generator=g)) / np.sqrt(2)
    return H + n.to(torch.complex64) * eps * nrm / np.sqrt(H.shape[0])


class Policy(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.fc = arm in ("FC", "FCX"); self.ctx = arm in ("CTX", "FCX")
        din = 2 * M * K + (2 * M * K + D + 1 if self.fc else 0) + (ZD if self.ctx else 0)
        self.net = nn.Sequential(nn.Linear(din, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, D))
        self.log_sd = nn.Parameter(torch.tensor(float(np.log(0.4))))

    def forward(self, x):
        return self.net(x)


class Runner:
    """runs one trajectory; carries warm solution, previous estimate, feedback signal, context z"""
    def __init__(self, arm, pol, model, true, Hs, tseed, test_lr):
        self.arm, self.pol, self.model, self.true, self.Hs, self.tseed = arm, pol, model, true, Hs, tseed
        g = torch.Generator().manual_seed(tseed)
        self.g = torch.Generator().manual_seed(tseed + 77)
        self.w_prev = torch.randn(D, generator=g) * 0.5
        self.Hh_prev = noisy_seeded(Hs[0], tseed * 1000)
        self.r_last, self.r_avg = 0.0, 1e-3
        self.z = torch.zeros(ZD, requires_grad=True) if pol is not None and pol.ctx else None
        self.zopt = torch.optim.Adam([self.z], lr=0.05) if self.z is not None else None
        self.wopt = torch.optim.Adam(pol.parameters(), lr=test_lr) if (pol is not None and test_lr) else None

    def inputs(self, Hh):
        parts = [feat(Hh)]
        if self.pol.fc:
            parts += [feat(self.Hh_prev), norm_w(self.w_prev), torch.tensor([self.r_last / self.r_avg - 1.0])]
        if self.pol.ctx:
            parts.append(self.z)
        return torch.cat(parts)

    def step(self, t, update_w, update_z):
        Hs = self.Hs; Ht = Hs[t + 1]; Hh = noisy_seeded(Hs[t], self.tseed * 1000 + t)
        wn = norm_w(self.w_prev)
        npol = 0 if self.pol is None else NPOL
        warm = wn.unsqueeze(0) + 0.3 * torch.randn(S - npol, D, generator=self.g)
        if npol:
            x = self.inputs(Hh)
            mu = self.pol(x); sd = self.pol.log_sd.exp()
            w0 = torch.cat([mu.detach() + sd.detach() * torch.randn(npol, D, generator=self.g), warm], 0)
        else:
            w0 = warm
        w = relax(self.model, Hh.unsqueeze(0).expand(S, -1, -1), w0, NREF)
        r_true = minrate(self.true, sgn(w), Ht[None].expand(S, -1, -1))
        gm = torch.Generator().manual_seed(self.tseed * 1000 + t + 500_000)
        r_meas = r_true * (1 + 0.05 * torch.randn(S, generator=gm))
        b = int(r_meas.argmax())
        if npol and (update_w or update_z):
            A = (r_meas - r_meas.mean()) / (r_meas.std() + 1e-6)
            lp = (-0.5 * ((w0[:npol] - mu) / sd) ** 2 - self.pol.log_sd).sum(-1)
            loss = -(A[:npol] * lp).mean() / D + (torch.softmax(A, 0).unsqueeze(1) * (mu.unsqueeze(0) - norm_w(w.detach())) ** 2).sum(0).mean()
            if update_w: self.wopt.zero_grad()
            if update_z: self.zopt.zero_grad()
            loss.backward()
            if update_w:
                nn.utils.clip_grad_norm_(self.pol.parameters(), 1.0); self.wopt.step()
                with torch.no_grad(): self.pol.log_sd.clamp_(np.log(0.1), np.log(0.8))
            if update_z: self.zopt.step()
            self.pol.zero_grad()
        self.w_prev = w[b].detach(); self.Hh_prev = Hh
        self.r_last = float(r_meas[b]); self.r_avg = 0.9 * self.r_avg + 0.1 * self.r_last if self.r_avg > 1e-3 else self.r_last
        return r_true[b].item(), int(b < npol) if npol else None


def setup(rho, v):
    Sy = System(K, M, L)
    th, _, _ = Sy.sample(LEVELS["moderate"], 7)
    true = Dev(Sy, th)
    return true, true


def train(arm, rho, v):
    model, true = setup(rho, v)
    torch.manual_seed(0)
    pol = Policy(arm)
    t0 = time.time(); tag = f"rho{rho}_v{v}"
    trajs = [traj(sd, TTR, rho, v) for sd in range(1000, 1000 + NTR)]
    if arm == "REP":
        meta = copy.deepcopy(pol.state_dict()); n_it = NTR * TTR // 10
        for it in range(n_it):
            eps = 1.0 - 0.8 * it / max(n_it - 1, 1)
            pol.load_state_dict(meta)
            j = it % NTR; start = (it // NTR) * 10 % (TTR - 10) + 1
            R = Runner(arm, pol, model, true, trajs[j], 100 + j, 1e-3)
            for t in range(start, start + 10):
                R.step(t, True, False)
            new = pol.state_dict()
            meta = {k: meta[k] + eps * (new[k] - meta[k]) for k in meta}
            if (it + 1) % 30 == 0:
                print(f"[{arm}] meta-iter {it+1}/{n_it} ({(time.time()-t0)/60:.1f} min)", flush=True)
        pol.load_state_dict(meta)
    else:
        for j, Hs in enumerate(trajs):
            R = Runner(arm, pol, model, true, Hs, 100 + j, 3e-4)
            rs = [R.step(t, True, pol.ctx)[0] for t in range(1, TTR + 1)]
            if (j + 1) % 5 == 0:
                print(f"[{arm}] traj {j+1}/{NTR} mean rate {np.mean(rs):.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(pol.state_dict(), f"mrs_{arm}_{tag}.pt")
    print(f"[{arm}] DONE training", flush=True)


TEST_RULE = {"MT": (True, False, 3e-4), "REP": (True, False, 1e-3), "CTX": (False, True, 0), "FC": (False, False, 0), "FCX": (False, True, 0)}


def test(arm, rho, v):
    model, true = setup(rho, v)
    tag = f"rho{rho}_v{v}"
    out, picks = [], []
    for i, sd in enumerate(TEST_SEEDS):
        Hs = traj(sd, TTE, rho, v)
        pol = None
        upd_w, upd_z, lr = TEST_RULE.get(arm, (False, False, 0))
        if arm != "D":
            pol = Policy(arm); pol.load_state_dict(torch.load(f"mrs_{arm}_{tag}.pt"))
        R = Runner(arm, pol, model, true, Hs, sd, lr if upd_w else 0)
        rs, pk = zip(*[R.step(t, upd_w, upd_z) for t in range(1, TTE + 1)])
        out.append(rs); picks += [p for p in pk if p is not None]
        print(f"[{arm}] test traj {i+1}/{len(TEST_SEEDS)} mean {np.mean(rs):.3f}", flush=True)
    np.save(f"mrs_test_{arm}_{tag}.npy", np.array(out))
    print(f"[{arm}] TEST mean {np.mean(out):.3f}" + (f" | policy candidate chosen {100*np.mean(picks):.0f}%" if picks else ""), flush=True)


def report(rho, v):
    from scipy import stats
    tag = f"rho{rho}_v{v}"
    R = {a: np.load(f"mrs_test_{a}_{tag}.npy") for a in ("D", "MT", "REP", "CTX", "FC", "FCX") if os.path.exists(f"mrs_test_{a}_{tag}.npy")}
    print(f"== {tag}: 8 held-out trajectories x {TTE} steps, 8 probes/step, paired noise ==")
    print(f"{'arm':5s} {'mean':>6s} {'first 20':>9s} {'last 50':>8s}")
    for a, r in R.items():
        print(f"{a:5s} {r.mean():6.3f} {r[:, :20].mean():9.3f} {r[:, 50:].mean():8.3f}")
    for ref in ("D", "MT"):
        if ref not in R: continue
        for a, r in R.items():
            if a == ref: continue
            d = (r - R[ref]).mean(1); ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            print(f"{a} - {ref}: {d.mean():+.4f} [95% CI {d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}] ({100*d.mean()/R[ref].mean():+.1f}%), wins {int((d>0).sum())}/{len(d)} trajectories")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train": train(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    elif mode == "test": test(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    elif mode == "report": report(float(sys.argv[2]), float(sys.argv[3]))
