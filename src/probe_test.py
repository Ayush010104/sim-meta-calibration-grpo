"""CSI-probe test (normal-operation loop). K=10, M=64, L=6, moderate device (impair_lib seed 7). The device model is assumed
already calibrated (we use the modeled-class true parameters) so that ONLY the CSI problem remains.
CSI available at time t: the channel of time t-1 (aging under user mobility) + estimation noise (NMSE -10 dB).
Every channel update, each method may probe S configurations (apply + read the users' rates, 5% noisy) and deploys the best probed.
  A  no-probe     : 1 design on the estimate, deployed blindly (0 probes)
  B  restart-S    : S designs from random starts, best measured deployed                     (no learning)
  D  warm-S       : S designs started near the previously deployed solution + fixed noise   (no learning)
  C  GRPO-probe   : like D, but each candidate's knobs (channel-prediction coefficient lambda, warm-start weight c,
                    start spread s) are sampled from a policy updated online by GRPO from the probed rates
  O  oracle CSI   : 1 design on the true current channel (upper reference, 0 probes)
Metric: true worst-user rate of the deployed configuration, averaged over time.  Usage: python3 probe_test.py <S> <T>"""
import sys, time, numpy as np, torch
torch.set_num_threads(2)
from big_sys import make_env, sgn, relax
from impair_lib import System, Dev, LEVELS, minrate
K, M, L = 10, 64, 6; D = L * M


def traj(seed, T):
    e = make_env(K, M, L, seed)
    H = [e.H_true.copy()]
    for _ in range(T + 1):
        e.advance_channel(); H.append(e.H_true.copy())
    return torch.tensor(np.stack(H), dtype=torch.complex64)


def noisy(H, db, g):
    eps = float(np.sqrt(10 ** (db / 10)))
    nrm = H.abs().pow(2).sum(0, keepdim=True).sqrt()                    # per-user column norm (as env_ris)
    n = (torch.randn(H.shape, generator=g) + 1j * torch.randn(H.shape, generator=g)) / np.sqrt(2)
    return H + n.to(torch.complex64) * eps * nrm / np.sqrt(H.shape[0])


def run(method, Hs, model, true, S, T, seed):
    g = torch.Generator().manual_seed(seed)
    w_prev = torch.randn(D, generator=g) * 0.5
    mu = torch.tensor([0.0, 2.0, float(np.log(0.3))], requires_grad=True)       # lambda, logit c, log s
    sd = torch.tensor([0.3, 0.5, 0.3])
    opt = torch.optim.Adam([mu], lr=0.05)
    rates, knobs = [], []
    Hh_prev = noisy(Hs[0], -10, g)
    for t in range(1, T + 1):
        Ht = Hs[t + 1]                                                    # true channel now
        Hh = noisy(Hs[t], -10, g)                                         # estimate available = aged + noisy
        if method == "O":
            x = sgn(relax(model, Ht[None], (torch.randn(1, D, generator=g) * 0.5), 100))
            rates.append(minrate(true, x, Ht[None]).item()); continue
        if method == "A":
            x = sgn(relax(model, Hh[None], (torch.randn(1, D, generator=g) * 0.5), 100))
            rates.append(minrate(true, x, Ht[None]).item()); Hh_prev = Hh; continue
        wn = 0.5 * w_prev / (w_prev.pow(2).mean().sqrt() + 1e-9)
        if method == "B":
            k = torch.tensor([0.0, -20.0, float(np.log(0.5))]).expand(S, 3)
        elif method == "D":
            k = mu.detach().expand(S, 3)
        else:
            eps = torch.randn(S // 2, 3, generator=g); eps = torch.cat([eps, -eps], 0)
            k = mu.detach() + sd * eps
        lam, c, s = k[:, 0], torch.sigmoid(k[:, 1]), k[:, 2].exp()
        Hused = Hh.unsqueeze(0) + lam.view(S, 1, 1) * (Hh - Hh_prev).unsqueeze(0)       # learned channel prediction
        w0 = c.view(S, 1) * wn.unsqueeze(0) + s.view(S, 1) * torch.randn(S, D, generator=g)
        w = relax(model, Hused, w0, 100)
        x = sgn(w)
        r_true = minrate(true, x, Ht[None].expand(S, -1, -1))
        r_meas = r_true * (1 + 0.05 * torch.randn(S, generator=g))
        b = int(r_meas.argmax())
        rates.append(r_true[b].item()); w_prev = w[b].detach(); Hh_prev = Hh
        if method == "C":
            A = (r_meas - r_meas.mean()) / (r_meas.std() + 1e-6)
            lp = (-0.5 * ((k - mu) / sd) ** 2).sum(1)
            loss = -(A * lp).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            knobs.append(mu.detach().clone().numpy())
    return np.array(rates), np.array(knobs)


def main():
    S, T = int(sys.argv[1]), int(sys.argv[2])
    Sy = System(K, M, L)
    th, ae, st = Sy.sample(LEVELS["moderate"], 7)
    true = Dev(Sy, th); model = true                                    # calibrated model = modeled-class truth
    seeds = [5000, 5001, 5002, 5003]
    trajs = [traj(sd, T) for sd in seeds]
    ag = [float(((tr[1:] - tr[:-1]).abs().pow(2).sum() / tr[1:].abs().pow(2).sum()).item()) for tr in trajs]
    print(f"channel aging per step: NMSE(H_t vs H_t-1) = {10*np.log10(np.mean(ag)):.1f} dB; estimation noise -10 dB; S={S} probes/update, T={T}", flush=True)
    res = {}
    for m in ("O", "A", "B", "D", "C"):
        t0 = time.time()
        out = [run(m, tr, model, true, S, T, 11 + i) for i, tr in enumerate(trajs)]
        res[m] = dict(rates=np.stack([o[0] for o in out]), knobs=[o[1] for o in out])
        r = res[m]["rates"]
        print(f"{m}: mean true rate {r.mean():.3f}  (per trajectory {np.round(r.mean(1),3).tolist()}, first half {r[:, :T//2].mean():.3f}, second half {r[:, T//2:].mean():.3f})  {time.time()-t0:.0f}s", flush=True)
        if m == "C":
            kn = np.stack([o[1][-1] for o in out])
            print(f"   learned knobs (end, per trajectory): lambda {np.round(kn[:,0],2).tolist()}, c {np.round(1/(1+np.exp(-kn[:,1])),2).tolist()}, s {np.round(np.exp(kn[:,2]),2).tolist()}", flush=True)
    for a, b in (("C", "D"), ("C", "B"), ("C", "A"), ("D", "B")):
        d = (res[a]["rates"] - res[b]["rates"]).mean(1)
        print(f"{a} - {b}: per-trajectory mean diff {np.round(d,3).tolist()}  overall {d.mean():+.3f}", flush=True)
    np.save(f"probe_test_S{S}.npy", np.array(res, dtype=object))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
