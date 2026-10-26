"""META-CALIBRATION across devices (Reptile over device-tasks, Calibration-GRPO as the inner learner).
Device population = one production batch: every device shares SYSTEMATIC errors (batch phase pattern, spacing bias,
typical amplitude imbalance) plus its own DEVICE-SPECIFIC errors:
   psi    = psi_batch (von Mises kappa=11, ~0.32 rad)  + N(0, 0.10^2) per element
   weak-state amplitude = 0.70 + N(0, 0.05^2) per layer        (strong state = 1.00)
   spacing = delta_batch (N(0, 3%)) + N(0, 1%) per layer
   lateral layer offset ~ N(0, 0.1 pitch) per device (assembly-specific, no shared part)
Plus CSI error (NMSE -10 dB) and 5% noisy rate feedback, as in calibration_full.py. (Magnitudes = assumptions.)
Meta-training: Reptile.  prior <- prior + eps * mean_devices(theta_after_k_updates - prior), fresh Adam per device.
Evaluation: held-out devices, calibrate from NOMINAL init vs from META prior, same budget; rate vs updates.
Usage: python3 meta_calibration.py train <outer_iters> <inner_updates> | eval <updates>"""
import sys, time, os, numpy as np, torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
K, M, L = 10, 64, 6
D = L * M
G, NCH, NREF = 32, 6, 20
S = dict(p=0.15, a=0.15, d=0.02, o=0.05)
LR = dict(p=0.012, a=0.01, d=0.003, o=0.005)
W = {}


def init_worker(threads=2):
    torch.set_num_threads(threads)
    from big_sys import make_env, channels
    from calibration_full import Geometry
    from env_ris import add_csi_estimation_error
    env = make_env(K, M, L, 0)
    W["geo"] = Geometry(env)
    W["P"] = torch.tensor(env.P, dtype=torch.float32); W["var"] = float(env.var)

    def est(Hs, seed):
        r = np.random.default_rng(seed)
        return torch.tensor(np.stack([add_csi_estimation_error(h.numpy().astype(complex), 0.316, r) for h in Hs]), dtype=torch.complex64)
    W["pool"] = channels(K, M, L, list(range(1000, 1020)), n_steps=100, stride=10); W["pool_h"] = est(W["pool"], 11)
    W["dev"] = channels(K, M, L, [4000, 4001], n_steps=100, stride=10)[:10]; W["dev_h"] = est(W["dev"], 12)
    W["test"] = channels(K, M, L, [5000, 5001, 5002, 5003], n_steps=100, stride=10)[:40]; W["test_h"] = est(W["test"], 13)


def batch_systematic():
    r = np.random.default_rng(7)
    return (torch.tensor(r.vonmises(0.0, 11.0, size=(L, M)), dtype=torch.float32),
            torch.tensor(r.normal(0, 0.03, L), dtype=torch.float32))


def device_params(seed):
    psi_b, dl_b = batch_systematic()
    r = np.random.default_rng(10_000 + seed)
    psi = psi_b + torch.tensor(r.normal(0, 0.10, (L, M)), dtype=torch.float32)
    weak = torch.tensor(0.70 + r.normal(0, 0.05, L), dtype=torch.float32).clamp(0.4, 1.0)
    dl = dl_b + torch.tensor(r.normal(0, 0.01, L), dtype=torch.float32)
    of = torch.tensor(r.normal(0, 0.1, (L, 2)), dtype=torch.float32)
    return dict(p=psi, a=torch.stack([torch.ones(L), weak], 1), d=dl, o=of)


def build(th):
    from calibration_full import Dev
    a = th["a"]
    return Dev(W["geo"].Ws(th["d"], th["o"]), th["p"], a[:, 0].clamp(0.05, 1.5), a[:, 1].clamp(0.05, 1.5), W["P"], W["var"])


def deploy_rate(model, true, H, Hh, steps=50, seed=1):
    from big_sys import sgn, relax
    from calibration_full import minrate
    w0 = torch.randn(H.shape[0], D, generator=torch.Generator().manual_seed(seed)) * 0.5
    return minrate(true, sgn(relax(model, Hh, w0, steps)), H)


def calibrate(args):
    """inner Calibration-GRPO on one device. args = (prior dict, device seed, updates, rng seed, eval_every, final_test)"""
    prior, dseed, U, seed, every, final_test = args
    if not W:
        init_worker()
    from big_sys import sgn, relax
    from calibration_full import minrate
    true = build(device_params(dseed))
    torch.manual_seed(seed)
    prm = {k: v.clone().requires_grad_(True) for k, v in prior.items()}
    opt = torch.optim.Adam([dict(params=[prm[k]], lr=LR[k]) for k in "pado"])
    curve = []
    for u in range(U + 1):
        if every and (u % every == 0 or u == U):
            curve.append((u, deploy_rate(build({k: v.detach() for k, v in prm.items()}), true, W["dev"], W["dev_h"]).mean().item()))
        if u == U: break
        idx = torch.randperm(W["pool"].shape[0])[:NCH]; Hb, Hbh = W["pool"][idx], W["pool_h"][idx]
        w0 = torch.randn(NCH, D) * 0.5
        e = {k: torch.randn((G // 2,) + tuple(prior[k].shape)) for k in "pado"}
        e = {k: torch.cat([v, -v], 0) for k, v in e.items()}
        R = torch.zeros(G)
        for g in range(G):
            cand = {k: prm[k].detach() + S[k] * e[k][g] for k in "pado"}
            r = minrate(true, sgn(relax(build(cand), Hbh, w0, NREF)), Hb)
            R[g] = (r * (1 + 0.05 * torch.randn_like(r))).mean()
        A = (R - R.mean()) / (R.std() + 1e-6)
        lp = 0
        for k in "pado":
            dims = tuple(range(1, e[k].dim()))
            lp = lp + (-0.5 * ((e[k] * S[k] - (prm[k] - prm[k].detach()).unsqueeze(0)) / S[k]) ** 2).sum(dims)
        loss = -(A * lp).mean() / D
        opt.zero_grad(); loss.backward(); opt.step()
    out = dict(theta={k: v.detach() for k, v in prm.items()}, curve=curve)
    if final_test:
        th = out["theta"]
        out["test"] = deploy_rate(build(th), true, W["test"], W["test_h"], 200).numpy()
        out["oracle"] = deploy_rate(true, true, W["test"], W["test_h"], 200).numpy()
        nom = dict(p=torch.zeros(L, M), a=torch.ones(L, 2), d=torch.zeros(L), o=torch.zeros(L, 2))
        out["nominal"] = deploy_rate(build(nom), true, W["test"], W["test_h"], 200).numpy()
        dp = device_params(dseed)
        out["err"] = dict(psi=torch.angle(torch.exp(1j * (th["p"] - dp["p"]))).pow(2).mean().sqrt().item(),
                          amp=(th["a"][:, 1] - dp["a"][:, 1]).abs().mean().item(),
                          spacing=(th["d"] - dp["d"]).abs().mean().item(), offset=(th["o"] - dp["o"]).abs().mean().item())
    return out


def nominal_prior():
    return dict(p=torch.zeros(L, M), a=torch.ones(L, 2), d=torch.zeros(L), o=torch.zeros(L, 2))


def train(T, k):
    prior = nominal_prior()
    ex = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"), initializer=init_worker, initargs=(2,))
    t0 = time.time()
    for t in range(T):
        eps = 1.0 - 0.7 * t / max(T - 1, 1)
        seeds = [100 + 4 * t + i for i in range(4)]                       # fresh training devices every iteration
        outs = list(ex.map(calibrate, [(prior, s, k, 1000 * t + i, 0, False) for i, s in enumerate(seeds)]))
        prior = {key: prior[key] + eps * torch.stack([o["theta"][key] - prior[key] for o in outs]).mean(0) for key in prior}
        psi_b, dl_b = batch_systematic()
        e_psi = torch.angle(torch.exp(1j * (prior["p"] - psi_b))).pow(2).mean().sqrt().item()
        print(f"outer {t+1}/{T} eps={eps:.2f} | prior vs batch: psi err {e_psi:.3f} rad (start 0.316), weak amp {prior['a'][:,1].mean():.3f} (batch 0.70), "
              f"spacing err {100*(prior['d']-dl_b).abs().mean():.2f}% | {(time.time()-t0)/60:.1f} min", flush=True)
        torch.save(prior, "meta_calib_prior.pt")
    print("DONE", flush=True)


def evaluate(U):
    prior = torch.load("meta_calib_prior.pt")
    jobs, names = [], []
    for ds in (900, 901, 902):
        for nm, pr in (("nominal-init", nominal_prior()), ("meta-init", prior)):
            jobs.append((pr, ds, U, 77, 10, True)); names.append((ds, nm))
    ex = ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn"), initializer=init_worker, initargs=(1,))
    outs = list(ex.map(calibrate, jobs))
    res = {f"{ds}_{nm}": o for (ds, nm), o in zip(names, outs)}
    np.save("meta_calib_eval.npy", np.array(res, dtype=object))
    print("held-out devices: dev-channel rate vs calibration updates (G=32 measurements x 6 channels per update)")
    us = [u for u, _ in outs[0]["curve"]]
    print("device arm          " + " ".join(f"u={u:<4d}" for u in us))
    for (ds, nm), o in zip(names, outs):
        print(f"{ds} {nm:14s} " + " ".join(f"{r:.3f} " for _, r in o["curve"]))
    print("\nTEST (40 channels, 200 steps):")
    for (ds, nm), o in zip(names, outs):
        print(f"{ds} {nm:14s} calibrated {o['test'].mean():.3f} | nominal {o['nominal'].mean():.3f} | oracle {o['oracle'].mean():.3f} | errs {({k: round(v,3) for k,v in o['err'].items()})}")
    print("DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "eval":
        evaluate(int(sys.argv[2]))
    else:   # smoke
        init_worker(4)
        o = calibrate((nominal_prior(), 900, 2, 0, 1, True))
        print(o["curve"], o["test"].mean(), o["nominal"].mean(), o["oracle"].mean(), o["err"])
