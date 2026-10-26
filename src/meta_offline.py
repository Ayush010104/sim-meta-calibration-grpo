"""Offline meta-training over a DISTRIBUTION of channels (tasks = channels), then few-shot adaptation
on HELD-OUT channels. This is the standard meta-learning setting (the online-drift setting gives
meta-learning almost nothing to do: the previous weights are already the best prior for the next,
nearly identical, task).

Policy context = channel only (phase part of the state fixed to the zero configuration), so the
policy is a pure function of the channel and offline-trained weights transfer.

Meta-algorithm: Reptile over channel tasks. One meta-iteration: sample B training channels; for each,
start from theta_meta, run k inner GRPO updates (fresh Adam), then
theta_meta += eps * mean_i(theta_i - theta_meta)   (eps annealed).
Evaluation (held-out channels from unseen trajectories): after u in {0,1,2,5,10} inner updates, the min-rate of
 - final quantized mean config            ('final'),
 - best-of-u quantized mean configs       ('best'),  and the same after the bit-flip polish ('*_post'),
for RANDOM init vs the META init.
Usage: python3 meta_offline.py train   |   python3 meta_offline.py smoke
"""
import copy
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

torch.set_num_threads(1)

G = 8
UEVAL = (0, 1, 2, 5, 10)
SCALE = 1.0 / 4.0e-5
_ENV = None


def env():
    global _ENV
    if _ENV is None:
        from moving_user_env import MovingUserEnv
        from run_robustness_sweep import BASE_CFG, T_S
        _ENV = MovingUserEnv(seed=0, temporal_rho=0.99, user_speed=1.4, T_s=T_S, direction_seed=0, **BASE_CFG)
        if os.environ.get("HARD_SIGMA"):   # hard config: fixed unknown element phase offsets psi (see hard_cfg.py)
            g = torch.Generator().manual_seed(int(os.environ.get("HARD_SEED", 0)))
            psi = (torch.randn(4, 36, generator=g) * float(os.environ["HARD_SIGMA"])).numpy().astype(np.float64)
            _ENV.Ws = [np.exp(1j * psi[l])[:, None] * _ENV.Ws[l] for l in range(4)]
    return _ENV


def ctx_state(H, n=144):
    return np.concatenate([H.real.ravel() * SCALE, H.imag.ravel() * SCALE, np.ones(n), np.zeros(n)])


def new_agent(seed, theta=None):
    from grpo_ris import build_best_agent
    a = build_best_agent(env(), torch.device("cpu"), n_bits=1, group_size=G, ppo_epochs=5,
                         entropy_anneal_updates=10 ** 6, seed=seed, sigma_floor_frac=0.18, log_std_init_frac=0.35)
    if theta is not None:
        a.actor.load_state_dict(theta)
        a.actor_old.load_state_dict(theta)
    return a


def quant_cfg(a, s):
    from grpo_ris import quantize_phase_hard_torch
    with torch.no_grad():
        mu, _ = a.actor.forward(torch.FloatTensor(s).unsqueeze(0))
        return quantize_phase_hard_torch(mu, 1).numpy()[0]


def minr(phi, H):
    from physics import eval_minr_batch
    e = env()
    e.H_true = H
    return float(eval_minr_batch(phi[None, :], e)[0])


def polish(phi, H, sweeps=20):
    from physics import greedy_bitflip_local_search
    e = env()
    e.H_true = H
    return float(greedy_bitflip_local_search(phi, e, 1, max_sweeps=sweeps)[1])


def inner(args):
    theta, H, k, seed = args
    from grpo_ris import update_quantized_best
    a = new_agent(seed, theta)
    s = ctx_state(H)
    for _ in range(k):
        update_quantized_best(a, [s] * G, [H] * G, n_bits=1, reparam_coef=1.0)
    return {kk: v.clone() for kk, v in a.actor.state_dict().items()}


def eval_one(args):
    theta, H, seed = args
    from grpo_ris import update_quantized_best
    a = new_agent(seed, theta)
    s = ctx_state(H)
    out, best, best_phi, done = {}, -1e9, None, 0
    for u in UEVAL:
        while done < u:
            update_quantized_best(a, [s] * G, [H] * G, n_bits=1, reparam_coef=1.0)
            done += 1
            phi = quant_cfg(a, s)
            m = minr(phi, H)
            if m > best:
                best, best_phi = m, phi
        phi = quant_cfg(a, s)
        fin = minr(phi, H)
        if u == 0:
            best, best_phi = fin, phi
        out[u] = dict(final=fin, final_post=polish(phi, H), best=best, best_post=polish(best_phi, H),
                      best_post2=polish(best_phi, H, 2))
    return out


def make_tasks(seeds, cells, n_steps=400, stride=8):
    from run_robustness_sweep import generate_channel_window, T_S
    T = []
    for sd in seeds:
        rho, v = cells[sd % len(cells)]
        H = generate_channel_window(sd, rho, v, T_S, n_steps)
        T += [H[t] for t in range(0, n_steps, stride)]
    return T


def evaluate(ex, theta, tests, tag):
    res = list(ex.map(eval_one, [(theta, H, 777 + i) for i, H in enumerate(tests)]))
    row = {u: {m: float(np.mean([r[u][m] for r in res])) for m in ("final", "final_post", "best", "best_post", "best_post2")} for u in UEVAL}
    print(f"  [{tag}] " + " | ".join(f"u={u}: pre={row[u]['best']:.3f} post2={row[u]['best_post2']:.3f} post20={row[u]['best_post']:.3f}" for u in UEVAL), flush=True)
    return row


def main():
    mode = sys.argv[1]
    smoke = mode == "smoke"
    CELLS = [(r, v) for r in (0.7, 0.9, 0.99) for v in (0.5, 1.4, 3.0)]
    train_seeds = list(range(1000, 1000 + (6 if smoke else 60)))
    test_seeds = [5000 + i for i in range(5)]
    tr = make_tasks(train_seeds, CELLS)
    te = make_tasks(test_seeds, [(0.99, 1.4)], stride=20 if not smoke else 100)[: (4 if smoke else 100)]
    print(f"train channels {len(tr)}, held-out test channels {len(te)} (rho=0.99, v=1.4, unseen seeds)", flush=True)
    iters, B, k = (6, 4, 3) if smoke else (2500, 14, 10)
    ex = ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn"))
    base = new_agent(0)
    theta = {kk: v.clone() for kk, v in base.actor.state_dict().items()}
    rng = np.random.default_rng(0)
    log = {}
    t0 = time.time()
    log["random"] = evaluate(ex, None, te, "random init")
    marks = {0, 3} if smoke else {250, 500, 1000, 1500, 2500}
    for it in range(1, iters + 1):
        eps = 0.5 + (0.05 - 0.5) * it / iters
        idx = rng.choice(len(tr), B, replace=False)
        outs = list(ex.map(inner, [(theta, tr[i], k, int(rng.integers(1 << 30))) for i in idx]))
        for kk in theta:
            if theta[kk].is_floating_point():
                theta[kk] = theta[kk] + eps * (torch.stack([o[kk] for o in outs]).mean(0) - theta[kk])
        if it % 50 == 0:
            print(f"iter {it}/{iters}  eps={eps:.2f}  {(time.time()-t0)/60:.1f} min", flush=True)
        if it in marks or (smoke and it == iters):
            log[it] = evaluate(ex, theta, te, f"meta @ iter {it}")
            torch.save(theta, "meta_theta.pt")
            np.save("meta_offline_log.npy", np.array(log, dtype=object))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
