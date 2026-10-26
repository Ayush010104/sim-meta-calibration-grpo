"""Paper run, single copy (N=1). Usage:  python3 run_paper_n1.py tune | main
tune : small hyper-parameter grids on the 2 held-out seeds only (headline cell, U=10) -> tuned_hp.json
main : block 1 (headline cell A, U=10, 10 seeds, 6 algos), block 3 (cells B-E, U=10, 5 seeds, 5 algos, no COLD),
       block 2 (cell A, U=50, 5 seeds, 6 algos). Everything checkpointed per run; safe to restart.
Each run = one independent 1000-step trajectory (own geometry/direction/scattering draw)."""
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

RES, LOG, HPFILE = "paper_n1_results.npy", "run_paper_n1.log", "tuned_hp.json"
N_STEPS, WORKERS = 1000, 7
TUNE_SEEDS = [45678, 99999]
HEAD_SEEDS = [12345, 23456, 34567, 56789, 67890, 78901, 89012, 90123, 11111, 22222]
CELLS = {"A": (0.99, 1.4), "B": (0.9, 1.4), "C": (0.7, 1.4), "D": (0.99, 0.5), "E": (0.99, 3.0)}
ALGOS6 = ("seq_warm", "cold", "reptile", "periodic_ema", "prox", "fomaml")


def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")


def hpk(hp):
    return tuple(sorted(hp.items()))


def key_of(task):
    rho, v, seed, algo, U, hp = task
    return (rho, v, seed, algo, U, hpk(hp))


def work(task):
    from n1_learners import run_learner, generate_channel_window, T_S
    rho, v, seed, algo, U, hp = task
    H = generate_channel_window(seed, rho, v, T_S, N_STEPS)
    return task, run_learner(algo, H, seed, rho, v, U, hp)


def tune_tasks():
    rho, v = CELLS["A"]
    T = []
    for s in TUNE_SEEDS:
        T += [(rho, v, s, "seq_warm", 10, {}), (rho, v, s, "cold", 10, {})]
        T += [(rho, v, s, "reptile", 10, {"eta": e}) for e in (0.1, 0.3, 0.5, 1.0)]
        T += [(rho, v, s, "periodic_ema", 10, {"eta": e}) for e in (0.1, 0.3, 0.5)]
        T += [(rho, v, s, "prox", 10, {"eta": 0.3, "alpha_prox": a}) for a in (0.02, 0.05, 0.1, 0.2)]
        T += [(rho, v, s, "fomaml", 10, {"beta": b, "m": m}) for b in (3e-4, 1e-3, 3e-3) for m in (1, 3)]
    return T


def select_hp(res):
    best = {}
    by = {}
    for (rho, v, seed, algo, U, hp), r in res.items():
        if seed in TUNE_SEEDS and U == 10 and algo not in ("seq_warm", "cold"):
            by.setdefault((algo, hp), []).append(np.mean(r["minrs"]))
    for (algo, hp), vals in by.items():
        if len(vals) == len(TUNE_SEEDS) and (algo not in best or np.mean(vals) > best[algo][0]):
            best[algo] = (float(np.mean(vals)), dict(hp))
    return {a: b[1] for a, b in best.items()}, {a: b[0] for a, b in best.items()}


def main_tasks(hp):
    h = lambda a: hp.get(a, {})
    T = []
    rho, v = CELLS["A"]
    for seed in HEAD_SEEDS:                                    # block 1
        T += [(rho, v, seed, a, 10, h(a)) for a in ALGOS6]
    for c in "BCDE":                                           # block 3
        rho, v = CELLS[c]
        for seed in HEAD_SEEDS[:5]:
            T += [(rho, v, seed, a, 10, h(a)) for a in ALGOS6 if a != "cold"]
    rho, v = CELLS["A"]
    for seed in HEAD_SEEDS[:5]:                                # block 2 (U=50)
        T += [(rho, v, seed, a, 50, h(a)) for a in ALGOS6]
    return T


def run_queue(tasks, res):
    todo = [t for t in tasks if key_of(t) not in res]
    log(f"queue: {len(tasks)} runs, {len(tasks) - len(todo)} done, {len(todo)} to go, {WORKERS} workers")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for f in as_completed(futs):
            task, r = f.result()
            res[key_of(task)] = r
            np.save(RES, np.array(res, dtype=object))
            done += 1
            rho, v, seed, algo, U, hp = task
            m = float(np.mean(r["minrs"]))
            log(f"[{done}/{len(todo)}] rho={rho} v={v} seed={seed} {algo:12s} U={U} hp={hp} mean={m:.4f} "
                f"resets={r['n_resets']} {r['wall']:.0f}s  (elapsed {(time.time()-t0)/60:.0f} min)")
    return res


def main():
    mode = sys.argv[1]
    res = np.load(RES, allow_pickle=True).item() if os.path.exists(RES) else {}
    if mode == "tune":
        res = run_queue(tune_tasks(), res)
        hp, score = select_hp(res)
        json.dump(hp, open(HPFILE, "w"), indent=1)
        log("TUNED (mean min-rate over the 2 tuning seeds): " + json.dumps({a: (hp[a], round(score[a], 4)) for a in hp}))
    elif mode == "main":
        hp = json.load(open(HPFILE))
        log("using tuned hp: " + json.dumps(hp))
        res = run_queue(main_tasks(hp), res)
        log("MAIN DONE")


if __name__ == "__main__":
    main()
