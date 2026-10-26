"""Multi-seed x multi-rho x multi-speed robustness sweep for the IEEE
Transactions paper: seq-WARM vs ENSEMBLE-META (proposed, N=10) across a
3x3x3 grid, so results can be reported as mean +/- std across seeds rather
than single-seed point estimates.

Grid: seeds=[12345, 23456, 34567], rho=[0.7, 0.9, 0.99],
      v=[0.5, 1.4, 3.0] m/s, T_s=0.5s fixed, N_STEPS=1000 (matching the
      main paper result's horizon), N_COPIES=10, stagger=50,
      sim_threshold=0.8, meta_lr=0.3.

27 combos x (ENSEMBLE-META ~17min + seq-WARM ~6min) ~= 10.3 hours total.
Checkpointed per-combo so an interruption only loses the in-progress combo.
"""
import os
import time
import math

import numpy as np
import torch

torch.set_num_threads(1)

from moving_user_env import MovingUserEnv
from ensemble_parallel import run_ensemble_parallel
from grpo_ris import build_best_agent, update_quantized_best, quantize_phase_hard_torch
from physics import greedy_bitflip_local_search

BASE_CFG = dict(num_antennas=5, num_RIS_elements=36, num_users=5, num_layers=4, dBS=30.0, radius=50.0)
T_S = 0.5
N_STEPS = 1000
N_COPIES = 10
STAGGER = 50
SIM_THRESHOLD = 0.8
META_LR = 0.3
N_BITS = 1

SEEDS = [12345, 23456, 34567]
RHOS = [0.7, 0.9, 0.99]
SPEEDS = [0.5, 1.4, 3.0]

LOG_FILE = "run_robustness_sweep.log"
RESULTS_FILE = "robustness_sweep_results.npy"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def generate_channel_window(seed, rho, v, t_s, n_steps):
    env = MovingUserEnv(seed=seed, temporal_rho=rho, user_speed=v, T_s=t_s, direction_seed=seed, **BASE_CFG)
    H_seq = [env.H_true.copy()]
    for _ in range(n_steps - 1):
        env.advance_channel()
        H_seq.append(env.H_true.copy())
    return H_seq


def run_seq_warm(H_seq, seed, updates_per_step=10, group_size=8):
    env = MovingUserEnv(seed=seed, temporal_rho=0.99, user_speed=1.4, T_s=T_S, direction_seed=seed, **BASE_CFG)
    agent = build_best_agent(env, torch.device("cpu"), n_bits=N_BITS, group_size=group_size,
                              ppo_epochs=5, entropy_anneal_updates=len(H_seq) * updates_per_step,
                              seed=seed, sigma_floor_frac=0.18, log_std_init_frac=0.35)
    minrs = []
    for t, H in enumerate(H_seq):
        env.H_true = H
        state = env.reset_state_only()
        best_minr, best_phi = -1e9, None
        for _ in range(updates_per_step):
            update_quantized_best(agent, [state] * group_size, [H] * group_size, n_bits=N_BITS, reparam_coef=1.0)
            with torch.no_grad():
                theta_mu, _ = agent.actor.forward(torch.FloatTensor(state).unsqueeze(0))
                phi_q = quantize_phase_hard_torch(theta_mu, N_BITS)
                action_q = agent.actor._phi_to_action(phi_q).numpy()
            _, r, minr, _, info = env.step(action_q)
            if minr > best_minr:
                best_minr, best_phi = minr, phi_q.numpy()[0].copy()
        _, polished_minr, _, _ = greedy_bitflip_local_search(best_phi, env, N_BITS, max_sweeps=20)
        minrs.append(polished_minr)
    return minrs


if __name__ == "__main__":
    resuming = os.path.exists(RESULTS_FILE)
    if not resuming:
        open(LOG_FILE, "w").close()
    results = np.load(RESULTS_FILE, allow_pickle=True).item() if resuming else {}
    log(f"ROBUSTNESS SWEEP: seeds={SEEDS}, rho={RHOS}, v={SPEEDS}, N_STEPS={N_STEPS}"
        f"{'  [RESUMING, ' + str(len(results)) + ' combos done]' if resuming else ''}")

    combos = [(seed, rho, v) for seed in SEEDS for rho in RHOS for v in SPEEDS]
    grand_t0 = time.time()
    n_done_this_session = 0

    for seed, rho, v in combos:
        key = (seed, rho, v)
        if key in results:
            continue
        t0 = time.time()
        H_seq = generate_channel_window(seed, rho, v, T_S, N_STEPS)

        seqw_minrs = run_seq_warm(H_seq, seed)
        sw_mean = float(np.mean(seqw_minrs))

        meta_minrs, reset_log = run_ensemble_parallel(H_seq, seed, rho, n_bits=N_BITS, base_cfg=BASE_CFG,
                                                       v=v, t_s=T_S, n_copies=N_COPIES, stagger=STAGGER,
                                                       sim_threshold=SIM_THRESHOLD, reset_mode="meta",
                                                       verbose=False)
        meta_mean = float(np.mean(meta_minrs))

        results[key] = dict(seq_warm=seqw_minrs, ensemble_meta=meta_minrs, reset_log=reset_log)
        np.save(RESULTS_FILE, np.array(results, dtype=object))

        n_done_this_session += 1
        elapsed = time.time() - t0
        total_elapsed = time.time() - grand_t0
        remaining = len(combos) - len(results)
        log(f"[seed={seed}, rho={rho}, v={v}] seq-WARM={sw_mean:.4f}  META={meta_mean:.4f}  "
            f"pct={100*meta_mean/sw_mean:.1f}%  resets={len(reset_log)}  took={elapsed:.0f}s  "
            f"(done {len(results)}/{len(combos)}, session elapsed {total_elapsed:.0f}s, "
            f"~{remaining * elapsed / 60:.0f} min remaining at this rate)")

    log(f"\nSWEEP COMPLETE: {len(results)}/{len(combos)} combos")
    log(f"\n{'rho':>6}{'v':>6}{'seqW_mean':>12}{'seqW_std':>10}{'META_mean':>12}{'META_std':>10}{'pct_mean':>10}")
    for rho in RHOS:
        for v in SPEEDS:
            sw_vals = [np.mean(results[(s, rho, v)]["seq_warm"]) for s in SEEDS if (s, rho, v) in results]
            meta_vals = [np.mean(results[(s, rho, v)]["ensemble_meta"]) for s in SEEDS if (s, rho, v) in results]
            if not sw_vals:
                continue
            pct_vals = [100 * m / s for m, s in zip(meta_vals, sw_vals)]
            log(f"{rho:>6}{v:>6}{np.mean(sw_vals):>12.4f}{np.std(sw_vals):>10.4f}"
                f"{np.mean(meta_vals):>12.4f}{np.std(meta_vals):>10.4f}{np.mean(pct_vals):>9.1f}%")
