"""Stage-1 screening of alternative RESET-INITIALIZATION rules for the staggered
GRPO ensemble (everything else -- N copies, stagger, change-point trigger,
best-of-N deployment, worker/step logic -- is identical to ensemble_enhanced):

  meta    -- reference: Reptile EMA of harvested weights into one shared prior
             (meta_state += meta_lr * (harvest - meta_state)).
  leader  -- PBT-style "exploit": a resetting copy clones the weights+optimizer
             of the best live copy that is NOT itself resetting this step.
             No averaging, so no weight-space basin mixing.
  attn    -- QKV-style retrieval, no training. Memory of past harvested
             snapshots (V = weights+optimizer) with keys K = normalized
             channel signature at harvest time; query Q = current channel.
             logits_i = (<Q,K_i> - 1)/tau + beta * zscore(quality_i); HARD
             attention (argmax) so the copy starts from one coherent network.

Also: run_ls_track -- a no-RL control that warm-starts greedy bit-flip local
search from the previous step's solution. Any claim that RL helps has to
beat this.
"""
import copy
import multiprocessing as mp
import time

import numpy as np
import torch

from ensemble_enhanced import _copy_worker, UPDATES_PER_STEP, GROUP_SIZE
from moving_user_env import MovingUserEnv
from physics import greedy_bitflip_local_search


def _signature(H):
    v = H.flatten()
    return v / (np.linalg.norm(v) + 1e-30)


def _mix_states(states, w):
    """Convex combination of state dicts (float tensors); non-float entries from the top-weighted one."""
    out, top = {}, int(np.argmax(w))
    for k in states[0]:
        if states[0][k].is_floating_point():
            out[k] = sum(float(wi) * st[k] for wi, st in zip(w, states))
        else:
            out[k] = states[top][k].clone()
    return out


def _softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def _copy_worker_logged(cid, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold, conn, h_scale=None, polish=True, prox=0.0):
    """Identical to ensemble_enhanced._copy_worker, except step also returns the polished phase config."""
    torch.set_num_threads(1)
    from moving_user_env import MovingUserEnv, channel_similarity
    from grpo_ris import build_best_agent, update_quantized_best, quantize_phase_hard_torch
    from physics import greedy_bitflip_local_search

    def make_env():
        return MovingUserEnv(seed=seed + cid, temporal_rho=rho, user_speed=v, T_s=t_s,
                             direction_seed=seed + cid, **base_cfg)

    env = agent = H_ref = None
    anchor = None
    recent_minrs = []
    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "activate":
            _, meta_state, meta_opt_state, t, explore_reset, log_std_widen = msg
            env = make_env()
            agent = build_best_agent(env, torch.device("cpu"), n_bits=n_bits, group_size=GROUP_SIZE,
                                     ppo_epochs=5, entropy_anneal_updates=1000 * UPDATES_PER_STEP,
                                     seed=seed + cid + t, sigma_floor_frac=0.18, log_std_init_frac=0.35)
            anchor = None
            if meta_state is not None:
                agent.actor.load_state_dict(meta_state)
                agent.actor_old.load_state_dict(meta_state)
                agent.actor_optimizer.load_state_dict(meta_opt_state)
                if prox > 0:
                    anchor = {n: meta_state[n].clone() for n, _ in agent.actor.named_parameters()}
            H_ref = None
            recent_minrs = []
            conn.send("activated")
        elif cmd == "step":
            _, H = msg
            env.H_true = H
            if h_scale is not None:          # channel-AWARE policy input (rescaled, refreshed every step)
                env.H_obs = H * h_scale
            state = env.reset_state_only()
            best_minr, best_phi = -1e9, None
            upd = []                    # deployed-config min-rate after EVERY update (pre-polish)
            for _ in range(UPDATES_PER_STEP):
                update_quantized_best(agent, [state] * GROUP_SIZE, [H] * GROUP_SIZE,
                                      n_bits=n_bits, reparam_coef=1.0)
                if anchor is not None:      # proximal (iMAML-style) pull toward the meta prior
                    with torch.no_grad():
                        for n_, p_ in agent.actor.named_parameters():
                            p_.sub_(prox * (p_ - anchor[n_]))
                with torch.no_grad():
                    theta_mu, _ = agent.actor.forward(torch.FloatTensor(state).unsqueeze(0))
                    phi_q = quantize_phase_hard_torch(theta_mu, n_bits)
                    action_q = agent.actor._phi_to_action(phi_q).numpy()
                _, r, minr, _, info = env.step(action_q)
                upd.append(float(minr))
                if minr > best_minr:
                    best_minr, best_phi = minr, phi_q.numpy()[0].copy()
            if polish:
                polished_phi, polished_minr, _, _ = greedy_bitflip_local_search(best_phi, env, n_bits, max_sweeps=20)
            else:
                polished_phi, polished_minr = best_phi, best_minr
            recent_minrs.append(polished_minr)
            if len(recent_minrs) > 5:
                recent_minrs.pop(0)
            if H_ref is None:
                H_ref = H
            should_reset = channel_similarity(H_ref, H) < sim_threshold
            conn.send((polished_minr, should_reset, np.array(polished_phi), upd))
        elif cmd == "harvest":
            task_state = {k: v_.clone() for k, v_ in agent.actor.state_dict().items()}
            task_opt_state = copy.deepcopy(agent.actor_optimizer.state_dict())
            probe_state = env.reset_state_only()
            avg_recent_minr = float(np.mean(recent_minrs)) if recent_minrs else 0.0
            conn.send((task_state, task_opt_state, probe_state, avg_recent_minr))
        elif cmd == "stop":
            conn.close()
            break


def run_ensemble_variants(H_seq_full, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold=0.8,
                          n_copies=10, stagger=50, reset_init="meta", meta_lr=0.3,
                          attn_tau=0.02, attn_beta=1.0, memory_size=30, soft_tau=0.1, soft_beta=0.0,
                          log_phi=False, aware=False, polish=True, prox=0.0, verbose=False):
    assert reset_init in ("meta", "leader", "attn", "soft", "dumb")
    worker = _copy_worker_logged if (log_phi or aware or not polish or prox > 0) else _copy_worker
    h_scale = 1.0 / np.abs(H_seq_full[0]).mean() if aware else None
    ctx = mp.get_context("spawn")
    parent_conns, procs = [], []
    for cid in range(n_copies):
        parent_conn, child_conn = ctx.Pipe()
        extra = (h_scale, polish, prox) if (log_phi or aware or not polish or prox > 0) else ()
        p = ctx.Process(target=worker,
                        args=(cid, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold, child_conn) + extra)
        p.start()
        parent_conns.append(parent_conn)
        procs.append(p)

    def activate(cid, t, state, opt_state):
        parent_conns[cid].send(("activate", state, opt_state, t, False, 0.3))
        parent_conns[cid].recv()

    def harvest(cid):
        parent_conns[cid].send(("harvest",))
        return parent_conns[cid].recv()  # task_state, task_opt_state, probe_state, avg_recent_minr

    diag = dict(retrieval_age=[], retrieved_is_latest=[], soft_eff_n=[], phi=[], copy_minr=[])
    try:
        active = [False] * n_copies
        meta_state, meta_opt_state = None, None
        memory = []
        minrs, reset_log = [], []

        for t, H in enumerate(H_seq_full):
            for cid in range(n_copies):
                if not active[cid] and t >= cid * stagger:
                    if reset_init == "leader" and any(active):
                        live = [c for c in range(n_copies) if active[c]]
                        ls, lo, _, _ = harvest(live[-1])
                        activate(cid, t, ls, lo)
                    else:
                        activate(cid, t, meta_state, meta_opt_state)
                    active[cid] = True

            active_ids = [c for c in range(n_copies) if active[c]]
            for cid in active_ids:
                parent_conns[cid].send(("step", H))
            results = {cid: parent_conns[cid].recv() for cid in active_ids}

            best_cid = max(active_ids, key=lambda c: results[c][0])
            minrs.append(results[best_cid][0])
            diag["copy_minr"].append([results[c][0] if c in results else np.nan for c in range(n_copies)])
            if log_phi:
                diag["phi"].append(results[best_cid][2])
                diag.setdefault("upd", []).append(results[best_cid][3])

            reset_ids = [c for c in active_ids if results[c][1]]
            if not reset_ids:
                continue

            if reset_init == "leader":
                pool = [c for c in active_ids if c not in reset_ids] or active_ids
                leader = max(pool, key=lambda c: results[c][0])
                ls, lo, _, _ = harvest(leader)
                for cid in reset_ids:
                    activate(cid, t, ls, lo)
                    reset_log.append((t, cid))

            elif reset_init == "dumb":      # control: random re-initialization on every reset
                for cid in reset_ids:
                    activate(cid, t, None, None)
                    reset_log.append((t, cid))

            elif reset_init == "meta":
                for cid in reset_ids:
                    task_state, task_opt, _, _ = harvest(cid)
                    if meta_state is None:
                        meta_state = task_state
                    else:
                        meta_state = {k: meta_state[k] + meta_lr * (task_state[k] - meta_state[k])
                                      for k in meta_state}
                    meta_opt_state = task_opt
                    activate(cid, t, meta_state, meta_opt_state)
                    reset_log.append((t, cid))

            elif reset_init == "soft":
                q = _signature(H)
                for cid in reset_ids:
                    task_state, task_opt, _, qual = harvest(cid)
                    entry = dict(key=q, state=task_state, opt=task_opt, qual=qual, t=t)
                    pool = memory + [entry]
                    sims = np.array([abs(np.vdot(q, e["key"])) for e in pool])
                    quals = np.array([e["qual"] for e in pool])
                    z = (quals - quals.mean()) / (quals.std() + 1e-6)
                    w = _softmax((sims - 1.0) / soft_tau + soft_beta * z)
                    diag["soft_eff_n"].append(float(1.0 / np.sum(w ** 2)))
                    init_state = _mix_states([e["state"] for e in pool], w)
                    memory.append(entry)
                    if len(memory) > memory_size:
                        memory.pop(0)
                    activate(cid, t, init_state, task_opt)
                    reset_log.append((t, cid))

            else:  # attn
                q = _signature(H)
                for cid in reset_ids:
                    task_state, task_opt, _, qual = harvest(cid)
                    if memory:
                        sims = np.array([abs(np.vdot(q, e["key"])) for e in memory])
                        quals = np.array([e["qual"] for e in memory])
                        z = (quals - quals.mean()) / (quals.std() + 1e-6)
                        logits = (sims - 1.0) / attn_tau + attn_beta * z
                        idx = int(np.argmax(logits))
                        init_state, init_opt = memory[idx]["state"], memory[idx]["opt"]
                        diag["retrieval_age"].append(t - memory[idx]["t"])
                        diag["retrieved_is_latest"].append(idx == len(memory) - 1)
                    else:
                        init_state, init_opt = task_state, task_opt
                    memory.append(dict(key=q, state=task_state, opt=task_opt, qual=qual, t=t))
                    if len(memory) > memory_size:
                        memory.pop(0)
                    activate(cid, t, init_state, init_opt)
                    reset_log.append((t, cid))

            if verbose and (t + 1) % 100 == 0:
                print(f"    [{reset_init}] step {t+1}/{len(H_seq_full)}, {len(reset_log)} resets", flush=True)
    finally:
        for cid in range(n_copies):
            try:
                parent_conns[cid].send(("stop",))
            except Exception:
                pass
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    return minrs, reset_log, diag


def run_ls_track(H_seq_full, seed, rho, n_bits, base_cfg, v, t_s, init_sweeps=60, max_sweeps=20):
    """No-RL control: previous step's solution -> greedy bit-flip local search on the new channel."""
    env = MovingUserEnv(seed=seed, temporal_rho=rho, user_speed=v, T_s=t_s, direction_seed=seed, **base_cfg)
    rng = np.random.default_rng(seed)
    levels = 2 ** n_bits
    phi = rng.integers(0, levels, size=env.L * env.M) * (2 * np.pi / levels)
    minrs = []
    for t, H in enumerate(H_seq_full):
        env.H_true = H
        phi, minr, _, _ = greedy_bitflip_local_search(phi, env, n_bits,
                                                       max_sweeps=init_sweeps if t == 0 else max_sweeps)
        minrs.append(minr)
    return minrs
