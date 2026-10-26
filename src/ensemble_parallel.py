"""Genuine intra-ensemble parallelism: each of the N copies runs in its own
PERSISTENT OS process (state -- network weights, optimizer momentum, H_ref
-- must survive across steps, so this is NOT a stateless worker pool; each
copy gets a long-lived process communicating over a Pipe).

Protocol per worker, driven by the main process:
  ('activate', meta_state, meta_opt_state, rho) -> worker rebuilds its
      agent/env, loads meta_state (if any), replies 'activated'
  ('step', H)                                    -> worker runs its 10
      GRPO updates + local search against H, replies (minr, should_reset)
  ('harvest',)                                    -> worker snapshots its
      current weights+optimizer state, replies (task_state, task_opt_state)
  ('stop',)                                       -> worker exits

Each step of the main loop: broadcast H to every ACTIVE copy's pipe, then
collect all replies -- the N copies' 10-update-each GRPO work genuinely
overlaps in wall-clock time across the N worker processes, rather than
running one after another in a single process.
"""
import copy
import multiprocessing as mp
import numpy as np
import torch

META_LR = 0.3
UPDATES_PER_STEP = 10
GROUP_SIZE = 8


def _copy_worker(cid, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold, conn):
    torch.set_num_threads(1)
    from moving_user_env import MovingUserEnv, channel_similarity
    from grpo_ris import build_best_agent, update_quantized_best, quantize_phase_hard_torch
    from physics import greedy_bitflip_local_search

    def make_env():
        return MovingUserEnv(seed=seed + cid, temporal_rho=rho, user_speed=v, T_s=t_s,
                              direction_seed=seed + cid, **base_cfg)

    env = agent = H_ref = None

    while True:
        msg = conn.recv()
        cmd = msg[0]

        if cmd == "activate":
            _, meta_state, meta_opt_state, t = msg
            env = make_env()
            agent = build_best_agent(env, torch.device("cpu"), n_bits=n_bits, group_size=GROUP_SIZE,
                                      ppo_epochs=5, entropy_anneal_updates=1000 * UPDATES_PER_STEP,
                                      seed=seed + cid + t, sigma_floor_frac=0.18, log_std_init_frac=0.35)
            if meta_state is not None:
                agent.actor.load_state_dict(meta_state)
                agent.actor_old.load_state_dict(meta_state)
                agent.actor_optimizer.load_state_dict(meta_opt_state)
            H_ref = None
            conn.send("activated")

        elif cmd == "step":
            _, H = msg
            env.H_true = H
            state = env.reset_state_only()
            best_minr, best_phi = -1e9, None
            for _ in range(UPDATES_PER_STEP):
                update_quantized_best(agent, [state] * GROUP_SIZE, [H] * GROUP_SIZE,
                                       n_bits=n_bits, reparam_coef=1.0)
                with torch.no_grad():
                    theta_mu, _ = agent.actor.forward(torch.FloatTensor(state).unsqueeze(0))
                    phi_q = quantize_phase_hard_torch(theta_mu, n_bits)
                    action_q = agent.actor._phi_to_action(phi_q).numpy()
                _, r, minr, _, info = env.step(action_q)
                if minr > best_minr:
                    best_minr, best_phi = minr, phi_q.numpy()[0].copy()
            _, polished_minr, _, _ = greedy_bitflip_local_search(best_phi, env, n_bits, max_sweeps=20)
            if H_ref is None:
                H_ref = H
            should_reset = channel_similarity(H_ref, H) < sim_threshold
            conn.send((polished_minr, should_reset))

        elif cmd == "harvest":
            task_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}
            task_opt_state = copy.deepcopy(agent.actor_optimizer.state_dict())
            conn.send((task_state, task_opt_state))

        elif cmd == "stop":
            conn.close()
            break


def run_ensemble_parallel(H_seq_full, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold=0.8,
                           n_copies=10, stagger=50, verbose=False, reset_mode="meta"):
    """reset_mode='meta' (default): resets harvest weights+optimizer into a
    SHARED Reptile meta_state, and re-activate from it -- this is the
    ENSEMBLE method used everywhere else in this session.
    reset_mode='dumb': ABLATION -- resets just reactivate that copy with a
    fresh random init (meta_state=None every time, no harvesting, no
    blending, no cross-copy sharing at all). Isolates how much of the
    ensemble's advantage comes from staggering + best-of-N selection alone,
    versus the learned Reptile prior specifically.
    reset_mode='never': ABLATION -- copies never reset at all (each is a
    plain, independently-seeded seq-WARM copy, staggered at start but
    otherwise just left to run forever). Isolates whether having several
    independent random initializations running in parallel, with best-of-N
    selection, already captures most of the gain WITHOUT any reset
    mechanism at all."""
    ctx = mp.get_context("spawn")
    parent_conns, procs = [], []
    for cid in range(n_copies):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_copy_worker,
                         args=(cid, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold, child_conn))
        p.start()
        parent_conns.append(parent_conn)
        procs.append(p)

    try:
        active = [False] * n_copies
        meta_state, meta_opt_state = None, None
        minrs, reset_log = [], []

        for t, H in enumerate(H_seq_full):
            for cid in range(n_copies):
                if not active[cid] and t >= cid * stagger:
                    parent_conns[cid].send(("activate", meta_state, meta_opt_state, t))
                    parent_conns[cid].recv()
                    active[cid] = True

            active_ids = [cid for cid in range(n_copies) if active[cid]]
            for cid in active_ids:
                parent_conns[cid].send(("step", H))
            results = {cid: parent_conns[cid].recv() for cid in active_ids}

            best_cid = max(active_ids, key=lambda c: results[c][0])
            minrs.append(results[best_cid][0])

            for cid in active_ids:
                minr, should_reset = results[cid]
                if should_reset and reset_mode != "never":
                    if reset_mode == "meta":
                        parent_conns[cid].send(("harvest",))
                        task_state, task_opt_state = parent_conns[cid].recv()
                        if meta_state is None:
                            meta_state = task_state
                        else:
                            meta_state = {k: meta_state[k] + META_LR * (task_state[k] - meta_state[k])
                                           for k in meta_state}
                        meta_opt_state = task_opt_state
                        parent_conns[cid].send(("activate", meta_state, meta_opt_state, t))
                    elif reset_mode == "dumb":
                        parent_conns[cid].send(("activate", None, None, t))
                    else:
                        raise ValueError(reset_mode)
                    parent_conns[cid].recv()
                    reset_log.append((t, cid))

            if verbose and (t + 1) % 100 == 0:
                print(f"    [parallel ensemble] step {t+1}/{len(H_seq_full)}, "
                      f"{len(active_ids)} active copies, {len(reset_log)} resets so far")
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

    return minrs, reset_log
