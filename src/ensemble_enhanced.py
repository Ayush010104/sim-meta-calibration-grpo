"""Four toggleable enhancements to the ENSEMBLE-META mechanism, each
testable in isolation against the existing baseline (META_LR=0.3, plain
Reptile, no advantage weighting, no explicit exploration reset).

1. meta_lr: override the fixed Reptile step size (never tuned before).
2. distill: replace the simple linear Reptile blend with a FUNCTION-SPACE
   distillation -- instead of interpolating WEIGHTS (which failed badly in
   MACRO-v2, landing between loss-landscape basins), take the newest
   harvest's weights as-is (a real, coherent network -- no averaging), then
   fine-tune it with a few gradient steps toward the CIRCULAR MEAN of what
   it and the previous 1-2 harvested networks would each predict on the
   current probe state. Matches behavior, not parameters.
3. advantage_weighted: scale the (or the distillation strength of the)
   update by how good that harvest's own recent performance was, so
   mediocre harvests contribute less than strong ones.
4. explore_reset: after loading meta_state into a freshly-reset copy,
   explicitly widen its log_std (exploration) rather than inheriting
   whatever value meta_state's own log_std had converged to.
"""
import copy
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F

UPDATES_PER_STEP = 10
GROUP_SIZE = 8


def _circular_mean(thetas):
    """thetas: list of (n_phase,) tensors of angles. Circular mean per element."""
    sin_sum = sum(torch.sin(t) for t in thetas)
    cos_sum = sum(torch.cos(t) for t in thetas)
    return torch.atan2(sin_sum, cos_sum)


def _copy_worker(cid, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold, conn):
    torch.set_num_threads(1)
    from moving_user_env import MovingUserEnv, channel_similarity
    from grpo_ris import build_best_agent, update_quantized_best, quantize_phase_hard_torch
    from physics import greedy_bitflip_local_search

    def make_env():
        return MovingUserEnv(seed=seed + cid, temporal_rho=rho, user_speed=v, T_s=t_s,
                              direction_seed=seed + cid, **base_cfg)

    env = agent = H_ref = None
    recent_minrs = []  # last few minr's THIS copy achieved, for advantage weighting

    while True:
        msg = conn.recv()
        cmd = msg[0]

        if cmd == "activate":
            _, meta_state, meta_opt_state, t, explore_reset, log_std_widen = msg
            env = make_env()
            agent = build_best_agent(env, torch.device("cpu"), n_bits=n_bits, group_size=GROUP_SIZE,
                                      ppo_epochs=5, entropy_anneal_updates=1000 * UPDATES_PER_STEP,
                                      seed=seed + cid + t, sigma_floor_frac=0.18, log_std_init_frac=0.35)
            if meta_state is not None:
                agent.actor.load_state_dict(meta_state)
                agent.actor_old.load_state_dict(meta_state)
                agent.actor_optimizer.load_state_dict(meta_opt_state)
                if explore_reset:
                    with torch.no_grad():
                        agent.actor.log_std.data.fill_(float(np.log(log_std_widen)))
                        agent.actor_old.log_std.data.fill_(float(np.log(log_std_widen)))
            H_ref = None
            recent_minrs = []
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
            recent_minrs.append(polished_minr)
            if len(recent_minrs) > 5:
                recent_minrs.pop(0)
            if H_ref is None:
                H_ref = H
            should_reset = channel_similarity(H_ref, H) < sim_threshold
            conn.send((polished_minr, should_reset))

        elif cmd == "harvest":
            task_state = {k: v_.clone() for k, v_ in agent.actor.state_dict().items()}
            task_opt_state = copy.deepcopy(agent.actor_optimizer.state_dict())
            probe_state = env.reset_state_only()
            avg_recent_minr = float(np.mean(recent_minrs)) if recent_minrs else 0.0
            conn.send((task_state, task_opt_state, probe_state, avg_recent_minr))

        elif cmd == "stop":
            conn.close()
            break


def _distill(base_state_dict, harvest_buffer, probe_state, n_bits, base_cfg,
              n_steps=8, lr=1e-2):
    """base_state_dict: the newest harvest's weights (starting point, real basin).
    harvest_buffer: list of (state_dict, weight) for up to the last few harvests
    (including the newest). Builds a circular-mean target action on probe_state
    from ALL buffered networks, then fine-tunes a copy of base_state_dict a few
    steps to match it -- function-space distillation, no weight averaging."""
    from grpo_ris import VonMisesActorQBest
    import env_ris  # for state_dim/action_dim constants via a throwaway env is overkill; infer from state_dict shapes

    # infer dims from the state_dict itself
    w0 = base_state_dict["net.0.weight"]
    state_dim = w0.shape[1]
    hidden_dim = w0.shape[0]
    n_phase = base_state_dict["log_std"].shape[0]
    # M, L aren't separately recoverable from n_phase alone in general, but for
    # this fixed problem M*L=n_phase with the run's known M=36 -> L=n_phase//36
    M = 36
    L = n_phase // M

    def build_actor(sd):
        actor = VonMisesActorQBest(state_dim=state_dim, action_dim=None, M=M, N=5, K=5, L=L,
                                    device=torch.device("cpu"), hidden_dim=hidden_dim, depth=1)
        actor.load_state_dict(sd)
        return actor

    probe_t = torch.FloatTensor(probe_state).unsqueeze(0)

    with torch.no_grad():
        target_thetas = []
        for sd, weight in harvest_buffer:
            a = build_actor(sd)
            theta_mu, _ = a.forward(probe_t)
            target_thetas.append(theta_mu.squeeze(0))
        target = _circular_mean(target_thetas).detach()

    student = build_actor(base_state_dict)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        theta_mu, _ = student.forward(probe_t)
        theta_mu = theta_mu.squeeze(0)
        loss = F.mse_loss(torch.cos(theta_mu), torch.cos(target)) + F.mse_loss(torch.sin(theta_mu), torch.sin(target))
        loss.backward()
        opt.step()

    return {k: v.clone() for k, v in student.state_dict().items()}


class EnhancedCopy:
    def __init__(self, cid, start_offset):
        self.cid, self.start_offset = cid, start_offset


def run_ensemble_enhanced(H_seq_full, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold=0.8,
                           n_copies=10, stagger=50,
                           meta_lr=0.3, distill=False, advantage_weighted=False,
                           explore_reset=False, log_std_widen=0.3,
                           distill_buffer_size=3, verbose=False):
    ctx = mp.get_context("spawn")
    parent_conns, procs = [], []
    for cid in range(n_copies):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_copy_worker,
                         args=(cid, seed, rho, n_bits, base_cfg, v, t_s, sim_threshold, child_conn))
        p.start()
        parent_conns.append(parent_conn)
        procs.append(p)

    def activate(cid, t, meta_state, meta_opt_state):
        parent_conns[cid].send(("activate", meta_state, meta_opt_state, t, explore_reset, log_std_widen))
        parent_conns[cid].recv()

    try:
        active = [False] * n_copies
        meta_state, meta_opt_state = None, None
        harvest_buffer = []  # list of (state_dict, weight) for distillation
        recent_pop_minrs = []  # rolling population-average minr, for advantage weighting
        minrs, reset_log = [], []

        for t, H in enumerate(H_seq_full):
            for cid in range(n_copies):
                if not active[cid] and t >= cid * stagger:
                    activate(cid, t, meta_state, meta_opt_state)
                    active[cid] = True

            active_ids = [cid for cid in range(n_copies) if active[cid]]
            for cid in active_ids:
                parent_conns[cid].send(("step", H))
            results = {cid: parent_conns[cid].recv() for cid in active_ids}

            best_cid = max(active_ids, key=lambda c: results[c][0])
            minrs.append(results[best_cid][0])
            recent_pop_minrs.append(np.mean([results[c][0] for c in active_ids]))
            if len(recent_pop_minrs) > 20:
                recent_pop_minrs.pop(0)
            pop_baseline = np.mean(recent_pop_minrs) if recent_pop_minrs else 0.0

            for cid in active_ids:
                minr, should_reset = results[cid]
                if should_reset:
                    parent_conns[cid].send(("harvest",))
                    task_state, task_opt_state, probe_state, avg_recent_minr = parent_conns[cid].recv()

                    weight = 1.0
                    if advantage_weighted:
                        # sigmoid-shaped weight: harvests better than the recent
                        # population baseline count more, worse ones count less
                        diff = (avg_recent_minr - pop_baseline) / (abs(pop_baseline) + 1e-6)
                        weight = float(1.0 / (1.0 + np.exp(-4.0 * diff)))  # in (0,1)

                    effective_lr = meta_lr * (2 * weight if advantage_weighted else 1.0)
                    effective_lr = min(effective_lr, 0.95)

                    if distill:
                        harvest_buffer.append((task_state, weight))
                        if len(harvest_buffer) > distill_buffer_size:
                            harvest_buffer.pop(0)
                        if len(harvest_buffer) == 1:
                            new_meta_state = task_state
                        else:
                            new_meta_state = _distill(task_state, harvest_buffer, probe_state, n_bits, base_cfg)
                        if meta_state is None:
                            meta_state = new_meta_state
                        else:
                            meta_state = {k: meta_state[k] + effective_lr * (new_meta_state[k] - meta_state[k])
                                          for k in meta_state}
                    else:
                        if meta_state is None:
                            meta_state = task_state
                        else:
                            meta_state = {k: meta_state[k] + effective_lr * (task_state[k] - meta_state[k])
                                          for k in meta_state}
                    meta_opt_state = task_opt_state
                    activate(cid, t, meta_state, meta_opt_state)
                    reset_log.append((t, cid))

            if verbose and (t + 1) % 100 == 0:
                print(f"    [enhanced ensemble] step {t+1}/{len(H_seq_full)}, "
                      f"{len(active_ids)} active, {len(reset_log)} resets so far")
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
