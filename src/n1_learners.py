"""Single-copy (N=1) learners for the paper's meta-RL comparison. One base learner
(channel-AWARE GRPO, 1-bit, best-of-U + bit-flip polish), six ways of carrying
knowledge across channel changes:

  seq_warm     never reset (continuous warm start)
  cold         fresh random agent every step
  reptile      at each change-point: prior <- EMA(prior, adapted weights); reload prior
               (+ the adapted copy's Adam state)
  periodic_ema same blend, but every K steps regardless of the channel (K matched to the
               trajectory's number of change-points)
  prox         reptile + decoupled proximal pull toward the segment's starting prior
               after every GRPO update (AdamW-style stand-in for lambda*||theta-prior||^2)
  fomaml       prior trained by a first-order MAML meta-gradient: policy gradient (group
               baseline) of the ADAPTED weights on the next `m` steps' channels (retrospective
               query, so the meta-update is delayed m steps); no EMA

Change-points are computed once per trajectory from the channel (similarity to a reference
< thr), so every learner resets at identical steps. Resets load weights IN PLACE, so the
entropy-annealing schedule is global time (same as seq_warm); only weights and Adam state
are carried.
"""
import copy
import time

import numpy as np
import torch

torch.set_num_threads(1)

from moving_user_env import MovingUserEnv, channel_similarity
from grpo_ris import build_best_agent, update_quantized_best, quantize_phase_hard_torch
from physics import greedy_bitflip_local_search
from run_robustness_sweep import BASE_CFG, T_S, generate_channel_window

G = 8
CPU = torch.device("cpu")
ALGOS = ("seq_warm", "cold", "reptile", "periodic_ema", "prox", "fomaml")
DEFAULT_HP = dict(eta=0.3, alpha_prox=0.05, m=3, beta=1e-3, thr=0.8)


def change_points(H_seq, thr=0.8):
    ref, cps = H_seq[0], set()
    for t in range(1, len(H_seq)):
        if channel_similarity(ref, H_seq[t]) < thr:
            cps.add(t)
            ref = H_seq[t]
    return cps


def _snap(agent):
    return ({k: v.clone() for k, v in agent.actor.state_dict().items()},
            copy.deepcopy(agent.actor_optimizer.state_dict()))


def _load(agent, st, opt):
    agent.actor.load_state_dict(st)
    agent.actor_old.load_state_dict(st)
    if opt is not None:
        agent.actor_optimizer.load_state_dict(copy.deepcopy(opt))


class Base:
    def __init__(self, H_seq, seed, rho, v, U):
        self.env = MovingUserEnv(seed=seed, temporal_rho=rho, user_speed=v, T_s=T_S, direction_seed=seed, **BASE_CFG)
        self.scale = 1.0 / np.abs(H_seq[0]).mean()
        self.U, self.n, self.seed = U, len(H_seq), seed
        self.agent = self.new_agent(seed)

    def new_agent(self, s):
        return build_best_agent(self.env, CPU, n_bits=1, group_size=G, ppo_epochs=5,
                                entropy_anneal_updates=self.n * self.U, seed=s,
                                sigma_floor_frac=0.18, log_std_init_frac=0.35)

    def state_for(self, H):
        self.env.H_true = H
        self.env.H_obs = H * self.scale
        return self.env.reset_state_only()

    def step(self, H, anchor=None, alpha=0.0):
        env, agent = self.env, self.agent
        state = self.state_for(H)
        best, best_phi = -1e9, None
        for _ in range(self.U):
            update_quantized_best(agent, [state] * G, [H] * G, n_bits=1, reparam_coef=1.0)
            if anchor is not None and alpha > 0:
                with torch.no_grad():
                    for name, p in agent.actor.named_parameters():
                        p.add_(alpha * (anchor[name] - p))
                    agent.actor_old.load_state_dict(agent.actor.state_dict())
            with torch.no_grad():
                theta_mu, _ = agent.actor.forward(torch.FloatTensor(state).unsqueeze(0))
                phi_q = quantize_phase_hard_torch(theta_mu, 1)
                action_q = agent.actor._phi_to_action(phi_q).numpy()
            _, _, minr, _, _ = env.step(action_q)
            if minr > best:
                best, best_phi = minr, phi_q.numpy()[0].copy()
        _, pm, _, _ = greedy_bitflip_local_search(best_phi, env, 1, max_sweeps=20)
        return pm


def query_grad(agent, net, state_np, H):
    """Group-baseline policy gradient of `net`'s policy on channel H (first-order MAML query gradient)."""
    s = torch.FloatTensor(state_np).unsqueeze(0)
    with torch.no_grad():
        _, phi, _, _ = net.sample(s, n_samples=G)
        phi_q = quantize_phase_hard_torch(phi, 1)
        acts = net._phi_to_action(phi_q).numpy().reshape(1, G, -1)
    rewards = agent._group_rewards_batched(acts, np.asarray(H)[None])
    adv = torch.FloatTensor(agent._compute_group_advantage(rewards)[0])
    th, ka = net.forward(s.expand(G, -1))
    loss = -(adv * net.log_prob(phi, th, ka)).mean()
    return torch.autograd.grad(loss, list(net.parameters()), allow_unused=True)


def run_learner(algo, H_seq, seed, rho, v, U, hp=None, n_run=None, progress_path=None, progress_every=500):
    assert algo in ALGOS, algo
    hp = {**DEFAULT_HP, **(hp or {})}
    R = Base(H_seq, seed, rho, v, U)
    cps = change_points(H_seq, hp["thr"])
    K = max(1, round(len(H_seq) / max(1, len(cps)))) if algo == "periodic_ema" else None
    meta = anchor = None
    meta_net = meta_opt = None
    pending = []
    out, n_resets, n_meta, t0 = [], 0, 0, time.time()
    for t, H in enumerate(H_seq[: (n_run or len(H_seq))]):
        if algo == "cold":
            R.agent = R.new_agent(seed + 7919 * t)
        else:
            reset = ((t in cps) if algo in ("reptile", "prox", "fomaml")
                     else (t > 0 and t % K == 0) if algo == "periodic_ema" else False)
            if reset:
                n_resets += 1
                st, opt = _snap(R.agent)
                if algo == "fomaml":
                    if meta_net is None:
                        meta_net = copy.deepcopy(R.agent.actor)
                        meta_opt = torch.optim.Adam(meta_net.parameters(), lr=hp["beta"])
                    else:
                        pending.append(dict(net=copy.deepcopy(R.agent.actor), t0=t, g=None, k=0))
                    _load(R.agent, meta_net.state_dict(), opt)
                else:
                    meta = st if meta is None else {k: meta[k] + hp["eta"] * (st[k] - meta[k]) for k in meta}
                    _load(R.agent, meta, opt)
                    anchor = {k: v.clone() for k, v in meta.items()}
        pm = R.step(H, anchor if algo == "prox" else None, hp["alpha_prox"] if algo == "prox" else 0.0)
        out.append(pm)
        if progress_path and (t + 1) % progress_every == 0:
            np.save(progress_path, np.array(out))
        if algo == "fomaml" and pending:
            state_np = R.state_for(H)
            for e in list(pending):
                if e["t0"] <= t < e["t0"] + hp["m"]:
                    g = query_grad(R.agent, e["net"], state_np, H)
                    e["g"] = g if e["g"] is None else [a + b if a is not None else None for a, b in zip(e["g"], g)]
                    e["k"] += 1
                    if e["k"] == hp["m"]:
                        for p, gg in zip(meta_net.parameters(), e["g"]):
                            p.grad = None if gg is None else gg / hp["m"]
                        meta_opt.step()
                        n_meta += 1
                        pending.remove(e)
    return dict(minrs=out, n_resets=n_resets, n_meta=n_meta, wall=time.time() - t0)
