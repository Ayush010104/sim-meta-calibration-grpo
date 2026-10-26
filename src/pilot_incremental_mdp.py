"""Pilot: does the INCREMENTAL (flip) MDP help, and is it the MDP or just channel visibility?

Arms (single copy, no ensemble, NO local-search polish, 1-bit, same channel sequence):
  A   blind absolute   -- the pipeline used so far: actor state = [H_obs (never refreshed; ~4e-5 scale),
                          cos/sin of current phases]; absolute phase action.
  A'  aware absolute   -- same actor/update, but H_obs := H * SCALE refreshed every step (O(1) inputs).
  B   aware incremental-- state = [H*SCALE, current bits]; action = Bernoulli flip mask on the running
                          configuration; reward = min-rate improvement; GRPO (group-relative, robust adv.)
                          + PPO clip; the configuration advances to the best improving candidate.
Equal budget: U updates/step, group size G=8 in every arm.
Deployed value per step = best-of-U min-rate (A, A') / running-config min-rate after U updates (B).
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from moving_user_env import MovingUserEnv
from grpo_ris import build_best_agent, update_quantized_best, quantize_phase_hard_torch
from physics import eval_minr_batch, greedy_bitflip_local_search
from run_robustness_sweep import generate_channel_window, BASE_CFG, T_S

G = 8


def make_env(seed, rho, v):
    return MovingUserEnv(seed=seed, temporal_rho=rho, user_speed=v, T_s=T_S, direction_seed=seed, **BASE_CFG)


def run_absolute(H_seq, seed, rho, v, U, aware, polish=False):
    env = make_env(seed, rho, v)
    scale = 1.0 / np.abs(H_seq[0]).mean()
    agent = build_best_agent(env, torch.device("cpu"), n_bits=1, group_size=G, ppo_epochs=5,
                             entropy_anneal_updates=len(H_seq) * U, seed=seed,
                             sigma_floor_frac=0.18, log_std_init_frac=0.35)
    out = []
    for H in H_seq:
        env.H_true = H
        if aware:
            env.H_obs = H * scale
        state = env.reset_state_only()
        best, best_phi = -1e9, None
        for _ in range(U):
            update_quantized_best(agent, [state] * G, [H] * G, n_bits=1, reparam_coef=1.0)
            with torch.no_grad():
                theta_mu, _ = agent.actor.forward(torch.FloatTensor(state).unsqueeze(0))
                phi_q = quantize_phase_hard_torch(theta_mu, 1)
                action_q = agent.actor._phi_to_action(phi_q).numpy()
            _, _, minr, _, _ = env.step(action_q)
            if minr > best:
                best, best_phi = minr, phi_q.numpy()[0].copy()
        if polish:
            _, best, _, _ = greedy_bitflip_local_search(best_phi, env, 1, max_sweeps=20)
        out.append(best)
    return out


class FlipPolicy(nn.Module):
    def __init__(self, in_dim, n, hidden=128, bias0=-3.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(), nn.Linear(hidden, n))
        with torch.no_grad():
            self.net[-1].bias.fill_(bias0)
            self.net[-1].weight.mul_(0.1)

    def forward(self, s):
        return self.net(s)


def _p(logits, floor):
    return floor + (1 - 2 * floor) * torch.sigmoid(logits)


def _logp_p(p, f):
    return (f * torch.log(p + 1e-8) + (1 - f) * torch.log(1 - p + 1e-8)).sum(-1)


def run_incremental(H_seq, seed, rho, v, U, ent_coef=0.01, clip=0.2, ppo_epochs=5, floor=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(seed, rho, v)
    L, M = env.L, env.M
    n = L * M
    scale = 1.0 / np.abs(H_seq[0]).mean()
    policy = FlipPolicy(2 * H_seq[0].size + n, n)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-5)
    x = rng.integers(0, 2, n).astype(np.int64)                      # running configuration (bits)

    def evalb(bits_batch):
        return eval_minr_batch(bits_batch.astype(float) * np.pi, env)

    out, flipmass = [], []
    for H in H_seq:
        env.H_true = H
        hs = H * scale
        hfeat = np.concatenate([hs.real.flatten(), hs.imag.flatten()])
        base = float(evalb(x[None, :])[0])
        for _ in range(U):
            s = torch.FloatTensor(np.concatenate([hfeat, 2.0 * x - 1.0])).unsqueeze(0)
            with torch.no_grad():
                p_old = _p(policy(s), floor)
                f = torch.bernoulli(p_old.expand(G, -1))
                lp_old = _logp_p(p_old.expand(G, -1), f)
            cand = np.bitwise_xor(x[None, :], f.numpy().astype(np.int64))
            mc = evalb(cand)
            r = mc - base
            med = np.median(r)
            mad = np.median(np.abs(r - med)) * 1.4826
            adv = torch.FloatTensor((r - med) / (mad + 1e-6)) if mad > 1e-9 else torch.zeros(G)
            for _ in range(ppo_epochs):
                pc = _p(policy(s), floor)
                ratio = torch.exp(_logp_p(pc.expand(G, -1), f) - lp_old)
                pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
                p = pc[0]
                ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
                loss = pg - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
            k = int(np.argmax(mc))
            if mc[k] > base:
                x, base = cand[k].copy(), float(mc[k])
        out.append(base)
        flipmass.append(float(p_old.sum()))
    return out, flipmass


ARMS = {
    "A_blind": lambda H, s, r, v, U: run_absolute(H, s, r, v, U, aware=False),
    "A_aware": lambda H, s, r, v, U: run_absolute(H, s, r, v, U, aware=True),
    "A_blind_polish": lambda H, s, r, v, U: run_absolute(H, s, r, v, U, aware=False, polish=True),
    "A_aware_polish": lambda H, s, r, v, U: run_absolute(H, s, r, v, U, aware=True, polish=True),
    "B_flip": lambda H, s, r, v, U: run_incremental(H, s, r, v, U),
    "B_flip_floor": lambda H, s, r, v, U: run_incremental(H, s, r, v, U, ent_coef=0.02, floor=0.02),
}

if __name__ == "__main__":
    arm, U, seed, rho, v, n_steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    H_seq = generate_channel_window(seed, rho, v, T_S, n_steps)
    t0 = time.time()
    res = ARMS[arm](H_seq, seed, rho, v, U)
    tag = f"pilot_{arm}_U{U}_s{seed}_r{rho}_v{v}"
    if isinstance(res, tuple):
        res, fm = res
        np.save(tag + "_flipmass.npy", np.array(fm))
        print("  expected #flips/step at steps 0/100/300/600/999: " + " ".join(f"{fm[min(i, len(fm)-1)]:.1f}" for i in (0, 100, 300, 600, 999)), flush=True)
    np.save(tag + ".npy", np.array(res))
    print(f"{tag}: mean={np.mean(res):.4f}  last-300 mean={np.mean(res[-300:]):.4f}  {time.time()-t0:.0f}s", flush=True)
