"""Layer-wise Alternating Optimization (AO) baseline for the L-layer SIM
cascade -- the literature-faithful classical baseline for this problem
(power allocation is fixed in this system model, so there is no
precoder-vs-phase alternation to do; the natural block-coordinate structure
is across the L=4 cascaded phase-only layers instead).

Genuine block-coordinate alternation: at any given gradient step, only ONE
layer's M phase variables are updated (via its own Adam optimizer); the
other L-1 layers' variables are held fixed (detached, not even tracked by
autograd for that step). The active layer cycles round-robin across the
per-step update budget, exactly mirroring classical AO's
fix-everything-else-and-optimize-one-block structure, as opposed to
Direct-STE's joint all-layers-at-once gradient descent.

Reuses the same differentiable reward (compute_reward_torch) and STE
quantization (quantize_phase_ste) as direct_ste_baseline.py / GRPO's
pathwise term, and the same continuous/warm-started, budget-matched,
moving-user scenario as the other baseline comparisons.
"""
import time
import numpy as np
import torch

torch.set_num_threads(1)

from moving_user_env import MovingUserEnv
from direct_ste_baseline import quantize_phase_ste, compute_reward_torch
from physics import greedy_bitflip_local_search, quantize_phase_hard

BASE_CFG = dict(num_antennas=5, num_RIS_elements=36, num_users=5, num_layers=4, dBS=30.0, radius=50.0)
V, T_S, RHO = 1.4, 0.5, 0.99
SEED = 12345
N_STEPS = 1000
N_BITS = 1
LR = 0.15


def run_ao_layerwise_continuous(H_seq, seed, updates_per_step, verbose=False):
    env = MovingUserEnv(seed=seed, temporal_rho=RHO, user_speed=V, T_s=T_S, direction_seed=seed, **BASE_CFG)
    L, M = env.L, env.M

    torch.manual_seed(seed)
    raw_r = [torch.randn(M, requires_grad=True) for _ in range(L)]
    raw_i = [torch.randn(M, requires_grad=True) for _ in range(L)]
    opts = [torch.optim.Adam([raw_r[l], raw_i[l]], lr=LR) for l in range(L)]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(opts[l], T_max=len(H_seq) * updates_per_step // L + 1,
                                                           eta_min=LR * 0.05) for l in range(L)]

    minrs = []
    t0 = time.time()
    for t, H in enumerate(H_seq):
        env.H_true = H
        H_t = torch.tensor(H, dtype=torch.complex64).unsqueeze(0)

        best_minsinr, best_action = -1e9, None
        for u in range(updates_per_step):
            active = u % L  # round-robin block-coordinate cycling across the L layers

            opts[active].zero_grad()
            blocks_r, blocks_i = [], []
            for l in range(L):
                if l == active:
                    blocks_r.append(raw_r[l]); blocks_i.append(raw_i[l])
                else:
                    with torch.no_grad():
                        blocks_r.append(raw_r[l].detach()); blocks_i.append(raw_i[l].detach())

            raw_r_full = torch.cat(blocks_r).unsqueeze(0)
            raw_i_full = torch.cat(blocks_i).unsqueeze(0)
            phi = torch.atan2(raw_i_full, raw_r_full)
            phi_used = quantize_phase_ste(phi, N_BITS)
            cos_phi, sin_phi = torch.cos(phi_used), torch.sin(phi_used)
            blocks = []
            for l in range(L):
                s = l * M
                blocks.append(cos_phi[:, s:s + M])
                blocks.append(sin_phi[:, s:s + M])
            action = torch.cat(blocks, dim=1)
            sinr = compute_reward_torch(action, H_t, env.Ws, env.P, env.var, L, M, env.K).mean()
            (-sinr).backward()
            torch.nn.utils.clip_grad_norm_([raw_r[active], raw_i[active]], max_norm=5.0)
            opts[active].step()
            scheds[active].step()

            if sinr.item() > best_minsinr:
                best_minsinr = sinr.item()
                best_action = action.detach().numpy().copy()

        phi_from_action = np.zeros(L * M)
        for l in range(L):
            s = l * 2 * M
            r, i = best_action[0, s:s + M], best_action[0, s + M:s + 2 * M]
            phi_from_action[l * M:(l + 1) * M] = np.arctan2(i, r)
        phi_hard = quantize_phase_hard(phi_from_action, N_BITS)
        _, polished_minr, _, _ = greedy_bitflip_local_search(phi_hard, env, N_BITS, max_sweeps=20)
        minrs.append(polished_minr)

        if verbose and (t + 1) % 100 == 0:
            print(f"    [AO-layerwise u={updates_per_step}] step {t+1}/{len(H_seq)}, elapsed {time.time()-t0:.0f}s", flush=True)

    return minrs


def generate_channel_window(seed, rho, n_steps):
    env = MovingUserEnv(seed=seed, temporal_rho=rho, user_speed=V, T_s=T_S, direction_seed=seed, **BASE_CFG)
    H_seq = [env.H_true.copy()]
    for _ in range(n_steps - 1):
        env.advance_channel()
        H_seq.append(env.H_true.copy())
    return H_seq


if __name__ == "__main__":
    import sys
    updates_per_step = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"AO-LAYERWISE (continuous, u={updates_per_step}) vs GRPO: v={V}m/s, T_s={T_S}s, rho={RHO}, "
          f"N_STEPS={N_STEPS}, seed={SEED}", flush=True)
    H_seq = generate_channel_window(SEED, RHO, N_STEPS)
    t0 = time.time()
    minrs = run_ao_layerwise_continuous(H_seq, SEED, updates_per_step, verbose=True)
    m = np.array(minrs)
    GRPO_SEQ_WARM_BASELINE = 0.938961806859735
    print(f"\n[AO-layerwise u={updates_per_step}] done in {time.time()-t0:.0f}s, mean={m.mean():.4f}, "
          f"pct_of_GRPO10={100*m.mean()/GRPO_SEQ_WARM_BASELINE:.1f}%", flush=True)
    np.save(f"ao_layerwise_results_u{updates_per_step}.npy", np.array(dict(minrs=minrs), dtype=object))
