"""Self-contained torch port of the notebook's Direct-STE optimizer +
greedy bit-flip polish, used purely as an apples-to-apples baseline to
compare against the MINLP solution (same env, same seed, same metric:
final min_rate after hard quantization).
"""
import numpy as np
import torch

from physics import eval_minr_batch, greedy_bitflip_local_search, quantize_phase_hard


def quantize_phase_ste(phi, n_bits):
    if n_bits is None or n_bits <= 0:
        return phi
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    phi_wrapped = torch.remainder(phi, 2 * np.pi)
    q = torch.round(phi_wrapped / step) * step
    q = torch.remainder(q, 2 * np.pi)
    return phi_wrapped + (q - phi_wrapped).detach()


def compute_reward_torch(action, H_batch, Ws, P, var, L, M, K):
    """SINR-only differentiable reward (no log2, no mean/min blend) so the
    gradient path optimizes the exact same quantity the MINLP does."""
    device = action.device
    cdtype = torch.complex64
    batch_n = action.shape[0]
    Ws_t = [torch.tensor(w, dtype=cdtype, device=device) for w in Ws]
    P_t = torch.tensor(P, dtype=torch.float32, device=device)
    H_t = H_batch.to(dtype=cdtype, device=device)

    d = []
    for l in range(L):
        start = l * 2 * M
        Phi_real_l = action[:, start:start + M]
        Phi_imag_l = action[:, start + M:start + 2 * M]
        d.append(torch.complex(Phi_real_l, Phi_imag_l))

    Gv = torch.eye(M, dtype=cdtype, device=device).unsqueeze(0).expand(batch_n, M, M).clone()
    for idx in range(L - 1, 0, -1):
        M_idx = d[idx].unsqueeze(-1) * Ws_t[idx].unsqueeze(0)
        Gv = torch.matmul(Gv, M_idx)
    Gv = Gv * d[0].unsqueeze(1)

    HH = H_t.conj().transpose(1, 2)
    temp = torch.matmul(HH, Gv)
    Full = torch.matmul(temp, Ws_t[0])

    power_all = Full.abs() ** 2 * P_t.view(1, 1, K)
    signal = torch.diagonal(power_all, dim1=-2, dim2=-1)
    interference = power_all.sum(dim=-1) - signal
    sinr = signal / (interference + var)
    return sinr.min(dim=1).values


def run_direct_ste(env, seed, n_bits, n_steps=5000, lr=0.15):
    L, M = env.L, env.M
    n_elements = L * M
    torch.manual_seed(seed)
    raw_r = torch.randn(n_elements, requires_grad=True)
    raw_i = torch.randn(n_elements, requires_grad=True)
    opt = torch.optim.Adam([raw_r, raw_i], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.05)
    H_t = torch.tensor(env.H_true, dtype=torch.complex64).unsqueeze(0)

    best_minsinr, best_action = -1e9, None
    for step in range(n_steps):
        opt.zero_grad()
        phi = torch.atan2(raw_i, raw_r).unsqueeze(0)
        phi_used = quantize_phase_ste(phi, n_bits) if n_bits else phi
        cos_phi, sin_phi = torch.cos(phi_used), torch.sin(phi_used)
        blocks = []
        for l in range(L):
            s = l * M
            blocks.append(cos_phi[:, s:s + M])
            blocks.append(sin_phi[:, s:s + M])
        action = torch.cat(blocks, dim=1)
        sinr = compute_reward_torch(action, H_t, env.Ws, env.P, env.var, L, M, env.K).mean()
        (-sinr).backward()
        torch.nn.utils.clip_grad_norm_([raw_r, raw_i], max_norm=5.0)
        opt.step()
        sched.step()

        if sinr.item() > best_minsinr:
            best_minsinr = sinr.item()
            best_action = action.detach().numpy().copy()

    phi_from_action = np.zeros(L * M)
    for l in range(L):
        s = l * 2 * M
        r, i = best_action[0, s:s + M], best_action[0, s + M:s + 2 * M]
        phi_from_action[l * M:(l + 1) * M] = np.arctan2(i, r)
    phi_hard = quantize_phase_hard(phi_from_action, n_bits)
    minr_hard = float(eval_minr_batch(phi_hard[None, :], env)[0])
    return dict(phi=phi_hard, minr=minr_hard)


def run_baselines(env, seed, n_bits, n_steps=5000, ls_sweeps=50):
    direct = run_direct_ste(env, seed, n_bits, n_steps=n_steps)
    polished_phi, polished_minr, n_evals, n_sweeps = greedy_bitflip_local_search(
        direct["phi"], env, n_bits, max_sweeps=ls_sweeps)
    return dict(direct_minr=direct["minr"], direct_phi=direct["phi"],
                polished_minr=polished_minr, polished_phi=polished_phi,
                n_ls_sweeps=n_sweeps, n_ls_evals=n_evals)
