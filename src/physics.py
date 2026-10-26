"""Pure-numpy physics helpers shared by the MINLP model and the baselines.
Reproduces exactly the reward/rate computation used throughout the notebook
(env.calculate_reward / per_user_rates_np / greedy_bitflip_local_search),
so every method here (MINLP, greedy local search, Direct-STE baseline) is
scored identically and comparably.
"""
import numpy as np


def phi_to_action(phi, L, M):
    """phi: (L*M,) angles -> (2*L*M,) action in env layout: per layer
    [real_l (M), imag_l (M)] concatenated across l."""
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    blocks = []
    for l in range(L):
        s = l * M
        blocks.append(cos_p[s:s + M])
        blocks.append(sin_p[s:s + M])
    return np.concatenate(blocks)


def per_user_rates(phi, env):
    """Exact replica of env.calculate_reward()'s per-user Shannon rates,
    but taking a phase vector phi (L*M,) directly instead of going through
    env.step (so it never mutates env state)."""
    L, M, K = env.L, env.M, env.K
    action = phi_to_action(phi, L, M)[None, :]
    d = []
    for l in range(L):
        s = l * 2 * M
        d.append(action[:, s:s + M] + 1j * action[:, s + M:s + 2 * M])
    Gv = np.broadcast_to(np.eye(M, dtype=complex), (1, M, M)).copy()
    for idx in range(L - 1, 0, -1):
        Gv = np.matmul(Gv, d[idx][:, :, None] * env.Ws[idx][None, :, :])
    Gv = Gv * d[0][:, None, :]
    HH = env.H_true.conj().T[None, :, :]
    Full = np.matmul(np.matmul(HH, Gv), env.Ws[0][None, :, :])
    power_all = np.abs(Full) ** 2 * env.P[None, None, :]
    diag_idx = np.arange(K)
    signal = power_all[:, diag_idx, diag_idx]
    interference = power_all.sum(axis=2) - signal
    rates = np.log2(1 + signal / (interference + env.var))
    return rates[0]


def per_user_sinr(phi, env):
    """Same as per_user_rates but returns raw SINR (no log2). Since
    log2(1+x) is strictly increasing, argmax_phi min_k rate_k(phi) ==
    argmax_phi min_k sinr_k(phi) -- the MINLP works directly on SINR to
    avoid a transcendental function in the model."""
    L, M, K = env.L, env.M, env.K
    action = phi_to_action(phi, L, M)[None, :]
    d = []
    for l in range(L):
        s = l * 2 * M
        d.append(action[:, s:s + M] + 1j * action[:, s + M:s + 2 * M])
    Gv = np.broadcast_to(np.eye(M, dtype=complex), (1, M, M)).copy()
    for idx in range(L - 1, 0, -1):
        Gv = np.matmul(Gv, d[idx][:, :, None] * env.Ws[idx][None, :, :])
    Gv = Gv * d[0][:, None, :]
    HH = env.H_true.conj().T[None, :, :]
    Full = np.matmul(np.matmul(HH, Gv), env.Ws[0][None, :, :])
    power_all = np.abs(Full) ** 2 * env.P[None, None, :]
    diag_idx = np.arange(K)
    signal = power_all[:, diag_idx, diag_idx]
    interference = power_all.sum(axis=2) - signal
    return (signal / (interference + env.var))[0]


def eval_minr_batch(phi_batch, env):
    """phi_batch: (S, L*M). Returns (S,) min-rate (over users) per row."""
    L, M, K = env.L, env.M, env.K
    S = phi_batch.shape[0]
    cos_p, sin_p = np.cos(phi_batch), np.sin(phi_batch)
    action = np.zeros((S, 2 * L * M))
    for l in range(L):
        s = l * M
        action[:, 2 * s:2 * s + M] = cos_p[:, s:s + M]
        action[:, 2 * s + M:2 * s + 2 * M] = sin_p[:, s:s + M]
    d = []
    for l in range(L):
        s = l * 2 * M
        d.append(action[:, s:s + M] + 1j * action[:, s + M:s + 2 * M])
    Gv = np.broadcast_to(np.eye(M, dtype=complex), (S, M, M)).copy()
    for idx in range(L - 1, 0, -1):
        Gv = np.matmul(Gv, d[idx][:, :, None] * env.Ws[idx][None, :, :])
    Gv = Gv * d[0][:, None, :]
    H_batch = np.broadcast_to(env.H_true, (S, M, K)).copy()
    HH = H_batch.conj().transpose(0, 2, 1)
    Full = np.matmul(np.matmul(HH, Gv), env.Ws[0][None, :, :])
    power_all = np.abs(Full) ** 2 * env.P[None, None, :]
    diag_idx = np.arange(K)
    signal = power_all[:, diag_idx, diag_idx]
    interference = power_all.sum(axis=2) - signal
    rates = np.log2(1 + signal / (interference + env.var))
    return rates.min(axis=1)


def quantize_phase_hard(phi, n_bits):
    if n_bits is None or n_bits <= 0:
        return phi
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    phi_wrapped = np.mod(phi, 2 * np.pi)
    q = np.round(phi_wrapped / step) * step
    return np.mod(q, 2 * np.pi)


def greedy_bitflip_local_search(phi0, env, n_bits, max_sweeps=30, verbose=False):
    """Verbatim (numpy) port of the notebook's greedy_bitflip_local_search:
    exact discrete coordinate-wise steepest-ascent local search over the
    2**n_bits phase levels per element. Monotonically non-decreasing in
    min_rate; terminates at a discrete local optimum."""
    L, M = env.L, env.M
    n_phase = L * M
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    phi = phi0.copy()
    n_evals = 0

    cur_minr = eval_minr_batch(phi[None, :], env)[0]
    n_evals += 1

    n_sweeps_used = 0
    for sweep in range(max_sweeps):
        candidates, meta = [], []
        for i in range(n_phase):
            cur_level = int(round(np.mod(phi[i], 2 * np.pi) / step)) % levels
            for lv in range(levels):
                if lv == cur_level:
                    continue
                cand = phi.copy()
                cand[i] = lv * step
                candidates.append(cand)
                meta.append((i, lv))
        cand_minrs = eval_minr_batch(np.stack(candidates, axis=0), env)
        n_evals += len(candidates)
        n_sweeps_used = sweep + 1

        best_idx = int(np.argmax(cand_minrs))
        if cand_minrs[best_idx] > cur_minr + 1e-9:
            phi = candidates[best_idx]
            cur_minr = cand_minrs[best_idx]
            if verbose:
                i, lv = meta[best_idx]
                print(f"  sweep {sweep}: flip element {i} -> level {lv}, min_rate={cur_minr:.4f}")
        else:
            break
    return phi, cur_minr, n_evals, n_sweeps_used
