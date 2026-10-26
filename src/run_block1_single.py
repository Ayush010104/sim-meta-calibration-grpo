"""Block 1 single-core queue: seq-WARM (blind / aware, with polish) and LS-track (polish-only attribution control),
deployed phases logged. rho in {0.99, 0.7}, v=1.4, 3 seeds, 1000 steps."""
import os, time
import numpy as np, torch
torch.set_num_threads(1)
from grpo_ris import build_best_agent, update_quantized_best, quantize_phase_hard_torch
from physics import greedy_bitflip_local_search
from pilot_incremental_mdp import make_env, G
from run_robustness_sweep import generate_channel_window, T_S

OUT, LOG, V, U = "block1_single_results.npy", "run_block1_single.log", 1.4, 10

def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

def seqwarm(H_seq, seed, rho, aware):
    env = make_env(seed, rho, V); scale = 1.0 / np.abs(H_seq[0]).mean()
    agent = build_best_agent(env, torch.device("cpu"), n_bits=1, group_size=G, ppo_epochs=5,
                             entropy_anneal_updates=len(H_seq) * U, seed=seed, sigma_floor_frac=0.18, log_std_init_frac=0.35)
    out, phis = [], []
    for H in H_seq:
        env.H_true = H
        if aware: env.H_obs = H * scale
        state = env.reset_state_only(); best, best_phi = -1e9, None
        for _ in range(U):
            update_quantized_best(agent, [state] * G, [H] * G, n_bits=1, reparam_coef=1.0)
            with torch.no_grad():
                theta_mu, _ = agent.actor.forward(torch.FloatTensor(state).unsqueeze(0))
                phi_q = quantize_phase_hard_torch(theta_mu, 1)
                action_q = agent.actor._phi_to_action(phi_q).numpy()
            _, _, minr, _, _ = env.step(action_q)
            if minr > best: best, best_phi = minr, phi_q.numpy()[0].copy()
        pphi, pm, _, _ = greedy_bitflip_local_search(best_phi, env, 1, max_sweeps=20)
        out.append(pm); phis.append(np.round(np.asarray(pphi) / np.pi).astype(np.int8) % 2)
    return out, np.stack(phis)

def lstrack(H_seq, seed, rho):
    env = make_env(seed, rho, V); rng = np.random.default_rng(seed)
    phi = rng.integers(0, 2, env.L * env.M) * np.pi; out, phis = [], []
    for t, H in enumerate(H_seq):
        env.H_true = H
        phi, pm, _, _ = greedy_bitflip_local_search(phi, env, 1, max_sweeps=60 if t == 0 else 20)
        out.append(pm); phis.append(np.round(np.asarray(phi) / np.pi).astype(np.int8) % 2)
    return out, np.stack(phis)

def main():
    res = np.load(OUT, allow_pickle=True).item() if os.path.exists(OUT) else {}
    for rho in (0.99, 0.7):
        for seed in (12345, 23456, 34567):
            H = generate_channel_window(seed, rho, V, T_S, 1000)
            for kind in ("lstrack", "seqwarm_aware", "seqwarm_blind"):
                key = (kind, rho, seed)
                if key in res: continue
                t0 = time.time()
                m, ph = lstrack(H, seed, rho) if kind == "lstrack" else seqwarm(H, seed, rho, kind == "seqwarm_aware")
                res[key] = dict(minrs=m, bits=ph); np.save(OUT, np.array(res, dtype=object))
                log(f"[{kind} rho={rho} seed={seed}] mean={np.mean(m):.4f} {time.time()-t0:.0f}s (done {len(res)}/18)")
    log("BLOCK 1 SINGLE DONE")

if __name__ == "__main__":
    main()
