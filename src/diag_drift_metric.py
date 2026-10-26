"""Phase 0b: which channel-similarity best predicts SOLUTION-TRANSFER loss?
transfer(t0,lag) = minr(phi_deployed[t0], H[t0+lag]) / minr(phi_deployed[t0], H[t0])
Candidates: S1 global cosine (current trigger), S2 mean per-user cosine, S3 min per-user cosine
(S2/S3 are invariant to per-user phase rotation, which the reward is invariant to)."""
import numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
from moving_user_env import MovingUserEnv
from physics import eval_minr_batch
from run_robustness_sweep import generate_channel_window, BASE_CFG, T_S

mpl.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight"})
res = np.load("stage1b_results.npy", allow_pickle=True).item()
phi = res["phi_log"]["phi"]
H = generate_channel_window(12345, 0.99, 1.4, T_S, len(phi))
env = MovingUserEnv(seed=12345, temporal_rho=0.99, user_speed=1.4, T_s=T_S, direction_seed=12345, **BASE_CFG)

def minr(p, Hh):
    env.H_true = Hh
    return eval_minr_batch(p[None, :], env)[0]

def s1(a, b): return abs(np.vdot(a.flatten(), b.flatten())) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
def per_user(a, b):
    return np.array([abs(np.vdot(a[:, k], b[:, k])) / (np.linalg.norm(a[:, k]) * np.linalg.norm(b[:, k]) + 1e-30)
                     for k in range(a.shape[1])])

rows = []
for t0 in range(0, len(phi) - 61, 8):
    base = minr(phi[t0], H[t0])
    for lag in range(1, 61):
        t = t0 + lag
        pu = per_user(H[t0], H[t])
        rows.append((minr(phi[t0], H[t]) / base, s1(H[t0], H[t]), pu.mean(), pu.min()))
R = np.array(rows)
tr = R[:, 0]
names = ["S1 global cosine (current)", "S2 mean per-user cosine", "S3 min per-user cosine"]
def auc(score, y):  # P(score of 'bad' pair < score of 'good' pair); y True = bad (loss>10%)
    r = rankdata(score); n1, n0 = y.sum(), (~y).sum()
    return 1 - (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
bad = tr < 0.9
print(f"pairs={len(tr)}  mean transfer={tr.mean():.3f}  frac(loss>10%)={bad.mean():.2f}")
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for i, ax in enumerate(axes):
    s = R[:, 1 + i]
    rho_s = spearmanr(s, tr).correlation
    a = auc(s, bad)
    print(f"{names[i]:32s} Spearman={rho_s:+.3f}  AUC(detect >10% loss)={a:.3f}")
    ax.scatter(s, tr, s=2, alpha=0.15, color="#2980B9")
    ax.set_xlabel("similarity"); ax.set_title(f"{names[i]}\n$\\rho_s$={rho_s:.2f}, AUC={a:.2f}", fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("solution transfer ratio")
fig.savefig("paper_figures/Fig11_drift_metric_diagnostic.png"); print("saved Fig11")
