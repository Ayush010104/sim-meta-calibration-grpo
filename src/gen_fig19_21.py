"""Fig19: single-impairment sensitivity of model-based design (3 system sizes). Fig20: Calibration-GRPO across configurations.
Fig21: operation under stale/noisy CSI (probing, warm start, pre-trained RL start policy)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRAY, BLUE, ORANGE, AQUA, DARK = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#3b3b39"

# ---------------- Fig 19 ----------------
CFG = (("5_36_4", "small: 5 users, 4×36", BLUE), ("8_49_5", "medium: 8 users, 5×49", ORANGE), ("10_64_6", "large: 10 users, 6×64", AQUA))
PAN = (("psi", "phase-tuning error σ (rad)", None), ("weak", "weak-state amplitude", None), ("offset", "lateral layer offset (pitch)", None),
       ("spacing", "layer-spacing error (fraction)", None), ("csi_db", "CSI error NMSE (dB)", None), ("amp_elem", "per-element amplitude spread*", None),
       ("stuck", "stuck elements (fraction)*", None))
S = {c: json.load(open(f"sens_{c}.json")) for c, _, _ in CFG}
fig, axs = plt.subplots(2, 4, figsize=(13, 6.2)); axs = axs.ravel()
for i, (key, lab, _) in enumerate(PAN):
    ax = axs[i]
    for c, nm, col in CFG:
        items = [(float(k.split("=")[1]), v) for k, v in S[c].items() if k.startswith(key + "=")]
        items.sort()
        xs = [x for x, _ in items]; ys = [100 * np.mean(v["ideal"]) / np.mean(v["oracle"]) for _, v in items]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=col, label=nm)
    ax.axhline(100, color=MUTE, lw=0.8, ls=":")
    ax.set_xlabel(lab, fontsize=9); ax.set_ylim(0, 112); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    if key == "weak": ax.invert_xaxis()
    if i % 4 == 0: ax.set_ylabel("rate kept by model-based design (% of oracle)", fontsize=8.5)
axs[7].axis("off")
h, l = axs[0].get_legend_handles_labels()
axs[7].legend(h, l, loc="center", frameon=False, fontsize=9.5, title="system size", title_fontsize=9.5)
axs[7].text(0.5, 0.12, "* not representable by the calibration model\nmean of 3 device draws, 20 test channels", ha="center", fontsize=8, color=MUTE, transform=axs[7].transAxes)
fig.suptitle("How much each hardware / CSI impairment costs a model-based (gradient) design", x=0.02, ha="left", fontsize=11.5)
fig.tight_layout(); fig.savefig("paper_figures/Fig19_impairment_sensitivity.png"); plt.close(fig)

# ---------------- Fig 20 ----------------
RUNS = (("c5_mod", "small\nmoderate"), ("c8_mod", "medium\nmoderate"), ("c10_mild", "large\nmild"), ("c10_mod", "large\nmoderate"),
        ("c10_mod_dev21", "large moderate\n(2nd device)"), ("c10_unmod", "large moderate\n+ unmodeled*"), ("c10_sev", "large\nsevere"))
KEYS = (("ideal-model gradient", "Gradient, ideal model", "#c9c8c3"), ("amplitude-aware gradient", "Gradient, amplitude-aware", BLUE),
        ("CALIBRATED (RL) gradient", "Gradient + Calibration-GRPO", ORANGE), ("oracle, full device", "Gradient, oracle device (same CSI)", DARK))
C = {t: np.load(f"calcfg_{t}.npy", allow_pickle=True).item() for t, _ in RUNS}
fig, ax = plt.subplots(figsize=(13, 4.6)); w = 0.2; xs = np.arange(len(RUNS))
for k, (key, lab, col) in enumerate(KEYS):
    vals = [C[t][key].mean() for t, _ in RUNS]
    ax.bar(xs + (k - 1.5) * (w + 0.01), vals, w, color=col, label=lab)
for i, (t, _) in enumerate(RUNS):
    cal, orc = C[t]["CALIBRATED (RL) gradient"].mean(), C[t]["oracle, full device"].mean()
    ax.text(xs[i] + 0.5 * (w + 0.01), cal + 0.015, f"{100*cal/orc:.0f}%", ha="center", fontsize=8, color=ORANGE, weight="bold")
ax.set_xticks(xs); ax.set_xticklabels([n for _, n in RUNS], fontsize=8.8)
ax.set_ylabel("worst-user rate, 40 test channels (bit/s/Hz)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper right")
ax.set_title("Calibration-GRPO across system sizes, impairment levels and devices (orange % = share of oracle reached; CSI error included)", loc="left", fontsize=10.5)
ax.text(0.99, -0.27, "* per-element amplitude spread + 3% stuck elements, outside the calibration model", transform=ax.transAxes, ha="right", fontsize=8, color=MUTE)
fig.tight_layout(); fig.savefig("paper_figures/Fig20_calibration_configurations.png"); plt.close(fig)

# ---------------- Fig 21 ----------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1, 1.35]))
ax = axs[0]
P = {s: np.load(f"probe_test_S{s}.npy", allow_pickle=True).item() for s in (4, 8)}
arms = (("A", "blind design\non stale CSI", "#c9c8c3"), ("B", "probe random\nrestarts", GRAY), ("D", "probe warm\nstarts", BLUE), ("C", "probe + GRPO\nknobs", ORANGE))
w = 0.38
for j, s in enumerate((4, 8)):
    vals = [P[s][a]["rates"].mean() for a, _, _ in arms]
    bars = ax.bar(np.arange(4) + (j - 0.5) * w, vals, w * 0.95, color=[c for _, _, c in arms], alpha=0.65 if s == 4 else 1.0)
    for x, v in zip(np.arange(4) + (j - 0.5) * w, vals): ax.text(x, v + 0.008, f"{v:.2f}", ha="center", fontsize=7.5)
ax.axhline(P[8]["O"]["rates"].mean(), color=INK, ls="--", lw=1.2); ax.text(3.45, P[8]["O"]["rates"].mean() + 0.01, "single design,\ntrue channel", fontsize=7.5, ha="right", color=MUTE)
ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n, _ in arms], fontsize=8.5)
ax.set_ylabel("worst-user rate (true channel)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("(a) slow fading ρ=0.99, 1.4 m/s: 4 (light) / 8 (dark) probes", loc="left", fontsize=10)
ax = axs[1]
Q = {v: np.load(f"probe_rl_rho0.7_v{v}_S8_n50.npy", allow_pickle=True).item() for v in ("3.0", "10.0")}
arms2 = (("A", "blind", "#c9c8c3"), ("B", "random\nrestarts", GRAY), ("D", "warm\nstarts", BLUE), ("E-scratch", "RL policy,\nno pre-training", AQUA), ("E-pre", "RL policy,\npre-trained", ORANGE))
for j, v in enumerate(("3.0", "10.0")):
    vals = [Q[v][a].mean() for a, _, _ in arms2]
    xs = np.arange(5) + (j - 0.5) * w
    ax.bar(xs, vals, w * 0.95, color=[c for _, _, c in arms2], alpha=0.65 if v == "3.0" else 1.0)
    for x, val in zip(xs, vals): ax.text(x, val + 0.008, f"{val:.2f}", ha="center", fontsize=7.5)
for v, ls in (("3.0", ":"), ("10.0", "--")):
    ax.axhline(Q[v]["O"].mean(), color=INK, ls=ls, lw=1.0)
ax.text(4.45, Q["10.0"]["O"].mean() + 0.01, "true-channel design", fontsize=7.5, ha="right", color=MUTE)
ax.set_xticks(range(5)); ax.set_xticklabels([n for _, n, _ in arms2], fontsize=8.5)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("(b) fast fading ρ=0.7: 3 m/s (light) / 10 m/s (dark), 8 probes per step", loc="left", fontsize=10)
fig.suptitle("Normal operation with stale + noisy CSI (calibrated device; 4 held-out trajectories × 100 steps)", x=0.02, ha="left", fontsize=11.5)
fig.tight_layout(); fig.savefig("paper_figures/Fig21_csi_probing_operation.png"); plt.close(fig)
print("ok")
