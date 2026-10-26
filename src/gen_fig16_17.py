import re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl
mpl.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRAY, BLUE, ORANGE = "#8a8984", "#2a78d6", "#eb6834"

# ---------- Fig 16: schematic ----------
fig, ax = plt.subplots(figsize=(11.5, 5.4)); ax.set_xlim(0, 100); ax.set_ylim(-6, 54); ax.axis("off")
def box(x, y, w, h, txt, fc="#f4f3ef", ec="#9a9994", bold=False, col=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.2, color=col, weight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2, col=MUTE, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.4, color=col, ls=ls, connectionstyle=f"arc3,rad={rad}"))
ax.text(1, 52, "A. Normal gradient method", fontsize=12, weight="bold", color=INK)
box(1, 38, 20, 9, "Assumed model\n(ideal: no phase error,\nequal amplitudes)", fc="#e9eef7", ec=BLUE)
box(28, 38, 20, 9, "Relaxed gradient steps\nx = tanh(βw), β: 1→5\nexact derivative of the\nASSUMED rate")
box(55, 38, 16, 9, "Round: sign(w)\n→ phases 0 / π")
box(78, 38, 20, 9, "REAL device\n(hidden phase errors,\namplitude imbalance)", fc="#fbeee8", ec=ORANGE)
arrow(21.5, 42.5, 27.5, 42.5); arrow(48.5, 42.5, 54.5, 42.5); arrow(71.5, 42.5, 77.5, 42.5)
ax.text(50, 33, "The derivative is of the wrong function: it points toward the best configuration of a device that does not exist.\nNo information about the real device ever flows back.  One pass, then done.", ha="center", fontsize=9.2, color=MUTE)
ax.plot([0, 100], [28, 28], color=GRID, lw=1.5)
ax.text(1, 24.5, "B. Calibration RL (GRPO) wraps the SAME gradient method", fontsize=12, weight="bold", color=INK)
box(1, 9, 22, 11, "Policy over device\nparameters: 384 phase\noffsets, amplitudes,\nlayer spacing", fc="#e9eef7", ec=BLUE, bold=True)
box(27, 10, 17, 9, "Candidate model\n(one guess)")
box(49, 10, 17, 9, "Same relaxed steps\nx = tanh(βw), then\nsign(w)")
box(71, 10, 17, 9, "REAL device\nmeasure min-rate", fc="#fbeee8", ec=ORANGE, bold=True)
arrow(21.5, 14.5, 26.5, 14.5); arrow(44.5, 14.5, 48.5, 14.5); arrow(66.5, 14.5, 70.5, 14.5)
box(66, -4, 22, 6, "Reward = measured rate", fc="#fdf5e0", ec="#eda100")
arrow(79.5, 9.6, 79.5, 2.4)
arrow(65.5, -1, 12, 8.6, col=ORANGE, rad=-0.05)
ax.text(36, -4.2, "GRPO update: keep guesses that scored above their group", fontsize=9, color=ORANGE, ha="center")
ax.text(91.5, 14.5, "× 48\nguesses\nper update", fontsize=8.5, color=MUTE, va="center")
fig.savefig("paper_figures/Fig16_gradient_vs_rl_schematic.png"); plt.close(fig)

# ---------- Fig 17: results ----------
R = {s: np.load(f"calibration_lit_{s}.npy", allow_pickle=True).item() for s in ("amp", "ampsp")}
names = [("nominal-model gradient", "Gradient, ideal model", GRAY),
         ("amplitude-aware gradient (knows beta(phi))", "Gradient, amplitude-aware model", BLUE),
         ("CALIBRATED (RL) gradient", "Gradient + calibration RL (proposed)", ORANGE)]
fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
def ci(a): return 1.96 * a.std() / np.sqrt(len(a))

ax = axs[0, 0]
w = 0.26
for si, (s, lab) in enumerate((("amp", "Phase error +\namplitude imbalance"), ("ampsp", "…plus 5% layer-\nspacing error"))):
    for k, (key, nm, c) in enumerate(names):
        a, p = R[s][key], R[s][key + "+polish"]
        x = si + (k - 1) * (w + 0.02)
        ax.bar(x, a.mean(), w, color=c, label=nm if si == 0 else None, yerr=ci(a), error_kw=dict(lw=1, ecolor=INK, capsize=2))
        ax.text(x, a.mean() + ci(a) + 0.012, f"{a.mean():.2f}", ha="center", fontsize=8.5, color=INK)
    o = R[s]["oracle-model gradient"].mean()
    ax.hlines(o, si - 0.46, si + 0.46, colors=INK, linestyles="--", lw=1.3, label="Oracle model (knows everything)" if si == 0 else None)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Phase error +\namplitude imbalance", "…plus 5% layer-\nspacing error"])
ax.set_ylabel("min-rate on the real device (bit/s/Hz)"); ax.set_ylim(0, 0.66); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("(a) Same gradient optimizer, different model", fontsize=10.5, loc="left"); ax.legend(fontsize=8, frameon=False, loc="upper left")

ax = axs[0, 1]
for s, mk, lab in (("amp", "o", "phase err + amplitude"), ("ampsp", "s", "…+ spacing err")):
    xa, ya = R[s]["amplitude-aware gradient (knows beta(phi))"], R[s]["CALIBRATED (RL) gradient"]
    ax.scatter(xa, ya, s=22, marker=mk, color=ORANGE if s == "amp" else "#b8470f", alpha=0.75, edgecolor="#fcfcfb", lw=0.6, label=f"{lab}: RL above on {100*np.mean(ya>xa):.0f}%")
ax.plot([0, 0.5], [0, 0.5], color=INK, lw=1, ls=":"); ax.text(0.33, 0.30, "equal", fontsize=8.5, color=MUTE, rotation=38)
ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5); ax.set_xlabel("amplitude-aware gradient, per test channel"); ax.set_ylabel("calibrated RL gradient, same channel")
ax.yaxis.grid(True, color=GRID); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("(b) Channel by channel (40 test channels)", fontsize=10.5, loc="left"); ax.legend(fontsize=8, frameon=False, loc="upper left")

def parse(s):
    rows = []
    for line in open(f"calibration_lit_{ {'amp':'amp','ampsp':'ampsp'}[s] }.log"):
        m = re.match(r"u=\s*(\d+) dev deploy ([\d.]+) \| a0_hat ([\d.]+) api_hat ([\d.]+) delta_hat ([+-][\d.]+)", line)
        if m: rows.append([float(v) for v in m.groups()])
    return np.array(rows)
ref = {"amp": (0.096, 0.141, 0.367), "ampsp": (0.086, 0.135, 0.348)}
ax = axs[1, 0]
for s, ls in (("amp", "-"), ("ampsp", "--")):
    d = parse(s)
    ax.plot(d[:, 0], d[:, 1], color=ORANGE, ls=ls, lw=2, label="RL calibration, " + ("no spacing err" if s == "amp" else "with spacing err"))
for v, c, nm in ((np.mean([ref[s][0] for s in ref]), GRAY, "ideal model"), (np.mean([ref[s][1] for s in ref]), BLUE, "amplitude-aware model"), (np.mean([ref[s][2] for s in ref]), INK, "oracle model")):
    ax.hlines(v, 0, 400, colors=c, lw=1.2, linestyles=":" if c == INK else "-", alpha=0.9); ax.text(406, v, nm, fontsize=8.5, va="center", color=MUTE)
ax.set_xlim(0, 400); ax.set_xlabel("RL updates (48 guesses each, 6 channels each)"); ax.set_ylabel("rate with current best model (10 dev channels)")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_title("(c) Learning from measured rate only", fontsize=10.5, loc="left"); ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_xlim(0, 480); ax.set_ylim(0.07, 0.46)

ax = axs[1, 1]
d, d0 = parse("ampsp"), parse("amp")
ax.plot(d[:, 0], d[:, 3] / 0.697, color=BLUE, lw=2, label="weak-state amplitude (true 0.70)")
ax.plot(d[:, 0], d[:, 4] / 0.05, color=ORANGE, lw=2, label="layer-spacing error (true +5%)")
ax.plot(d0[:, 0], d0[:, 4] / 0.05, color=ORANGE, lw=1.2, ls="--", alpha=0.7, label="spacing error, scenario with none (should be 0; shown ÷0.05)")
ax.axhline(1, color=INK, lw=1, ls=":"); ax.axhline(0, color=GRID, lw=1)
ax.text(2, 1.04, "estimate = truth", fontsize=8.5, color=MUTE)
ax.set_xlabel("RL updates"); ax.set_ylabel("learned value ÷ true value"); ax.set_ylim(-0.6, 2.1)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_title("(d) The agent recovers physical parameters", fontsize=10.5, loc="left"); ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("paper_figures/Fig17_gradient_vs_rl_results.png"); plt.close(fig)
print("ok")
