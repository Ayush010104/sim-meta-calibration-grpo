import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 220, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
R = np.load("meta_calib_eval.npy", allow_pickle=True).item()
GRAY, ORANGE, INK, GRID, MUTE = "#8a8984", "#eb6834", "#0b0b0b", "#e4e3df", "#52514e"
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
for arm, c, lab in (("nominal-init", GRAY, "Calibration-GRPO from nominal model"), ("meta-init", ORANGE, "Calibration-GRPO from META prior (proposed)")):
    cs = np.array([[r for _, r in R[f"{d}_{arm}"]["curve"]] for d in (900, 901, 902)])
    us = [u for u, _ in R[f"900_{arm}"]["curve"]]
    for row in cs: ax[0].plot(us, row, color=c, lw=0.8, alpha=0.35)
    ax[0].plot(us, cs.mean(0), color=c, lw=2.4, marker="o", ms=4, label=lab + " (mean of 3 devices)")
ax[0].set_xlabel("calibration updates on the NEW device (32 guesses × 6 rate measurements each)")
ax[0].set_ylabel("worst-user rate on the device (10 dev channels)")
ax[0].set_title("(a) A new, unseen device: rate vs measurement budget", loc="left", fontsize=10.5)
ax[0].yaxis.grid(True, color=GRID); ax[0].set_axisbelow(True); ax[0].set_ylim(0.0, 0.32); ax[0].legend(fontsize=8, frameon=False, loc="upper left")
w = 0.2; xs = np.arange(3)
keys = (("nominal", GRAY, "Gradient, nominal model", "#c9c8c3"), ("nominal-init", GRAY, "Calibrated 100 updates from nominal", GRAY),
        ("meta-init", ORANGE, "Calibrated 100 updates from META prior", ORANGE), ("oracle", INK, "Oracle device model", "#3b3b39"))
for i, (k, _, lab, col) in enumerate(keys):
    vals = [R[f"{d}_meta-init"][k].mean() if k in ("nominal", "oracle") else R[f"{d}_{k}"]["test"].mean() for d in (900, 901, 902)]
    b = ax[1].bar(xs + (i - 1.5) * (w + 0.01), vals, w, color=col, label=lab)
    for x, v in zip(xs + (i - 1.5) * (w + 0.01), vals): ax[1].text(x, v + 0.006, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
ax[1].set_xticks(xs); ax[1].set_xticklabels(["held-out device 1", "device 2", "device 3"])
ax[1].set_ylabel("worst-user rate, 40 test channels"); ax[1].set_ylim(0, 0.45)
ax[1].set_title("(b) After 100 updates: test channels", loc="left", fontsize=10.5)
ax[1].yaxis.grid(True, color=GRID); ax[1].set_axisbelow(True); ax[1].legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig("paper_figures/Fig18_meta_calibration_new_devices.png"); print("ok")
