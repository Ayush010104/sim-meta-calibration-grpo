# Meta-Calibration GRPO for Imperfect 1-Bit Stacked Intelligent Metasurfaces

**Reinforcement learning that learns what the hardware actually is, from measured rates alone, and meta-learns that knowledge across devices.**

*Research code for a manuscript in preparation (IEEE Transactions–style). Ayush B. Patel & Abhishek Jindal, DAIICT.*

**Project page:** https://ayush010104.github.io/sim-meta-calibration-grpo/

---

## The problem in one paragraph

A stacked intelligent metasurface (SIM) is a stack of programmable metasurface layers that beamforms directly in the wave domain. With 1-bit elements (phase 0 or π), configuring it to serve several users fairly (max-min rate) is a combinatorial problem over 144–384 binary variables. An exact relaxed-gradient solver handles this very well **when the hardware model is perfect**. We verified that it beats our RL agent in that case. Real devices have phase-tuning errors, state-dependent amplitudes, inter-layer misalignment, imperfect CSI and noisy feedback, so every model is wrong. There, the gradient solver optimizes a device that does not exist and loses most of its rate. This repository contains the full path from that observation to the proposed fix.

## Proposed method

**Calibration-GRPO** is a bilevel loop:

1. **Outer loop (RL, critic-free GRPO):** a Gaussian policy over the *device model*, made of 384 phase offsets, per-layer state amplitudes, per-layer spacing and per-layer lateral offsets.
2. **Inner loop (gradient):** for each sampled device model, the relaxed-gradient solver (x = tanh(βw), β: 1→5, smooth min of user rates) designs a configuration, which is rounded to phases {0, π}.
3. **Reward:** the worst-user rate *measured on the real device*. It is scalar and non-differentiable, and no gradient ever passes through the hardware.
4. **Update:** group-relative advantages over 32–48 antithetic candidates, with common random numbers to reduce variance.

**Meta-Calibration-GRPO** wraps this in Reptile over a population of devices from one fabrication batch, so a new device starts from a learned prior instead of the ideal model.

## Headline results (simulation; K = 10 users, 6 × 64 elements = 384 bits)

| Setting (40 held-out test channels, real-device rate, phases ∈ {0, π}) | Gradient, ideal model | Gradient, amplitude-aware | **Gradient + Calibration-GRPO** | Gradient, oracle model |
|---|---|---|---|---|
| Phase error + amplitude imbalance | 0.065 | 0.170 | **0.310** | 0.411 |
| + 5% layer-spacing error | 0.056 | 0.161 | **0.291** | 0.386 |
| All impairments (incl. CSI NMSE −10 dB and 5% noisy feedback) | 0.032 | 0.081 | **0.276** | 0.328 (same CSI) |

- **All impairments:** **3.4×** the worst-user rate of the amplitude-aware design, better on **100%** of test channels, and **84%** of the best achievable with that CSI.
- **Physical parameters recovered from rates alone:** the weak-state amplitude (true 0.70, estimated 0.70–0.72), lateral layer offsets (error 0.10 → 0.018 element pitch), and phase offsets (error 0.32 → 0.21–0.23 rad).
- **Meta-calibration on 3 unseen devices:**
  - about **2.8×** the rate *before any measurement*;
  - matches from-scratch calibration with **5–10× fewer** rate measurements;
  - **+50–60%** final worst-user rate (3/3 devices).
- **Offline Reptile meta-initialization of GRPO** (144-bit system, 250 unseen mobile-user channels): **+0.26** bit/s/Hz at 50 updates [95% CI 0.24–0.28], better on 92%. It reaches the cold start's 50-update level in about 7 updates.

See [`figures/`](figures) and the project page in [`docs/`](docs).

## Repository map (`src/`)

All modules are flat, so run scripts from inside `src/`.

| Area | Files | What they do |
|---|---|---|
| **Physics & environments** | `env_ris.py`, `moving_user_env.py`, `physics.py`, `run_robustness_sweep.py` | SIM cascade, correlated Rician channels, moving users (ρ, v), numpy evaluators, channel-trajectory generator |
| **GRPO agent** | `grpo_ris.py` | von Mises actor, group-relative robust advantage, PPO clip, STE term, 1-bit quantization |
| **Mobility tracking & meta-priors** | `ensemble_parallel.py`, `ensemble_enhanced.py`, `ensemble_reset_variants.py`, `context_attention.py`, `run_stage1*.py`, `run_block1*.py`, `run_aware_meta.py`, `diag_drift_metric.py`, `pilot_incremental_mdp.py` | Staggered GRPO ensembles with change-point resets (Reptile meta-prior / random / never), attention-based reset rules, channel-aware policy, flip-MDP pilot |
| **Single-copy meta-learners** | `n1_learners.py`, `run_paper_n1.py`, `check_seqwarm.py` | seq-WARM, COLD, Reptile, periodic-EMA, proximal Reptile, FOMAML under identical change-points |
| **Offline meta-learning** | `meta_offline.py`, `adapt250.py`, `adapt500.py` | Reptile meta-initialization across a channel distribution; 50- and 500-update adaptation curves |
| **Gradient baselines & unrolled optimizers** | `unrolled_meta.py`, `diag_relax.py`, `diag_grad.py`, `verify_hard.py`, `unrolled_relax.py`, `adapt_curves.py`, `direct_ste_baseline.py`, `ao_layerwise_baseline.py`, `minlp_ris.py` | Differentiable torch physics, exact relaxed ±1 solver, three-way {0, π} verification, meta-learned unrolled optimizers, AO / MINLP baselines |
| **Hard system & mismatch diagnostics** | `big_sys.py`, `hard_cfg.py`, `diag_hard.py`, `hybrid_grpo.py`, `compare_hard.py`, `diag_calib.py` | K = 10, M = 64, L = 6 system; CSI / mismatch / multi-start diagnostics; per-channel hybrid GRPO vs random restarts; reward-landscape probe |
| **Calibration-GRPO** | `calibration_grpo.py`, `calibration_lit.py`, `calibration_full.py` | Phase-offset calibration; literature-style impairments (amplitude curve, spacing, one-channel vs pool); all impairments together (vectorized geometry with spacing and lateral offsets) |
| **Meta-Calibration-GRPO** | `meta_calibration.py` | Reptile over device-tasks (batch-shared + device-specific errors), held-out device evaluation |
| **Figures** | `gen_fig16_17.py`, `gen_fig18.py` | Gradient-vs-RL schematic and results; meta-calibration figure |

## Reproducing the main results

```bash
pip install -r requirements.txt
cd src

# exact gradient vs mismatch diagnostics
python3 diag_hard.py
python3 big_sys.py 10 64 6

# Calibration-GRPO, literature-style impairments (about 1 h each on 3 CPU threads)
python3 calibration_lit.py amp 400 48 0.012 amp
python3 calibration_lit.py amp_spacing 400 48 0.012 ampsp
python3 calibration_lit.py amp 400 48 0.012 one_ch one      # calibrate on ONE channel

# all impairments together
python3 calibration_full.py 500 48 all

# meta-calibration across devices (about 45 min training + evaluation)
python3 meta_calibration.py train 30 25
python3 meta_calibration.py eval 100

# figures (they read the .npy files written above)
mkdir -p paper_figures && python3 gen_fig16_17.py && python3 gen_fig18.py
```

Logs and result files from the runs reported above are in [`results/logs`](results/logs) and [`results/data`](results/data).

## Citation

Manuscript in preparation. Citation details will be added on submission.

## License

Copyright © 2026 Ayush Brijesh Patel. All rights reserved.

This code accompanies a manuscript in preparation and is shared for review and reference only. No license is granted to copy, modify or redistribute it without written permission from the author.
