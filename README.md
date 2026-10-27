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

## In-depth investigation of impairments and configurations

### Which impairments matter (Fig. 19)

Each impairment is switched on alone and swept in size. The table gives the rate a model-based gradient design keeps, as a percentage of an oracle that knows the impairment. It covers three system sizes and 3 device draws per point.

| Impairment | Small (5 users, 4×36) | Medium (8 users, 5×49) | Large (10 users, 6×64) |
|---|---|---|---|
| Phase-tuning error 0.1 / 0.2 / 0.32 / 0.5 rad | 91 / 76 / 50 / 24% | 94 / 70 / 41 / 10% | 84 / 62 / 29 / **6%** |
| Weak-state amplitude 0.85 / 0.7 / 0.5 | 89 / 70 / 37% | 87 / 60 / 22% | 82 / 52 / **15%** |
| Lateral layer offset 0.05 / 0.1 / 0.2 pitch | 93 / 71 / 38% | 93 / 67 / 21% | 85 / 55 / **12%** |
| Layer spacing 1 / 3 / 5% | 99 / 98 / 95% | ≈100% | 103 / 99 / 97% |
| CSI error −15 / −10 / −5 dB | 96 / 83 / 55% | 91 / 76 / 51% | 91 / 78 / 49% |
| Per-element amplitude spread 10 / 20% (not modeled) | 88 / 69% | 95 / 73% | 90 / 59% |
| Stuck elements 1 / 3 / 5% (not modeled) | 86 / 72 / 67% | 77 / 52 / 47% | 75 / 45 / **30%** |

- **Larger surfaces are more fragile.** Every impairment hurts more on the large system.
- **The costly impairments** are phase errors, amplitude imbalance, lateral misalignment and stuck elements. Spacing errors up to 5% are nearly harmless.

### Calibration-GRPO across sizes, levels and devices (Fig. 20)

All rows include CSI error. Rates are worst-user rates on 40 test channels.

| Configuration | Ideal-model gradient | Amplitude-aware | **Calibration-GRPO** | Oracle | Share of oracle |
|---|---|---|---|---|---|
| Small, moderate | 0.277 | 0.405 | **0.781** | 0.819 | **95%** |
| Medium, moderate | 0.079 | 0.155 | **0.392** | 0.441 | **89%** |
| Large, mild | 0.198 | 0.281 | **0.390** | 0.446 | 87% |
| Large, moderate | 0.033 | 0.061 | **0.259** | 0.332 | 78% |
| Large, moderate, 2nd device | 0.038 | 0.118 | **0.262** | 0.354 | 74% |
| Large, moderate + unmodeled errors | 0.020 | 0.044 | **0.114** | 0.346 | 33% |
| Large, severe | 0.003 | 0.009 | 0.013 | 0.167 | 8% (fails) |

- **Gain over the amplitude-aware design:**
  - small, moderate: +0.376 ± 0.084;
  - medium, moderate: +0.238 ± 0.025 (100% of channels);
  - large, moderate: +0.198 ± 0.024 (100%);
  - large, moderate, 2nd device: +0.144 ± 0.026 (92%).
- **Limits:**
  - At the severe level every candidate model gives near-zero rate, so the reward carries no signal.
  - Errors outside the calibration model cap the gain.

### Normal operation with stale and noisy CSI (Fig. 21)

The device is calibrated. CSI is one step stale plus −10 dB noise, and users report noisy rates for a few probed configurations per step.

| Scenario (4 held-out trajectories × 100 steps) | Blind design | Probe random restarts | Probe warm starts | **Pre-trained RL start policy** | Single design, true channel |
|---|---|---|---|---|---|
| ρ = 0.99, 1.4 m/s, 8 probes | 0.293 | 0.374 | **0.550** | — | 0.367 |
| ρ = 0.7, 3 m/s, 8 probes | 0.188 | 0.270 | 0.368 | **0.377** | 0.339 |
| ρ = 0.7, 10 m/s, 8 probes | 0.084 | 0.158 | 0.183 | **0.202** | 0.346 |

- **Slow fading:** probing a few warm-started candidates recovers the whole CSI loss (+88% over blind design).
- **Fast fading at 10 m/s:**
  - An RL start policy, pre-trained offline across 30 simulated trajectories, beats the best heuristic by **+10%** (4/4 trajectories) and the blind design by **2.4×**.
  - The same policy without pre-training does not beat the heuristic.

### Offline meta-RL study (in progress)

`mrs_meta_rl_study.py` compares multi-task pre-training, Reptile, CAVIA-style context adaptation and a new feedback-conditioned in-context policy, under paired noise, on 8 held-out trajectories at ρ = 0.7 and 10 m/s.

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
| **Impairment library & sensitivity** | `impair_lib.py`, `sens_sweep.py` | Generic K/M/L system with modeled (phase, amplitude, spacing, offset) and unmodeled (per-element amplitude, stuck elements) impairments, CSI error and noisy feedback; single-impairment sweeps |
| **Calibration across configurations** | `calibration_cfg.py` | Calibration-GRPO for any (K, M, L) and impairment level, one device draw |
| **Operation under stale CSI** | `probe_test.py`, `probe_rl.py`, `probe_rl2.py` | Probing with warm starts; pre-trained RL start policy (GRPO + advantage-weighted regression); learned channel-prediction knob |
| **Offline meta-RL study** | `mrs_meta_rl_study.py`, `mrs_run_study.sh` | Multi-task pre-training, Reptile, CAVIA-style context, feedback-conditioned in-context policy; paired evaluation |
| **Figures** | `gen_fig16_17.py`, `gen_fig18.py`, `gen_fig19_21.py` | Gradient-vs-RL schematic and results; meta-calibration; sensitivity, configurations, CSI operation |

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

# impairment sensitivity (one impairment at a time) for three system sizes
python3 sens_sweep.py 5 36 4; python3 sens_sweep.py 8 49 5; python3 sens_sweep.py 10 64 6

# Calibration-GRPO across configurations: K M L level device updates candidates tag
python3 calibration_cfg.py 10 64 6 moderate 7 400 48 c10_mod

# operation under stale / noisy CSI
python3 probe_test.py 8 100                          # rho=0.99, probing with warm starts
python3 probe_rl.py 0.7 10.0 8 100 30 50 50          # rho=0.7, 10 m/s, pre-trained RL start policy

# offline meta-RL study (train 5 learned arms, test 6 arms, report)
./mrs_run_study.sh

# figures (they read the .npy files written above)
mkdir -p paper_figures && python3 gen_fig16_17.py && python3 gen_fig18.py && python3 gen_fig19_21.py
```

Logs and result files from the runs reported above are in [`results/logs`](results/logs) and [`results/data`](results/data).

## Citation

Manuscript in preparation. Citation details will be added on submission.

## License

Copyright © 2026 Ayush Brijesh Patel. All rights reserved.

This code accompanies a manuscript in preparation and is shared for review and reference only. No license is granted to copy, modify or redistribute it without written permission from the author.
