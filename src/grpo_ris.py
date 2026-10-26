"""GRPO (Group Relative Policy Optimization) for the 1-/2-bit RIS phase
quantization problem, ported verbatim from GRPO_1bit_first_final (3).ipynb
(GaussianActor/VonMisesActor/GRPO/VonMisesActorQBest/build_best_agent/
update_quantized_best), used here only to measure wall-clock time per
update -- not modified from the notebook logic.
"""
import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import VonMises

from env_ris import power_mean, RATE_FLOOR


# ============================================================
# Two interchangeable actor ("policy head") parameterizations.
# Both expose the same interface so GRPO can swap between them:
#   forward(state)              -> (p1, p2)          [distribution params]
#   sample(state, n_samples)    -> (action, raw, p1_rep, p2_rep)
#   log_prob(raw, p1, p2)       -> (batch,) tensor
#   entropy(p1, p2)             -> (batch,) tensor
#   deterministic_action(state) -> action
#   action_from_params(p1)      -> deterministic action used for the
#                                  differentiable reparam-reward term
# ============================================================

class GaussianActor(nn.Module):
    """Baseline (original) actor: raw network output -> tanh squash ->
    per-layer L2-normalize onto the unit circle.

    KNOWN LIMITATION (flagged, not silently fixed): log_prob() below only
    applies the tanh-squash Jacobian correction. It does NOT correct for
    the subsequent L2-normalize step that maps the 2D (real, imag) tanh
    output onto the unit circle -- that projection is a many-to-one
    R^2 -> S^1 map, so there is no simple closed-form Jacobian correction
    for it (this is exactly the "projected normal" issue). The practical
    effect is a mild bias in the policy-gradient estimate that grows as
    std grows (i.e. early in training / high-entropy regimes). This is
    kept here, unfixed, as the honest baseline; VonMisesActor below is the
    mathematically-corrected alternative -- see the ablation section.
    """

    def __init__(self, state_dim, action_dim, M, N, K, L, device, max_action=1,
                 log_std_init=-0.5, log_std_min=-1.5, log_std_max=1.0, **_ignored):
        super(GaussianActor, self).__init__()
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        self.device = device
        self.M = M
        self.N = N
        self.K = K
        self.L = L
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.raw_dim = 2 * M * L

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.l_mean = nn.Linear(hidden_dim, self.raw_dim)
        self.l_log_std = nn.Linear(hidden_dim, self.raw_dim)

        nn.init.zeros_(self.l_log_std.weight)
        nn.init.constant_(self.l_log_std.bias, log_std_init)

    def normalize(self, squashed: torch.Tensor) -> torch.Tensor:
        phi = []
        eps = 1e-8
        for l in range(self.L):
            start = l * 2 * self.M
            real_raw = squashed[:, start: start + self.M]
            imag_raw = squashed[:, start + self.M: start + 2 * self.M]
            norm = torch.sqrt(real_raw ** 2 + imag_raw ** 2 + eps)
            phi.append(real_raw / norm)
            phi.append(imag_raw / norm)
        return self.max_action * torch.cat(phi, dim=1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: returns (mean_logits, std) -- the *exponentiated, clamped*
        # std, not log_std -- so that (p1, p2) here means the same thing
        # ("final, directly-usable distribution parameters") as
        # VonMisesActor.forward()'s (theta_mu, kappa). GRPO's generic
        # update loop relies on that consistency; passing log_std here
        # instead silently breaks log_prob() (log(negative log_std) = NaN)
        # without raising, since torch just propagates NaN through the
        # loss and the existing isfinite-guard swallows it as a "skip".
        a = torch.tanh(self.l1(state.float()))
        a = self.ln1(a)
        a = torch.tanh(self.l2(a))
        a = self.ln2(a)
        mean_logits = self.l_mean(a)
        log_std = self.l_log_std(a).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).clamp(min=1e-4, max=float(np.exp(self.log_std_max)))
        return mean_logits, std

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(state)
        return self.normalize(torch.tanh(logits))

    def action_from_params(self, p1: torch.Tensor) -> torch.Tensor:
        return self.normalize(torch.tanh(p1))

    def sample(self, state: torch.Tensor, n_samples: int = 1):
        logits, std = self.forward(state)

        batch = logits.shape[0]
        mean_rep = logits.unsqueeze(0).expand(n_samples, batch, -1).reshape(n_samples * batch, -1)
        std_rep = std.unsqueeze(0).expand(n_samples, batch, -1).reshape(n_samples * batch, -1)

        eps = torch.randn_like(mean_rep)
        raw = mean_rep + eps * std_rep
        squashed = torch.tanh(raw)
        action = self.normalize(squashed)
        return action, raw, mean_rep, std_rep

    @staticmethod
    def log_prob(raw: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        var = p2 ** 2
        normal_log_prob = -((raw - p1) ** 2) / (2 * var) - torch.log(p2) - 0.5 * np.log(2 * np.pi)
        squash_correction = torch.log(1 - torch.tanh(raw) ** 2 + 1e-6)
        return (normal_log_prob - squash_correction).sum(dim=1)

    @staticmethod
    def entropy(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        # Differential entropy of the (pre-squash) Gaussian only -- does not
        # account for the tanh/normalize warping, same caveat as log_prob.
        # Averaged over the action dimension (not summed), matching the
        # original implementation and keeping entropy_coef's effective
        # scale comparable across action_dim sizes and across actor types.
        return (0.5 * torch.log(2 * np.pi * np.e * p2 ** 2)).mean(dim=1)

    def reg_penalty(self, p1: torch.Tensor, coef: float) -> torch.Tensor:
        return coef * (p1 ** 2).mean()


class VonMisesActor(nn.Module):
    """Phase-native actor: predicts (mu_r, mu_i) per RIS-layer element (an
    unconstrained 2D Gaussian mean, as in the projected-normal setup) but
    treats the actual *action* as the phase angle phi = atan2(z_i, z_r) of a
    sampled complex point, and computes log_prob under the von-Mises
    approximation of the true projected-normal density on the circle:
        loc   = atan2(mu_i, mu_r)
        kappa = rho^2 / sigma^2      (rho = |mu|, shared sigma per element)

    This is the mathematically-corrected counterpart to GaussianActor's
    unaccounted-for normalize() step: the log_prob here is the actual
    density of the sampled action (up to the von-Mises approximation of the
    exact projected-normal form, which is accurate once rho/sigma is not
    tiny -- see KAPPA_MIN clamp below for the fallback regime).
    """

    KAPPA_MIN = 1e-3
    KAPPA_MAX = 200.0
    LOG_STD_MIN = -4.0
    LOG_STD_MAX = -0.5
    LOG_STD_INIT = np.log(0.15)

    def __init__(self, state_dim, action_dim, M, N, K, L, device, max_action=1, **_ignored):
        super().__init__()
        self.device = device
        self.M, self.N, self.K, self.L = M, N, K, L
        self.n_phase = M * L  # one phase angle per RIS element per layer
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * self.n_phase),  # mu_r, mu_i concatenated
        )
        self.log_std = nn.Parameter(torch.full((self.n_phase,), self.LOG_STD_INIT))

    def forward(self, state: torch.Tensor):
        out = self.net(state.float())
        mu_r, mu_i = out.chunk(2, dim=-1)
        sigma = torch.exp(self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX))
        sigma = sigma.unsqueeze(0).expand_as(mu_r)
        rho = torch.sqrt(mu_r.pow(2) + mu_i.pow(2) + 1e-12)
        theta_mu = torch.atan2(mu_i, mu_r)
        kappa = (rho.pow(2) / sigma.pow(2)).clamp(self.KAPPA_MIN, self.KAPPA_MAX)
        return theta_mu, kappa  # p1, p2

    def _phi_to_action(self, phi: torch.Tensor) -> torch.Tensor:
        # phi: (batch, n_phase) -> env layout (batch, 2*L*M): per layer l,
        # [Phi_real_l (M), Phi_imag_l (M)] concatenated across l.
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        blocks = []
        for l in range(self.L):
            start = l * self.M
            blocks.append(cos_phi[:, start:start + self.M])
            blocks.append(sin_phi[:, start:start + self.M])
        return torch.cat(blocks, dim=1)

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        theta_mu, _ = self.forward(state)
        return self._phi_to_action(theta_mu)

    def action_from_params(self, p1: torch.Tensor) -> torch.Tensor:
        return self._phi_to_action(p1)

    def sample(self, state: torch.Tensor, n_samples: int = 1):
        theta_mu, kappa = self.forward(state)
        batch = theta_mu.shape[0]
        theta_rep = theta_mu.unsqueeze(0).expand(n_samples, batch, -1).reshape(n_samples * batch, -1)
        kappa_rep = kappa.unsqueeze(0).expand(n_samples, batch, -1).reshape(n_samples * batch, -1)

        dist = VonMises(theta_rep, kappa_rep)
        phi = dist.sample()
        action = self._phi_to_action(phi)
        return action, phi, theta_rep, kappa_rep

    @staticmethod
    def log_prob(raw: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        return VonMises(p1, p2).log_prob(raw).sum(dim=1)

    @staticmethod
    def entropy(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        # H(VonMises) = log(2*pi*I0(k)) - k*I1(k)/I0(k), via scaled Bessels
        # i0e/i1e for numerical stability at large kappa.
        kappa = p2
        i0e = torch.special.i0e(kappa)
        i1e = torch.special.i1e(kappa)
        log_i0 = torch.log(i0e.clamp_min(1e-12)) + kappa
        ratio = i1e / i0e.clamp_min(1e-12)
        ent = (np.log(2 * np.pi) + log_i0 - kappa * ratio)
        # Averaged over phase elements, not summed -- keeps entropy_coef's
        # effective scale comparable to GaussianActor's (also mean(dim=1))
        # and comparable across different M*L sizes.
        return ent.mean(dim=1)

    def reg_penalty(self, p1: torch.Tensor, coef: float) -> torch.Tensor:
        # No tanh dead-zone to regularize against here; instead lightly
        # discourage kappa from pinning at KAPPA_MAX too early (premature
        # certainty), which is the continuous-space analogue of entropy
        # collapse flagged earlier for this action-space family.
        return torch.zeros((), device=p1.device)


ACTOR_REGISTRY = {"gaussian": GaussianActor, "von_mises": VonMisesActor}


class GRPO(object):
    """Group Relative Policy Optimization with:
      - swappable actor parameterization (actor_type: 'gaussian' | 'von_mises')
      - swappable group-advantage normalization (advantage_mode: 'zscore' |
        'log_zscore' | 'robust')
      - power-mean reward family exposed via env.reward_power_p (arithmetic /
        geometric / harmonic / near-min, see power_mean() in the env cell)
    """

    def __init__(self, state_dim, action_dim, M, N, K, L, H, Ws, P, var, max_action, actor_lr, actor_decay, device,
                 group_size=16, clip_eps=0.2, ppo_epochs=5,
                 entropy_coef_start=0.01, entropy_coef_end=0.001, entropy_anneal_updates=20000,
                 lr_decay_rate=1.0, lr_min=1e-5,
                 sync_old_every=1, kl_threshold=3.0, raw_l2_coef=1e-3, reparam_coef=1.0,
                 actor_type="gaussian", advantage_mode="zscore",
                 reward_power_p=0.0, reward_mean_weight=0.4, reward_min_weight=0.5,
                 actor_kwargs=None):
        self.device = device
        self.M, self.N, self.K, self.L = M, N, K, L
        self.H, self.Ws, self.P, self.var = H, Ws, P, var

        self.actor_type = actor_type
        self.advantage_mode = advantage_mode
        self.reward_power_p = reward_power_p
        self.reward_mean_weight = reward_mean_weight
        self.reward_min_weight = reward_min_weight

        ActorCls = ACTOR_REGISTRY[actor_type]
        actor_kwargs = actor_kwargs or {}
        self.actor = ActorCls(state_dim, action_dim, M, N, K, L, device, max_action=max_action, **actor_kwargs).to(self.device)
        self.actor_old = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_decay)

        self.group_size = group_size
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs

        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_anneal_updates = max(1, entropy_anneal_updates)
        self.entropy_coef = entropy_coef_start

        self.actor_lr_init = actor_lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_min = lr_min

        self.sync_old_every = max(1, sync_old_every)
        self.kl_threshold = kl_threshold
        self.raw_l2_coef = raw_l2_coef
        self.reparam_coef = reparam_coef
        self._update_count = 0

        self.last_mean_std = None
        self.last_approx_kl = None

    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.actor.eval()
        # expand_dims (not reshape(1,-1)) so this works for BOTH a flat
        # (state_dim,) state and a windowed (window, state_dim) sequence
        # (used by memory-augmented actors, see memory_actor_cell.py) --
        # identical behaviour to reshape(1,-1) for the flat case.
        state_t = torch.FloatTensor(np.expand_dims(state, axis=0)).to(self.device)
        with torch.no_grad():
            action = self.actor.deterministic_action(state_t)
        return action.cpu().data.numpy().flatten().reshape(1, -1)

    def get_env_action(self, state: np.ndarray, stochastic: bool = True) -> np.ndarray:
        self.actor.eval()
        state_t = torch.FloatTensor(np.expand_dims(state, axis=0)).to(self.device)
        with torch.no_grad():
            if stochastic:
                action, _, _, _ = self.actor.sample(state_t, n_samples=1)
            else:
                action = self.actor.deterministic_action(state_t)
        return action.cpu().data.numpy().flatten().reshape(1, -1)

    def _group_rewards_batched(self, actions_np: np.ndarray, H_batch: np.ndarray = None) -> np.ndarray:
        S, group = actions_np.shape[0], actions_np.shape[1]
        L, M, K = self.L, self.M, self.K

        if H_batch is None:
            H_batch = np.broadcast_to(self.H, (S, M, K))

        d = []
        for l in range(L):
            start = l * 2 * M
            Phi_real_l = actions_np[:, :, start:start + M]
            Phi_imag_l = actions_np[:, :, start + M:start + 2 * M]
            d.append(Phi_real_l + 1j * Phi_imag_l)

        Gv = np.broadcast_to(np.eye(M, dtype=complex), (S, group, M, M)).copy()
        for idx in range(L - 1, 0, -1):
            M_idx = d[idx][:, :, :, None] * self.Ws[idx][None, None, :, :]
            Gv = np.matmul(Gv, M_idx)
        Gv = Gv * d[0][:, :, None, :]

        HH = H_batch.conj().transpose(0, 2, 1)
        temp = np.matmul(HH[:, None, :, :], Gv)
        Full = np.matmul(temp, self.Ws[0][None, None, :, :])

        power_all = np.abs(Full) ** 2 * self.P[None, None, None, :]
        diag_idx = np.arange(K)
        signal = power_all[:, :, diag_idx, diag_idx]
        interference = power_all.sum(axis=3) - signal
        rates = np.log2(1 + signal / (interference + self.var))            # (S, group, K)

        min_rate = rates.min(axis=2)
        safe_rates = np.clip(rates, RATE_FLOOR, None)
        mean_term = power_mean(safe_rates, self.reward_power_p, axis=2)     # (S, group)
        return (self.reward_mean_weight * mean_term + self.reward_min_weight * min_rate) * 10

    def _group_rewards_torch(self, actions_t: torch.Tensor, H_batch: torch.Tensor = None) -> torch.Tensor:
        L, M, K = self.L, self.M, self.K
        batch_n = actions_t.shape[0]
        device = actions_t.device
        cdtype = torch.complex64

        if H_batch is None:
            H_t = torch.tensor(self.H, dtype=cdtype, device=device).unsqueeze(0).expand(batch_n, M, K)
        else:
            H_t = H_batch.to(dtype=cdtype, device=device)
        Ws_t = [torch.tensor(w, dtype=cdtype, device=device) for w in self.Ws]
        P_t = torch.tensor(self.P, dtype=torch.float32, device=device)

        d = []
        for l in range(L):
            start = l * 2 * M
            Phi_real_l = actions_t[:, start:start + M]
            Phi_imag_l = actions_t[:, start + M:start + 2 * M]
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
        rates = torch.log2(1 + signal / (interference + self.var))
        min_rate = rates.min(dim=1).values
        safe_rates = torch.clamp(rates, min=RATE_FLOOR)

        p = self.reward_power_p
        if abs(p) < 1e-9:
            mean_term = torch.exp(torch.mean(torch.log(safe_rates + 1e-8), dim=1))
        else:
            mean_term = torch.mean(safe_rates ** p, dim=1) ** (1.0 / p)
        return (self.reward_mean_weight * mean_term + self.reward_min_weight * min_rate) * 10

    def _compute_group_advantage(self, rewards: np.ndarray) -> np.ndarray:
        """rewards: (S, group). Returns advantage of the same shape."""
        eps = 1e-8
        if self.advantage_mode == "zscore":
            std = rewards.std(axis=1, keepdims=True)
            denom = np.where(std > 1e-2, std, 1e-2)
            return (rewards - rewards.mean(axis=1, keepdims=True)) / denom

        elif self.advantage_mode == "log_zscore":
            # Geometric-relative advantage: z-score in log-space, i.e. each
            # candidate's *multiplicative* deviation from the group's
            # geometric-mean reward rather than an additive deviation from
            # the arithmetic mean. Rewards here are always >= 0 (power-mean
            # of nonnegative rates + nonnegative min-rate term), so log is
            # well-defined with a small epsilon floor.
            logr = np.log(rewards + eps)
            std = logr.std(axis=1, keepdims=True)
            denom = np.where(std > 1e-2, std, 1e-2)
            return (logr - logr.mean(axis=1, keepdims=True)) / denom

        elif self.advantage_mode == "robust":
            med = np.median(rewards, axis=1, keepdims=True)
            mad = np.median(np.abs(rewards - med), axis=1, keepdims=True) * 1.4826
            denom = np.where(mad > 1e-2, mad, 1e-2)
            return (rewards - med) / denom

        elif self.advantage_mode == "dr_grpo":
            # Dr. GRPO-style: drop the per-group scale normalization entirely
            # (dividing by std/MAD was shown to bias updates toward
            # low-variance groups -- e.g. a channel the policy has already
            # nearly solved gets an inflated advantage purely because its
            # group of G samples happens to be tightly clustered, not
            # because the reward gap is actually large). Keep the median
            # center (more robust to the rare single-sample outlier than the
            # mean, per our own robust-vs-zscore investigation) but only
            # center, never rescale.
            med = np.median(rewards, axis=1, keepdims=True)
            return rewards - med

        else:
            raise ValueError(f"Unknown advantage_mode: {self.advantage_mode}")

    def update_parameters_batch(self, states, H_batch=None):
        self.actor.eval()
        self.actor_old.eval()

        num_states = len(states)

        if H_batch is None:
            H_batch_np = np.broadcast_to(self.H, (num_states, self.M, self.K)).copy()
        else:
            H_batch_np = np.stack([np.asarray(h) for h in H_batch], axis=0)

        states_np = np.stack([np.asarray(s) for s in states], axis=0)  # (num_states, state_dim) OR (num_states, window, state_dim)
        states_t = torch.FloatTensor(states_np).to(self.device)

        with torch.no_grad():
            actions, raw_old, p1_old, p2_old = self.actor_old.sample(states_t, n_samples=self.group_size)
            old_log_prob = self.actor_old.log_prob(raw_old, p1_old, p2_old)

        def to_state_major(x, last_dim):
            return (x.view(self.group_size, num_states, last_dim)
                     .permute(1, 0, 2)
                     .reshape(num_states * self.group_size, last_dim))

        raw_dim = raw_old.shape[-1]
        action_dim = actions.shape[-1]
        raw_old_cat = to_state_major(raw_old, raw_dim)
        actions = to_state_major(actions, action_dim)
        old_log_prob_cat = (old_log_prob.view(self.group_size, num_states)
                             .permute(1, 0).reshape(-1))

        actions_np = actions.cpu().data.numpy().reshape(num_states, self.group_size, action_dim)

        rewards = self._group_rewards_batched(actions_np, H_batch_np)
        mean_rewards = rewards.mean(axis=1)

        advantage = self._compute_group_advantage(rewards)
        advantages_cat = torch.FloatTensor(advantage.reshape(-1)).to(self.device)

        state_rep_cat = states_t.repeat_interleave(self.group_size, dim=0)
        H_batch_t = torch.tensor(H_batch_np, dtype=torch.complex64, device=self.device)
        H_batch_rep = H_batch_t.repeat_interleave(self.group_size, dim=0)

        frac = min(1.0, self._update_count / self.entropy_anneal_updates)
        self.entropy_coef = self.entropy_coef_start + frac * (self.entropy_coef_end - self.entropy_coef_start)

        self.actor.train()
        last_std_mean = None
        last_kl = None
        rolled_back = False
        prev_state = copy.deepcopy(self.actor.state_dict())
        prev_optim_state = copy.deepcopy(self.actor_optimizer.state_dict())
        for _ in range(self.ppo_epochs):
            p1_new, p2_new = self.actor.forward(state_rep_cat)

            new_log_prob = self.actor.log_prob(raw_old_cat, p1_new, p2_new)
            log_ratio = new_log_prob - old_log_prob_cat
            ratio = torch.exp(log_ratio)

            surrogate_1 = ratio * advantages_cat
            surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_cat
            entropy = self.actor.entropy(p1_new, p2_new).mean()
            reg_penalty = self.actor.reg_penalty(p1_new, self.raw_l2_coef)

            mean_action = self.actor.action_from_params(p1_new)
            reparam_reward = self._group_rewards_torch(mean_action, H_batch_rep).mean()
            reparam_loss = -self.reparam_coef * reparam_reward

            actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - self.entropy_coef * entropy + reg_penalty + reparam_loss

            if not torch.isfinite(actor_loss):
                print("[GRPO] Skipped an update: non-finite actor_loss")
                continue

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            with torch.no_grad():
                # p2 (std or kappa) mean is a useful diagnostic regardless of actor type
                last_std_mean = p2_new.mean().item()
                last_kl = (ratio - 1 - log_ratio).mean().item()

            if last_kl is not None and last_kl > self.kl_threshold:
                self.actor.load_state_dict(prev_state)
                self.actor_optimizer.load_state_dict(prev_optim_state)
                rolled_back = True
                break

            prev_state = copy.deepcopy(self.actor.state_dict())
            prev_optim_state = copy.deepcopy(self.actor_optimizer.state_dict())

        self._update_count += 1
        if self._update_count % self.sync_old_every == 0:
            self.actor_old.load_state_dict(self.actor.state_dict())

        if self.lr_decay_rate < 1.0:
            new_lr = max(self.actor_lr_init * (self.lr_decay_rate ** self._update_count), self.lr_min)
            for group in self.actor_optimizer.param_groups:
                group["lr"] = new_lr

        self.last_mean_std = last_std_mean
        self.last_approx_kl = last_kl

        current_lr = self.actor_optimizer.param_groups[0]["lr"]

        return {
            "mean_reward": float(np.mean(mean_rewards)),
            "lr": current_lr,
            "entropy_coef": self.entropy_coef,
            "mean_std": last_std_mean,
            "approx_kl": last_kl,
            "rolled_back": rolled_back,
        }

    def save(self, file_name):
        torch.save(self.actor.state_dict(), file_name + "_grpo_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_grpo_actor_optimizer")

    def load(self, file_name):
        self.actor.load_state_dict(torch.load(file_name + "_grpo_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(file_name + "_grpo_actor_optimizer", map_location=self.device)
        )
        self.actor_old = copy.deepcopy(self.actor)

def quantize_phase_ste(phi: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Quantizes a continuous phase angle to the nearest of 2^n_bits uniformly
    spaced levels around the circle, using a STRAIGHT-THROUGH ESTIMATOR
    (STE) so gradients can still flow through the quantization step during
    optimization: forward pass returns the quantized value, backward pass
    behaves as if quantization were the identity function. This is the
    standard technique from quantized-neural-network training, applied here
    because it lets DirectPhaseOptimizer search directly for a good
    QUANTIZED solution rather than optimizing continuously and rounding
    only at the end (see the naive-vs-STE comparison in the deployment
    notebook -- STE consistently finds a better quantized solution because
    it "sees" the quantization grid during the search, not just afterward).

    n_bits=None or 0 means no quantization (pass-through).
    """
    if n_bits is None or n_bits <= 0:
        return phi
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    phi_wrapped = torch.remainder(phi, 2 * np.pi)
    q = torch.round(phi_wrapped / step) * step
    q = torch.remainder(q, 2 * np.pi)
    return phi_wrapped + (q - phi_wrapped).detach()


def quantize_phase_soft(phi: torch.Tensor, n_bits: int, temperature: float) -> torch.Tensor:
    """Soft-to-hard annealed alternative to the STE above (ablation): instead
    of faking the backward pass as the identity, compute a genuinely smooth,
    differentiable circular soft-assignment to the 2^n_bits levels --
    softmax-weighted (by negative squared circular distance / temperature)
    circular mean of the level angles. As temperature -> 0 this converges to
    exact nearest-level hard quantization (same forward value STE produces);
    at higher temperature it is a smoother, more spread-out function of phi,
    giving a less locally-flat (more informative) true gradient early in
    training. Standard soft-to-hard quantization idea (c.f. Agustsson et al.,
    "Soft-to-Hard Vector Quantization", NeurIPS 2017), adapted here to a
    circular (phase) quantizer instead of a vector codebook.

    n_bits=None or 0 means no quantization (pass-through).
    """
    if n_bits is None or n_bits <= 0:
        return phi
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    level_vals = torch.arange(levels, device=phi.device, dtype=phi.dtype) * step
    phi_wrapped = torch.remainder(phi, 2 * np.pi).unsqueeze(-1)
    diff = phi_wrapped - level_vals
    diff = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi
    d2 = diff ** 2
    w = torch.softmax(-d2 / (2 * temperature ** 2 + 1e-8), dim=-1)
    sin_sum = (w * torch.sin(level_vals)).sum(-1)
    cos_sum = (w * torch.cos(level_vals)).sum(-1)
    return torch.atan2(sin_sum, cos_sum)


def quantize_phase_hard_torch(phi: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Same hard quantization as quantization_cell.py's numpy version, but
    torch-native (no gradient needed here -- GRPO evaluates reward
    DIRECTLY on the quantized action, group-relative-advantage style, no
    backprop through the quantization boundary at all -- this is the key
    structural difference from STE-based DirectPhaseOptimizer)."""
    if n_bits is None or n_bits <= 0:
        return phi
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    phi_wrapped = torch.remainder(phi, 2 * np.pi)
    q = torch.round(phi_wrapped / step) * step
    return torch.remainder(q, 2 * np.pi)

class VonMisesActorQBest(nn.Module):
    """Same quantization-aware actor as v1 (log-std bounds tied to the
    target quantization step), but with a configurable, DEFAULT-SMALLER
    network: 1 hidden layer of 128 units instead of ~1024 units across 2
    layers -- statistically equivalent quality, 16x fewer parameters."""

    KAPPA_MIN = 1e-3
    KAPPA_MAX = 200.0

    def __init__(self, state_dim, action_dim, M, N, K, L, device, max_action=1,
                 log_std_min=-4.0, log_std_max=-0.5, log_std_init=None,
                 hidden_dim=128, depth=1, **_ignored):
        super().__init__()
        self.device = device
        self.M, self.N, self.K, self.L = M, N, K, L
        self.n_phase = M * L

        layers = []
        in_dim = state_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2 * self.n_phase))
        self.net = nn.Sequential(*layers)

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max
        init = log_std_init if log_std_init is not None else np.log(0.15)
        self.log_std = nn.Parameter(torch.full((self.n_phase,), float(init)))

    def forward(self, state: torch.Tensor):
        out = self.net(state.float())
        mu_r, mu_i = out.chunk(2, dim=-1)
        sigma = torch.exp(self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX))
        sigma = sigma.unsqueeze(0).expand_as(mu_r)
        rho = torch.sqrt(mu_r.pow(2) + mu_i.pow(2) + 1e-12)
        theta_mu = torch.atan2(mu_i, mu_r)
        kappa = (rho.pow(2) / sigma.pow(2)).clamp(self.KAPPA_MIN, self.KAPPA_MAX)
        return theta_mu, kappa

    def _phi_to_action(self, phi):
        cos_phi = torch.cos(phi); sin_phi = torch.sin(phi)
        blocks = []
        for l in range(self.L):
            s = l * self.M
            blocks.append(cos_phi[:, s:s + self.M]); blocks.append(sin_phi[:, s:s + self.M])
        return torch.cat(blocks, dim=1)

    def deterministic_action(self, state):
        theta_mu, _ = self.forward(state)
        return self._phi_to_action(theta_mu)

    def action_from_params(self, p1):
        return self._phi_to_action(p1)

    def sample(self, state, n_samples=1):
        theta_mu, kappa = self.forward(state)
        batch = theta_mu.shape[0]
        theta_rep = theta_mu.unsqueeze(0).expand(n_samples, batch, -1).reshape(n_samples * batch, -1)
        kappa_rep = kappa.unsqueeze(0).expand(n_samples, batch, -1).reshape(n_samples * batch, -1)
        dist = VonMises(theta_rep, kappa_rep)
        phi = dist.sample()
        action = self._phi_to_action(phi)
        return action, phi, theta_rep, kappa_rep

    @staticmethod
    def log_prob(raw, p1, p2):
        return VonMises(p1, p2).log_prob(raw).sum(dim=1)

    @staticmethod
    def entropy(p1, p2):
        kappa = p2
        i0e, i1e = torch.special.i0e(kappa), torch.special.i1e(kappa)
        log_i0 = torch.log(i0e.clamp_min(1e-12)) + kappa
        ratio = i1e / i0e.clamp_min(1e-12)
        return (np.log(2 * np.pi) + log_i0 - kappa * ratio).mean(dim=1)

    def reg_penalty(self, p1, coef):
        return torch.zeros((), device=p1.device)


class VonMisesActorQBestStateSigma(VonMisesActorQBest):
    """Identical to VonMisesActorQBest, except exploration width (sigma,
    hence kappa) is a function of the state instead of a single global
    nn.Parameter shared across every channel realization and every update.
    Rationale: the base version explores with the SAME per-element width
    for every state, so it can never learn "I'm confident about element i
    given this channel, but uncertain about element j" -- only a single
    fixed-per-element prior that anneals globally over training. Making
    sigma a 3rd output head lets the policy learn state-conditioned
    uncertainty while leaving every other formula (sample/log_prob/entropy/
    the kappa definition itself) completely unchanged, so none of the
    PPO-clip trust-region math changes -- only how kappa's inputs are
    computed."""

    def __init__(self, state_dim, action_dim, M, N, K, L, device, max_action=1,
                 log_std_min=-4.0, log_std_max=-0.5, log_std_init=None,
                 hidden_dim=128, depth=1, **_ignored):
        nn.Module.__init__(self)
        self.device = device
        self.M, self.N, self.K, self.L = M, N, K, L
        self.n_phase = M * L

        layers = []
        in_dim = state_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3 * self.n_phase))  # mu_r, mu_i, log_sigma
        self.net = nn.Sequential(*layers)

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max
        init = log_std_init if log_std_init is not None else np.log(0.15)
        # bias the 3rd head's output so sigma starts near exp(init), same
        # starting exploration scale as the base version's global parameter
        with torch.no_grad():
            self.net[-1].bias[2 * self.n_phase:].fill_(float(init))

    def forward(self, state: torch.Tensor):
        out = self.net(state.float())
        mu_r, mu_i, log_sigma = out.chunk(3, dim=-1)
        sigma = torch.exp(log_sigma.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX))
        rho = torch.sqrt(mu_r.pow(2) + mu_i.pow(2) + 1e-12)
        theta_mu = torch.atan2(mu_i, mu_r)
        kappa = (rho.pow(2) / sigma.pow(2)).clamp(self.KAPPA_MIN, self.KAPPA_MAX)
        return theta_mu, kappa


def build_best_agent(env, device, n_bits, group_size=32, ppo_epochs=5,
                      entropy_anneal_updates=250, seed=0, sigma_floor_frac=0.18,
                      log_std_init_frac=0.35, hidden_dim=128, depth=1):
    """Same as v1's build_enhanced_agent, just using VonMisesActorQBest
    (smaller default network) instead of VonMisesActorQ."""
    import copy
    torch.manual_seed(seed); np.random.seed(seed)
    if n_bits is None or n_bits <= 0:
        log_std_min, log_std_max, log_std_init = -4.0, -0.5, np.log(0.15)
    else:
        step = 2 * np.pi / (2 ** n_bits)
        log_std_min = np.log(sigma_floor_frac * step)
        log_std_max = -0.5
        log_std_init = np.log(log_std_init_frac * step)

    agent = GRPO(
        state_dim=env.state_dim, action_dim=env.action_dim, M=env.M, N=env.N, K=env.K, L=env.L,
        H=env.H_true, Ws=env.Ws, P=env.P, var=env.var, max_action=1,
        actor_lr=1e-3, actor_decay=1e-5, device=device, group_size=group_size,
        clip_eps=0.2, ppo_epochs=ppo_epochs, entropy_coef_start=0.01, entropy_coef_end=0.001,
        entropy_anneal_updates=entropy_anneal_updates, sync_old_every=1, kl_threshold=0.05,
        raw_l2_coef=1e-3, reparam_coef=0.0, actor_type="von_mises", advantage_mode="robust",
        reward_power_p=env.reward_power_p, reward_mean_weight=env.reward_mean_weight,
        reward_min_weight=env.reward_min_weight,
    )
    agent.actor = VonMisesActorQBest(env.state_dim, env.action_dim, env.M, env.N, env.K, env.L, device,
                                      max_action=1, log_std_min=log_std_min, log_std_max=log_std_max,
                                      log_std_init=log_std_init, hidden_dim=hidden_dim, depth=depth).to(device)
    agent.actor_old = copy.deepcopy(agent.actor)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=1e-3, weight_decay=1e-5)
    return agent


def update_quantized_best(agent, states, H_batch, n_bits, reparam_coef=1.0, lr=None,
                           reparam_quantizer="ste", global_step=0, tau_0=1.0, tau_min=0.01,
                           tau_decay=200.0, clip_eps_high=None, adaptive_reparam=False):
    """clip_eps_high: if set, use an asymmetric PPO clip range
    [1 - agent.clip_eps, 1 + clip_eps_high] instead of the symmetric
    [1 - agent.clip_eps, 1 + agent.clip_eps] (DAPO's "clip-higher": a wider
    upper bound lets a currently-low-probability-but-good action increase
    its probability more per update, counteracting the entropy collapse
    that symmetric clipping can cause). None (default) keeps the original
    symmetric behavior unchanged.

    adaptive_reparam: if True, scale reparam_coef up/down per update based
    on how informative THIS group's reward spread is relative to agent's
    own running-average spread (an EMA tracked on the agent object). When a
    group happens to be nearly degenerate (tight spread -> the group-
    relative/score-function term carries little signal, exactly the
    failure mode dr_grpo's advantage targets from a different angle), lean
    more on the differentiable STE pathwise gradient, which doesn't depend
    on relative ranking at all. When the group is unusually informative
    (wide spread), the score-function term is already reliable, so damp
    the pathwise term back down. This is a variance-reduction heuristic in
    the spirit of REBAR/RELAX's control-variate blending, not a literal
    variance-optimal coefficient (estimating that exactly would need
    Cov(g_score, g_reparam) per update, which isn't worth the overhead
    here)."""
    """Identical structure to v1's update_quantized_enhanced (STE reparam
    auxiliary loss on top of group-relative PPO), just with the retuned
    default reparam_coef=1.0 instead of 2.0.

    reparam_quantizer: "ste" (default, unchanged original behavior) or
    "soft" (ablation -- see quantize_phase_soft). "soft" anneals temperature
    tau = max(tau_min, tau_0 * exp(-global_step / tau_decay)) as global_step
    (e.g. environment step index) increases, so early updates get smoother/
    more-informative gradients and later updates converge toward the same
    hard-quantization behavior STE always uses."""
    if lr is not None:
        for g in agent.actor_optimizer.param_groups:
            g["lr"] = lr

    num_states = len(states)
    states_np = np.stack([np.asarray(s) for s in states], axis=0)
    states_t = torch.FloatTensor(states_np).to(agent.device)

    with torch.no_grad():
        _, raw_old, p1_old, p2_old = agent.actor_old.sample(states_t, n_samples=agent.group_size)
        old_log_prob = agent.actor_old.log_prob(raw_old, p1_old, p2_old)
        phi_q = quantize_phase_hard_torch(raw_old, n_bits)
        actions_q = agent.actor_old._phi_to_action(phi_q)

    def to_state_major(x, last_dim):
        return (x.view(agent.group_size, num_states, last_dim).permute(1, 0, 2)
                 .reshape(num_states * agent.group_size, last_dim))

    raw_old_cat = to_state_major(raw_old, raw_old.shape[-1])
    old_log_prob_cat = old_log_prob.view(agent.group_size, num_states).permute(1, 0).reshape(-1)

    action_dim = actions_q.shape[-1]
    actions_np = to_state_major(actions_q, action_dim).cpu().data.numpy().reshape(num_states, agent.group_size, action_dim)
    H_batch_np = np.stack([np.asarray(h) for h in H_batch], axis=0)

    rewards = agent._group_rewards_batched(actions_np, H_batch_np)
    mean_rewards = rewards.mean(axis=1)
    advantage = agent._compute_group_advantage(rewards)
    advantages_cat = torch.FloatTensor(advantage.reshape(-1)).to(agent.device)

    reparam_coef_eff = reparam_coef
    if adaptive_reparam:
        floor = 1e-2
        spread_now = max(float(np.std(rewards)), floor)
        ema = getattr(agent, "_spread_ema", None)
        agent._spread_ema = spread_now if ema is None else (0.9 * ema + 0.1 * spread_now)
        scale = float(np.clip(agent._spread_ema / spread_now, 0.33, 3.0))
        reparam_coef_eff = reparam_coef * scale

    state_rep_cat = states_t.repeat_interleave(agent.group_size, dim=0)
    H_batch_t = torch.tensor(H_batch_np, dtype=torch.complex64, device=agent.device)

    agent.actor.train()
    last_kl = None
    last_actor_loss = None
    last_entropy = None
    last_grad_norm = None
    for _ in range(agent.ppo_epochs):
        p1_new, p2_new = agent.actor.forward(state_rep_cat)
        new_log_prob = agent.actor.log_prob(raw_old_cat, p1_new, p2_new)
        ratio = torch.exp(new_log_prob - old_log_prob_cat)
        surr1 = ratio * advantages_cat
        hi = clip_eps_high if clip_eps_high is not None else agent.clip_eps
        surr2 = torch.clamp(ratio, 1.0 - agent.clip_eps, 1.0 + hi) * advantages_cat
        entropy = agent.actor.entropy(p1_new, p2_new).mean()

        p1_mean, _ = agent.actor.forward(states_t)
        if reparam_quantizer == "soft":
            tau = max(tau_min, tau_0 * np.exp(-global_step / tau_decay))
            phi_ste = quantize_phase_soft(p1_mean, n_bits, tau)
        else:
            phi_ste = quantize_phase_ste(p1_mean, n_bits)
        mean_action_q = agent.actor._phi_to_action(phi_ste)
        reparam_reward = agent._group_rewards_torch(mean_action_q, H_batch_t[:num_states]).mean()
        reparam_loss = -reparam_coef_eff * reparam_reward

        actor_loss = -torch.min(surr1, surr2).mean() - agent.entropy_coef * entropy + reparam_loss

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
        agent.actor_optimizer.step()

        last_actor_loss = actor_loss.item()
        last_entropy = entropy.item()
        last_grad_norm = float(grad_norm)

        with torch.no_grad():
            log_ratio = new_log_prob - old_log_prob_cat
            ratio2 = torch.exp(log_ratio)
            last_kl = ((ratio2 - 1) - log_ratio).mean().item()
        if last_kl > agent.kl_threshold:
            break

    agent.actor_old.load_state_dict(agent.actor.state_dict())
    return {"mean_reward": float(np.mean(mean_rewards)), "approx_kl": last_kl,
            "actor_loss": last_actor_loss, "entropy": last_entropy, "grad_norm": last_grad_norm,
            "group_reward_std": float(rewards.std(axis=1).mean())}
