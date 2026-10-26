"""SIM/RIS channel model and environment, extracted verbatim (physics unchanged)
from GRPO_1bit_first_final (3).ipynb so the MINLP formulation optimizes against
the exact same objective/channel as the notebook's GRPO / Direct-STE baselines.
"""
import numpy as np
from scipy.linalg import sqrtm


def db_to_linear(db):
    return 10 ** (db / 10)


RATE_FLOOR = 1e-3


def power_mean(rates, p, axis=-1, eps=1e-8):
    rates = np.asarray(rates, dtype=float)
    if abs(p) < 1e-9:
        return np.exp(np.mean(np.log(rates + eps), axis=axis))
    return np.mean(rates ** p, axis=axis) ** (1.0 / p)


class CorrelatedRicianChannel:
    def __init__(self, M, K, user_pos, last_RIS_pos, r_et, lamda, Co, alpha,
                 rician_K_factor=3.0, temporal_rho=0.98, seed=0):
        self.M = M
        self.K = K
        self.user_pos = user_pos
        self.last_RIS_pos = last_RIS_pos
        self.r_et = r_et
        self.lamda = lamda
        self.Co = Co
        self.alpha = alpha
        self.rician_K_factor = rician_K_factor
        self.temporal_rho = temporal_rho

        self.rng = np.random.default_rng(seed)

        self._build_geometry()
        self._build_R_RIS()
        self._build_los_and_pathloss()
        self.reset()

    def _cal_dis(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _build_geometry(self):
        dim = int(round(np.sqrt(self.M)))
        assert dim * dim == self.M, "M must be a perfect square for the planar RIS grid"
        self.ris_coords = []
        for m in range(self.M):
            z_idx = m / dim
            y_idx = m % dim
            z_idx = (dim - 1) / 2 - z_idx
            y_idx = (dim - 1) / 2 - y_idx
            self.ris_coords.append((y_idx * self.r_et, z_idx * self.r_et))
        self.ris_coords = np.array(self.ris_coords)

    def _build_R_RIS(self):
        dim = int(round(np.sqrt(self.M)))
        R = np.zeros((self.M, self.M))
        for i in range(self.M):
            for j in range(self.M):
                xi, yi = i / dim, i % dim
                xj, yj = j / dim, j % dim
                dij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) * self.r_et
                R[i, j] = np.sinc((2 * dij) / self.lamda)
        self.R_RIS_sqrt = sqrtm(R).real

    def _build_los_and_pathloss(self):
        self.a_los = np.zeros((self.M, self.K), dtype=complex)
        self.b_k = np.zeros(self.K)

        for k in range(self.K):
            d = self._cal_dis(self.user_pos[k], self.last_RIS_pos)
            self.b_k[k] = self.Co * (1.0 / d) ** self.alpha

            dx = self.user_pos[k][1] - self.last_RIS_pos[1]
            dy = self.user_pos[k][2] - self.last_RIS_pos[2]
            horiz = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
            sin_theta = dx / horiz
            cos_theta = dy / horiz

            phase = (2 * np.pi / self.lamda) * (
                self.ris_coords[:, 0] * sin_theta + self.ris_coords[:, 1] * cos_theta
            )
            self.a_los[:, k] = np.exp(1j * phase)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.g_state = (self.rng.standard_normal((self.M, self.K)) +
                         1j * self.rng.standard_normal((self.M, self.K))) / np.sqrt(2)
        return self._compose()

    def _compose(self):
        Kf = self.rician_K_factor
        los_w = np.sqrt(Kf / (Kf + 1.0))
        nlos_w = np.sqrt(1.0 / (Kf + 1.0))
        sqrt_b = np.sqrt(self.b_k)[None, :]
        nlos_component = self.R_RIS_sqrt @ self.g_state
        H = los_w * sqrt_b * self.a_los + nlos_w * sqrt_b * nlos_component
        return H

    def step(self):
        rho = self.temporal_rho
        innovation = (self.rng.standard_normal((self.M, self.K)) +
                      1j * self.rng.standard_normal((self.M, self.K))) / np.sqrt(2)
        self.g_state = rho * self.g_state + np.sqrt(max(0.0, 1 - rho ** 2)) * innovation
        return self._compose()


def add_csi_estimation_error(H_true, relative_error_std, rng):
    if relative_error_std <= 0:
        return H_true.copy()
    per_user_norm = np.linalg.norm(H_true, axis=0, keepdims=True)
    M = H_true.shape[0]
    noise = (rng.standard_normal(H_true.shape) + 1j * rng.standard_normal(H_true.shape)) / np.sqrt(2)
    noise = noise * (relative_error_std * per_user_norm / np.sqrt(M))
    return H_true + noise


class DynamicRicianEnv(object):
    def __init__(self, num_antennas, num_RIS_elements, num_users, num_layers, seed,
                 rician_K_factor=3.0, temporal_rho=0.98, csi_relative_error_std=0.0,
                 reward_power_p=0.0, reward_mean_weight=0.4, reward_min_weight=0.5):
        self.N = num_antennas
        self.M = num_RIS_elements
        self.K = num_users
        self.L = num_layers

        self.carrier_f = 28 * 1e9
        self.lamda = 3e8 / self.carrier_f
        self.dx = self.lamda / 2
        self.dy = self.lamda / 2
        self.T_sim = 5 * self.lamda
        self.d_layer = self.T_sim / self.L
        self.r_et = self.lamda / 2
        self.P_max = db_to_linear(20) * 1e-3
        self.Co = db_to_linear(-30)
        self.alpha = 3.5
        self.var = db_to_linear(-90) * 1e-3

        assert self.N == self.K

        self.PP = np.ones(self.K) / self.K
        self.P = self.PP * self.P_max

        self._get_positions(seed)

        self.channel_gen = CorrelatedRicianChannel(
            M=self.M, K=self.K, user_pos=self.user_pos, last_RIS_pos=self.last_RIS_pos,
            r_et=self.r_et, lamda=self.lamda, Co=self.Co, alpha=self.alpha,
            rician_K_factor=rician_K_factor, temporal_rho=temporal_rho, seed=seed,
        )
        self.csi_relative_error_std = csi_relative_error_std
        self._csi_rng = np.random.default_rng(seed + 999)

        self._init_SIM(seed)
        self.Ws = self._build_Ws()

        self.H_true = self.channel_gen.reset(seed=seed)
        self.H_obs = add_csi_estimation_error(self.H_true, self.csi_relative_error_std, self._csi_rng)
        self.Gv = self._get_Gv()

        self.action_dim = 2 * self.L * self.M
        self.state_dim = 2 * self.M * self.K + 2 * self.L * self.M

        self.reward_power_p = reward_power_p
        self.reward_mean_weight = reward_mean_weight
        self.reward_min_weight = reward_min_weight

    def _cal_dis(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _get_positions(self, seed):
        rng = np.random.default_rng(seed)
        self.user_pos = []
        centre = (10, 0, 0)
        x_c, y_c, z_c = centre
        radius = 10
        for _ in range(self.K):
            theta = 2 * np.pi * rng.random()
            r = radius * np.sqrt(rng.random())
            y = y_c + r * np.cos(theta)
            z = z_c + r * np.sin(theta)
            self.user_pos.append((x_c, y, z))
        self.last_RIS_pos = (5 * self.lamda, 0, 0)

    def _init_SIM(self, seed):
        rng = np.random.default_rng(seed + 1)
        self.SIM = []
        self.d_angles = []
        for l in range(self.L):
            angles = rng.uniform(0, 2 * np.pi, size=self.M)
            phase_shifts = np.exp(1j * angles)
            self.d_angles.extend(phase_shifts)
            self.SIM.append(np.diag(phase_shifts))

    def _build_Ws(self):
        Ws = []
        RIS_cor = []
        di_r = np.sqrt(self.M)
        for m in range(self.M):
            z_cor = (m / di_r)
            y_cor = (m % di_r)
            z_cor = (di_r - 1) / 2 - z_cor
            y_cor = (di_r - 1) / 2 - y_cor
            RIS_cor.append((self.d_layer, y_cor, z_cor))

        ant_cor = [(0, 0, (self.N - 1) / 2 - n) for n in range(self.N)]

        mat = np.zeros((self.M, self.N)) + 1j * np.zeros((self.M, self.N))
        for i in range(self.M):
            for j in range(self.N):
                dmn = self._cal_dis(RIS_cor[i], ant_cor[j]) * self.r_et
                dt = self.T_sim / self.L
                mat[i][j] = ((self.dx * self.dy * dt) / (dmn ** 2)) * ((1 / (2 * np.pi * dmn)) - (1j / self.lamda)) * np.exp(1j * (2 * np.pi * dmn) / self.lamda)
        Ws.append(mat)

        for l in range(1, self.L):
            mat = np.zeros((self.M, self.M)) + 1j * np.zeros((self.M, self.M))
            for i in range(self.M):
                for j in range(self.M):
                    xi, yi = i / di_r, i % di_r
                    xj, yj = j / di_r, j % di_r
                    dij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) * self.r_et
                    dij = np.sqrt(dij ** 2 + self.d_layer ** 2)
                    mat[i][j] = ((self.dx * self.dy * self.d_layer) / (dij ** 2)) * ((1 / (2 * np.pi * dij)) - (1j / self.lamda)) * np.exp(1j * (2 * np.pi * dij) / self.lamda)
            Ws.append(mat)
        return Ws

    def _get_Gv(self):
        Gv = np.eye(self.M)
        for k in range(1, self.L):
            Gv = np.matmul(Gv, np.matmul(self.SIM[self.L + 1 - k - 1], self.Ws[self.L + 1 - k - 1]))
        Gv = np.matmul(Gv, self.SIM[0])
        return Gv

    def _build_state(self):
        phase = np.array(self.d_angles)
        parts = [self.H_obs, phase]
        return np.hstack([
            np.hstack([np.real(h).flatten(), np.imag(h).flatten()]) if np.iscomplexobj(h) else h.flatten()
            for h in parts
        ])

    def advance_channel(self):
        self.H_true = self.channel_gen.step()
        self.H_obs = add_csi_estimation_error(self.H_true, self.csi_relative_error_std, self._csi_rng)
        self.Gv = self._get_Gv()
        return self._build_state()

    def reset_state_only(self):
        return self._build_state()

    def calculate_reward(self):
        rates = []
        H = self.H_true
        hk_all = [H[:, k:k + 1] for k in range(self.K)]
        W1 = self.Ws[0]
        for k in range(self.K):
            hk = hk_all[k]
            wk = W1[:, k:k + 1]
            num = np.matmul(hk.conj().T, np.matmul(self.Gv, wk))
            num_val = np.abs(num[0][0]) ** 2 * self.P[k]
            den_val = 0.0
            for o in range(self.K):
                if o == k:
                    continue
                wo = W1[:, o:o + 1]
                den = np.matmul(hk.conj().T, np.matmul(self.Gv, wo))
                den_val += np.abs(den[0][0]) ** 2 * self.P[o]
            den_val += self.var
            rates.append(np.log2(1 + num_val / den_val))
        rates_arr = np.array(rates, dtype=float)
        min_rate = float(np.min(rates_arr))
        safe_rates = np.clip(rates_arr, RATE_FLOOR, None)
        mean_term = float(power_mean(safe_rates, self.reward_power_p))
        reward = (self.reward_mean_weight * mean_term + self.reward_min_weight * min_rate) * 10
        return reward, min_rate, rates_arr

    def step(self, action):
        action = np.asarray(action).reshape(1, -1)
        self.SIM = []
        self.d_angles = []
        for l in range(self.L):
            start = l * 2 * self.M
            Phi_real_l = action[:, start: start + self.M].reshape(-1)
            Phi_imag_l = action[:, start + self.M: start + 2 * self.M].reshape(-1)
            phase_l = Phi_real_l + 1j * Phi_imag_l
            self.SIM.append(np.diag(phase_l))
            self.d_angles.extend(phase_l)
        self.Gv = self._get_Gv()
        reward, minr, rates_arr = self.calculate_reward()
        new_state = self._build_state()
        return new_state, reward, minr, False, {"rates": rates_arr}


class DynamicRicianEnvFixedGeometry(DynamicRicianEnv):
    def _build_Ws(self):
        Ws = []
        RIS_cor = []
        di_r = np.sqrt(self.M)
        d_layer_index_units = self.d_layer / self.r_et
        for m in range(self.M):
            z_cor = (m / di_r)
            y_cor = (m % di_r)
            z_cor = (di_r - 1) / 2 - z_cor
            y_cor = (di_r - 1) / 2 - y_cor
            RIS_cor.append((d_layer_index_units, y_cor, z_cor))

        ant_cor = [(0, 0, (self.N - 1) / 2 - n) for n in range(self.N)]

        mat = np.zeros((self.M, self.N)) + 1j * np.zeros((self.M, self.N))
        for i in range(self.M):
            for j in range(self.N):
                dmn = self._cal_dis(RIS_cor[i], ant_cor[j]) * self.r_et
                dt = self.T_sim / self.L
                mat[i][j] = ((self.dx * self.dy * dt) / (dmn ** 2)) * ((1 / (2 * np.pi * dmn)) - (1j / self.lamda)) * np.exp(1j * (2 * np.pi * dmn) / self.lamda)
        Ws.append(mat)

        for l in range(1, self.L):
            mat = np.zeros((self.M, self.M)) + 1j * np.zeros((self.M, self.M))
            for i in range(self.M):
                for j in range(self.M):
                    xi, yi = i / di_r, i % di_r
                    xj, yj = j / di_r, j % di_r
                    dij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) * self.r_et
                    dij = np.sqrt(dij ** 2 + self.d_layer ** 2)
                    mat[i][j] = ((self.dx * self.dy * self.d_layer) / (dij ** 2)) * ((1 / (2 * np.pi * dij)) - (1j / self.lamda)) * np.exp(1j * (2 * np.pi * dij) / self.lamda)
            Ws.append(mat)
        return Ws


class DynamicRicianEnvCorrected(DynamicRicianEnvFixedGeometry):
    def __init__(self, *args, dBS=20.0, radius=50.0, **kwargs):
        self._dBS = dBS
        self._radius = radius
        super().__init__(*args, **kwargs)

    def _get_positions(self, seed):
        rng = np.random.default_rng(seed)
        self.user_pos = []
        x_c, y_c, z_c = self._dBS, 0, 0
        for _ in range(self.K):
            theta = 2 * np.pi * rng.random()
            r = self._radius * np.sqrt(rng.random())
            y = y_c + r * np.cos(theta)
            z = z_c + r * np.sin(theta)
            self.user_pos.append((x_c, y, z))
        self.last_RIS_pos = (5 * self.lamda, 0, 0)
