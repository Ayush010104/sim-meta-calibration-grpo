"""Extends DynamicRicianEnvCorrected with GENUINE user mobility: each user
gets a fixed-speed random-walk-direction velocity, and at every
advance_channel() call, users actually move (v * T_s) before the LOS
steering vector / path-loss are recomputed from the new positions -- on
top of the existing NLOS AR(1) scattering evolution (temporal_rho), giving
the combined effect of real geometric Doppler-like phase change PLUS
ambient scattering drift, rather than either one in isolation.

T_s is the real-world seconds represented by one "channel step" -- decoupled
from how long our GRPO code actually takes to run on this CPU, so we can
test "if the control loop were fast enough to react every T_s seconds,
would this still help a user moving at v m/s" independent of today's
actual compute speed.
"""
import numpy as np

from env_ris import DynamicRicianEnvCorrected


class MovingUserEnv(DynamicRicianEnvCorrected):
    def __init__(self, *args, user_speed=1.4, T_s=0.01, direction_seed=0, **kwargs):
        self.user_speed = user_speed  # m/s
        self.T_s = T_s                # seconds per channel step
        self._direction_seed = direction_seed
        super().__init__(*args, **kwargs)
        rng = np.random.default_rng(direction_seed)
        self._center = (self._dBS, 0.0, 0.0)
        self.user_vel = []
        for k in range(self.K):
            theta = rng.uniform(0, 2 * np.pi)
            self.user_vel.append((user_speed * np.cos(theta), user_speed * np.sin(theta)))

    def _move_users(self):
        cx, cy, cz = self._center
        for k in range(self.K):
            x, y, z = self.user_pos[k]
            vy, vz = self.user_vel[k]
            y_new, z_new = y + vy * self.T_s, z + vz * self.T_s
            # bounce off the disc boundary so users stay in the same region
            if (y_new - cy) ** 2 + (z_new - cz) ** 2 > self._radius ** 2:
                vy, vz = -vy, -vz
                self.user_vel[k] = (vy, vz)
                y_new, z_new = y + vy * self.T_s, z + vz * self.T_s
            self.user_pos[k] = (x, y_new, z_new)
        # user_pos is the SAME list object the channel generator holds
        # (passed by reference at construction) -- recompute the
        # deterministic LOS/path-loss terms from the new positions.
        self.channel_gen._build_los_and_pathloss()

    def advance_channel(self):
        self._move_users()
        return super().advance_channel()


def channel_similarity(H_a, H_b):
    num = np.abs(np.vdot(H_a.flatten(), H_b.flatten()))
    den = np.linalg.norm(H_a) * np.linalg.norm(H_b) + 1e-30
    return num / den


if __name__ == "__main__":
    from scipy.special import j0

    BASE_CFG = dict(num_antennas=5, num_RIS_elements=36, num_users=5, num_layers=4,
                     dBS=30.0, radius=50.0)
    fc = 28e9
    c = 3e8

    print(f"{'v (m/s)':>10}{'T_s (s)':>10}{'f_d (Hz)':>10}{'lag=1 sim (measured)':>22}"
          f"{'lag=1 theory J0':>18}{'lag=20 sim':>12}{'lag=20 theory':>15}")
    for v, T_s in [(0.01, 0.01), (1.4, 0.001), (1.4, 0.01), (1.4, 0.1), (0.1, 1.0)]:
        env = MovingUserEnv(seed=12345, temporal_rho=0.999, user_speed=v, T_s=T_s,
                             direction_seed=0, **BASE_CFG)
        H0 = env.H_true.copy()
        H_seq = [H0]
        for _ in range(20):
            env.advance_channel()
            H_seq.append(env.H_true.copy())
        sim1 = channel_similarity(H0, H_seq[1])
        sim20 = channel_similarity(H0, H_seq[20])
        fd = v * fc / c
        theory1 = j0(2 * np.pi * fd * T_s)
        theory20 = j0(2 * np.pi * fd * (20 * T_s))
        print(f"{v:>10}{T_s:>10}{fd:>10.2f}{sim1:>22.4f}{theory1:>18.4f}{sim20:>12.4f}{theory20:>15.4f}")
