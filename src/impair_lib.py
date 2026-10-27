"""Generic SIM impairment library (any K=N, M (square), L). Physics identical to env_ris (geometry verified against env._build_Ws).
Modeled-class impairments (the calibration latent can represent them):
   psi   per-element phase-tuning error (von Mises)          amp   per-layer state amplitudes (strong=1, weak=a)
   delta per-layer spacing error (fraction)                  off   per-layer lateral offset (element pitch)
Unmodeled impairments (outside the calibration latent):
   amp_elem  per-element multiplicative amplitude spread     stuck  elements stuck at +1 / -1 (ignore the command)
System impairments:  CSI error (NMSE dB),  multiplicative noise on measured rate."""
import numpy as np, torch
from big_sys import make_env, channels, sgn, relax
from env_ris import add_csi_estimation_error
PI = float(np.pi)

LEVELS = {
    "mild":     dict(psi=0.15, weak=0.85, spacing=0.01, offset=0.05, csi_db=-15, noise=0.02, amp_elem=0.0, stuck=0.0),
    "moderate": dict(psi=0.32, weak=0.70, spacing=0.03, offset=0.10, csi_db=-10, noise=0.05, amp_elem=0.0, stuck=0.0),
    "severe":   dict(psi=0.55, weak=0.50, spacing=0.05, offset=0.20, csi_db=-5,  noise=0.10, amp_elem=0.0, stuck=0.0),
    "moderate+unmodeled": dict(psi=0.32, weak=0.70, spacing=0.03, offset=0.10, csi_db=-10, noise=0.05, amp_elem=0.10, stuck=0.03),
}
NONE = dict(psi=0.0, weak=1.0, spacing=0.0, offset=0.0, csi_db=None, noise=0.0, amp_elem=0.0, stuck=0.0)


class Geometry:
    def __init__(self, env):
        self.M, self.L, self.N = env.M, env.L, env.N
        self.lam, self.r_et, self.dx, self.dy = env.lamda, env.r_et, env.dx, env.dy
        self.d_layer, self.T_sim = env.d_layer, env.T_sim
        di = np.sqrt(self.M); m = np.arange(self.M)
        self.yc = torch.tensor((di - 1) / 2 - (m % di), dtype=torch.float64)
        self.zc = torch.tensor((di - 1) / 2 - (m / di), dtype=torch.float64)
        self.xi = torch.tensor(m / di, dtype=torch.float64)
        self.yi = torch.tensor(m % di, dtype=torch.float64)
        self.az = torch.tensor((self.N - 1) / 2 - np.arange(self.N), dtype=torch.float64)

    def kernel(self, d, pref):
        return (pref / d ** 2) * ((1 / (2 * PI * d)) - (1j / self.lam)) * torch.exp(1j * (2 * PI * d) / self.lam)

    def Ws(self, delta, off):
        delta, off = delta.double(), off.double()
        x0 = self.d_layer * (1 + delta[0]) / self.r_et
        dy = (self.yc + off[0, 0]).unsqueeze(1)
        dz = (self.zc + off[0, 1]).unsqueeze(1) - self.az.unsqueeze(0)
        out = [self.kernel(torch.sqrt(x0 ** 2 + dy ** 2 + dz ** 2) * self.r_et, self.dx * self.dy * self.T_sim / self.L).to(torch.complex64)]
        for l in range(1, self.L):
            dl = self.d_layer * (1 + delta[l])
            px = (self.xi.unsqueeze(1) + off[l, 0]) - self.xi.unsqueeze(0)
            py = (self.yi.unsqueeze(1) + off[l, 1]) - self.yi.unsqueeze(0)
            dij = torch.sqrt((torch.sqrt(px ** 2 + py ** 2) * self.r_et) ** 2 + dl ** 2)
            out.append(self.kernel(dij, self.dx * self.dy * dl).to(torch.complex64))
        return out


class Dev:
    def __init__(self, sysm, th, amp_elem=None, stuck=None):
        self.s = sysm
        self.K, self.M, self.L = sysm.K, sysm.M, sysm.L
        self.Ws = sysm.geo.Ws(th["d"], th["o"])
        self.psi, self.ap, self.an = th["p"], th["a"][:, 0].clamp(0.05, 1.5), th["a"][:, 1].clamp(0.05, 1.5)
        self.amp_elem, self.stuck = amp_elem, stuck

    def rates(self, x, H):
        B, M, L, K = x.shape[0], self.M, self.L, self.K
        if self.stuck is not None:
            st = self.stuck.reshape(1, L * M)
            x = torch.where(st != 0, st.expand(B, -1), x)
        d = []
        for l in range(L):
            xl = x[:, l * M:(l + 1) * M]
            amp = self.ap[l] * (1 + xl) / 2 + self.an[l] * (1 - xl) / 2
            if self.amp_elem is not None:
                amp = amp * self.amp_elem[l].unsqueeze(0)
            d.append(torch.polar(torch.ones(B, M), self.psi[l].unsqueeze(0).expand(B, M)) * (amp * xl))
        Gv = torch.eye(M, dtype=torch.complex64).unsqueeze(0).expand(B, M, M)
        for idx in range(L - 1, 0, -1):
            Gv = torch.matmul(Gv, d[idx].unsqueeze(-1) * self.Ws[idx].unsqueeze(0))
        Gv = Gv * d[0].unsqueeze(1)
        Full = torch.matmul(torch.matmul(H.conj().transpose(1, 2), Gv), self.Ws[0])
        power = Full.abs() ** 2 * self.s.P.view(1, 1, K)
        sig = torch.diagonal(power, dim1=-2, dim2=-1)
        return torch.log2(1 + sig / (power.sum(-1) - sig + self.s.var))


def minrate(dev, x, H):
    with torch.no_grad():
        return dev.rates(x, H).min(-1).values


class System:
    def __init__(self, K, M, L):
        self.K, self.M, self.L = K, M, L
        self.env = make_env(K, M, L, 0)
        self.geo = Geometry(self.env)
        self.P = torch.tensor(self.env.P, dtype=torch.float32); self.var = float(self.env.var)

    def nominal(self):
        L, M = self.L, self.M
        return dict(p=torch.zeros(L, M), a=torch.ones(L, 2), d=torch.zeros(L), o=torch.zeros(L, 2))

    def check(self):
        ref = [torch.tensor(w, dtype=torch.complex64) for w in self.env.Ws]
        n = self.nominal()
        return max((a - b).abs().max().item() / b.abs().max().item() for a, b in zip(self.geo.Ws(n["d"], n["o"]), ref))

    def sample(self, lv, seed):
        """true device theta + unmodeled extras for impairment level dict lv"""
        L, M = self.L, self.M
        r = np.random.default_rng(seed)
        psi = r.vonmises(0.0, 1.0 / lv["psi"] ** 2, size=(L, M)) if lv["psi"] > 0 else np.zeros((L, M))
        th = dict(p=torch.tensor(psi, dtype=torch.float32),
                  a=torch.stack([torch.ones(L), torch.full((L,), float(lv["weak"]))], 1),
                  d=torch.tensor(r.normal(0, lv["spacing"], L) if lv["spacing"] > 0 else np.zeros(L), dtype=torch.float32),
                  o=torch.tensor(r.normal(0, lv["offset"], (L, 2)) if lv["offset"] > 0 else np.zeros((L, 2)), dtype=torch.float32))
        ae = torch.tensor(np.clip(1 + r.normal(0, lv["amp_elem"], (L, M)), 0.3, 1.7), dtype=torch.float32) if lv["amp_elem"] > 0 else None
        st = None
        if lv["stuck"] > 0:
            mask = r.random((L, M)) < lv["stuck"]
            st = torch.tensor(np.where(mask, r.choice([-1.0, 1.0], (L, M)), 0.0), dtype=torch.float32)
        return th, ae, st

    def channels(self, seeds, n):
        return channels(self.K, self.M, self.L, seeds, n_steps=100, stride=10)[:n]

    @staticmethod
    def estimate(Hs, csi_db, seed):
        if csi_db is None:
            return Hs.clone()
        eps = float(np.sqrt(10 ** (csi_db / 10)))
        r = np.random.default_rng(seed)
        return torch.tensor(np.stack([add_csi_estimation_error(h.numpy().astype(complex), eps, r) for h in Hs]), dtype=torch.complex64)


def design_rate(model, true, H, Hh, steps=200, seed=1):
    D = true.L * true.M
    w0 = torch.randn(H.shape[0], D, generator=torch.Generator().manual_seed(seed)) * 0.5
    return minrate(true, sgn(relax(model, Hh, w0, steps)), H)
