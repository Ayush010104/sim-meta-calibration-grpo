"""Calibration GRPO with ALL defensible non-idealities ON at once (K=10, M=64, L=6, 1-bit, commanded phases exactly {0,pi}).
True device / system (unknown to the designer):
  (1) per-element phase-tuning error   psi ~ von Mises(0, kappa=11) (~0.32 rad)       [SIM phase-tuning error model, arXiv 2502.11182]
  (2) state-dependent amplitude         practical phase-dependent amplitude curve (bmin=.2, phi0=.43pi, kap=1.6),
                                        operating point -> amplitudes 1.00 / 0.70       [practical RIS amplitude model]
  (3) inter-layer geometric deviation   per-layer spacing error delta_l ~ N(0, 3%) and lateral layer offset
                                        (dy, dz)_l ~ N(0, 0.1 element pitch)            [SIM interlayer error calibration, arXiv 2603.08040]
  (4) imperfect CSI                     designer sees H_hat = H + E, NMSE = -10 dB (relative std 0.316)
  (5) noisy reward measurement          measured min-rate x (1 + N(0, 0.05^2))
  (6) 1-bit phase quantisation          (inherent)
MAGNITUDES of (1),(3),(4),(5) are assumptions, not values taken from a paper.
Learned latent: psi_hat (L x M), per-layer amplitudes (L x 2), per-layer spacing (L), per-layer lateral offset (L x 2).
Usage: python3 calibration_full.py <updates> <G> <tag>"""
import sys, time, numpy as np, torch
torch.set_num_threads(3)
from big_sys import make_env, channels, sgn, relax, softmin
from env_ris import add_csi_estimation_error
K, M, L = 10, 64, 6
D = L * M
PI = float(np.pi)


def beta(phi, bmin=0.2, phi0=0.43 * PI, kap=1.6):
    return (1 - bmin) * ((np.sin(phi - phi0) + 1) / 2) ** kap + bmin


_phi1 = 0.43 * PI + 0.2
_b = np.array([beta(_phi1), beta(_phi1 + PI)])
A0, API = float(_b[0] / _b.max()), float(_b[1] / _b.max())


class Geometry:
    """Vectorised re-implementation of env._build_Ws with per-layer spacing errors and lateral offsets (units: element pitch r_et)."""
    def __init__(self, env):
        self.lam, self.r_et, self.dx, self.dy = env.lamda, env.r_et, env.dx, env.dy
        self.d_layer, self.T_sim, self.N = env.d_layer, env.T_sim, env.N
        di = np.sqrt(M)
        m = np.arange(M)
        self.yc = torch.tensor((di - 1) / 2 - (m % di), dtype=torch.float64)
        self.zc = torch.tensor((di - 1) / 2 - (m / di), dtype=torch.float64)
        self.xi = torch.tensor(m / di, dtype=torch.float64)
        self.yi = torch.tensor(m % di, dtype=torch.float64)
        self.az = torch.tensor((self.N - 1) / 2 - np.arange(self.N), dtype=torch.float64)

    def kernel(self, d, pref):
        return (pref / d ** 2) * ((1 / (2 * PI * d)) - (1j / self.lam)) * torch.exp(1j * (2 * PI * d) / self.lam)

    def Ws(self, delta, off):                     # delta (L,), off (L,2) torch float
        delta, off = delta.double(), off.double()
        out = []
        x0 = self.d_layer * (1 + delta[0]) / self.r_et
        dy = (self.yc + off[0, 0]).unsqueeze(1) - 0.0
        dz = (self.zc + off[0, 1]).unsqueeze(1) - self.az.unsqueeze(0)
        dmn = torch.sqrt(x0 ** 2 + dy ** 2 + dz ** 2) * self.r_et
        out.append(self.kernel(dmn, self.dx * self.dy * self.T_sim / L).to(torch.complex64))
        for l in range(1, L):
            dl = self.d_layer * (1 + delta[l])
            px = (self.xi.unsqueeze(1) + off[l, 0]) - self.xi.unsqueeze(0)
            py = (self.yi.unsqueeze(1) + off[l, 1]) - self.yi.unsqueeze(0)
            dij = torch.sqrt((torch.sqrt(px ** 2 + py ** 2) * self.r_et) ** 2 + dl ** 2)
            out.append(self.kernel(dij, self.dx * self.dy * dl).to(torch.complex64))
        return out


class Dev:
    def __init__(self, Ws, psi, ap, an, P, var):
        self.Ws, self.psi, self.ap, self.an, self.P, self.var = Ws, psi, ap, an, P, var

    def rates(self, x, H):
        B = x.shape[0]
        d = []
        for l in range(L):
            xl = x[:, l * M:(l + 1) * M]
            amp = self.ap[l] * (1 + xl) / 2 + self.an[l] * (1 - xl) / 2
            d.append(torch.polar(torch.ones(B, M), self.psi[l].unsqueeze(0).expand(B, M)) * (amp * xl))
        Gv = torch.eye(M, dtype=torch.complex64).unsqueeze(0).expand(B, M, M)
        for idx in range(L - 1, 0, -1):
            Gv = torch.matmul(Gv, d[idx].unsqueeze(-1) * self.Ws[idx].unsqueeze(0))
        Gv = Gv * d[0].unsqueeze(1)
        Full = torch.matmul(torch.matmul(H.conj().transpose(1, 2), Gv), self.Ws[0])
        power = Full.abs() ** 2 * self.P.view(1, 1, K)
        sig = torch.diagonal(power, dim1=-2, dim2=-1)
        return torch.log2(1 + sig / (power.sum(-1) - sig + self.var))


def minrate(dev, x, H):
    with torch.no_grad():
        return dev.rates(x, H).min(-1).values


def polish(dev, x, H, sweeps=10):
    x = x.clone(); best = minrate(dev, x, H)
    for _ in range(sweeps):
        ch = False
        for j in torch.randperm(D).tolist():
            x2 = x.clone(); x2[:, j] = -x2[:, j]
            r2 = minrate(dev, x2, H); up = r2 > best + 1e-9
            if up.any():
                x[up] = x2[up]; best = torch.where(up, r2, best); ch = True
        if not ch: break
    return best


def main():
    U, G, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    env = make_env(K, M, L, 0)
    geo = Geometry(env)
    P = torch.tensor(env.P, dtype=torch.float32); var = float(env.var)
    z = torch.zeros(L); z2 = torch.zeros(L, 2); one = torch.ones(L)
    ref = [torch.tensor(w, dtype=torch.complex64) for w in env.Ws]
    print("geometry check vs env._build_Ws:", max((a - b).abs().max().item() / b.abs().max().item() for a, b in zip(geo.Ws(z, z2), ref)), flush=True)

    rng = np.random.default_rng(7)
    psi_t = torch.tensor(rng.vonmises(0.0, 11.0, size=(L, M)), dtype=torch.float32)
    dl_t = torch.tensor(rng.normal(0, 0.03, L), dtype=torch.float32)
    of_t = torch.tensor(rng.normal(0, 0.1, (L, 2)), dtype=torch.float32)
    true = Dev(geo.Ws(dl_t, of_t), psi_t, A0 * one, API * one, P, var)
    nominal = Dev(geo.Ws(z, z2), torch.zeros(L, M), one, one, P, var)
    ampaware = Dev(geo.Ws(z, z2), torch.zeros(L, M), A0 * one, API * one, P, var)
    print(f"true device: psi std {psi_t.std():.3f} rad | amps {A0:.2f}/{API:.2f} | spacing err % {np.round(100*dl_t.numpy(),1).tolist()} | offsets (pitch) {np.round(of_t.numpy(),2).tolist()}", flush=True)

    def est(Hs, seed):
        r = np.random.default_rng(seed)
        return torch.tensor(np.stack([add_csi_estimation_error(h.numpy().astype(complex), 0.316, r) for h in Hs]), dtype=torch.complex64)
    pool = channels(K, M, L, list(range(1000, 1020)), n_steps=100, stride=10); pool_h = est(pool, 11)
    dev = channels(K, M, L, [4000, 4001], n_steps=100, stride=10)[:10]; dev_h = est(dev, 12)
    test = channels(K, M, L, [5000, 5001, 5002, 5003], n_steps=100, stride=10)[:40]; test_h = est(test, 13)

    def deploy(model, H, Hh, steps=50, seed=1):
        w0 = torch.randn(H.shape[0], D, generator=torch.Generator().manual_seed(seed)) * 0.5
        x = sgn(relax(model, Hh, w0, steps))
        return x, minrate(true, x, H)

    for nm, mdl in (("nominal", nominal), ("amplitude-aware", ampaware), ("oracle device (CSI error remains)", true)):
        print(f"  dev reference {nm:34s} {deploy(mdl, dev, dev_h)[1].mean():.3f}", flush=True)

    torch.manual_seed(0)
    mp_ = torch.zeros(L, M, requires_grad=True); ma = torch.ones(L, 2, requires_grad=True)
    md = torch.zeros(L, requires_grad=True); mo = torch.zeros(L, 2, requires_grad=True)
    S = dict(p=0.15, a=0.15, d=0.02, o=0.05)
    opt = torch.optim.Adam([dict(params=[mp_], lr=0.012), dict(params=[ma], lr=0.01), dict(params=[md], lr=0.003), dict(params=[mo], lr=0.005)])

    def build(p, a, d, o):
        return Dev(geo.Ws(d, o), p, a[:, 0].clamp(0.05, 1.5), a[:, 1].clamp(0.05, 1.5), P, var)

    def cur():
        return build(mp_.detach(), ma.detach(), md.detach(), mo.detach())

    def errs():
        rm = torch.angle(torch.exp(1j * (mp_.detach() - psi_t))).pow(2).mean().sqrt().item()
        return rm, (ma.detach()[:, 1] - API).abs().mean().item(), (md.detach() - dl_t).abs().mean().item(), (mo.detach() - of_t).abs().mean().item()

    t0 = time.time(); log = []
    for u in range(U + 1):
        if u % 25 == 0:
            dr = deploy(cur(), dev, dev_h)[1].mean().item(); e = errs(); log.append((u, dr) + e)
            print(f"u={u:4d} dev {dr:.3f} | err: psi {e[0]:.3f} rad, weak amp {e[1]:.3f}, spacing {100*e[2]:.2f}%, offset {e[3]:.3f} pitch ({time.time()-t0:.0f}s)", flush=True)
        if u == U: break
        idx = torch.randperm(pool.shape[0])[:6]; Hb, Hbh = pool[idx], pool_h[idx]
        w0 = torch.randn(6, D) * 0.5
        e = {k: torch.randn((G // 2,) + s) for k, s in (("p", (L, M)), ("a", (L, 2)), ("d", (L,)), ("o", (L, 2)))}
        e = {k: torch.cat([v, -v], 0) for k, v in e.items()}
        R = torch.zeros(G)
        for g in range(G):
            mdl = build(mp_.detach() + S["p"] * e["p"][g], ma.detach() + S["a"] * e["a"][g], md.detach() + S["d"] * e["d"][g], mo.detach() + S["o"] * e["o"][g])
            x = sgn(relax(mdl, Hbh, w0, 20))
            r = minrate(true, x, Hb)
            R[g] = (r * (1 + 0.05 * torch.randn_like(r))).mean()                   # noisy measurement
        A = (R - R.mean()) / (R.std() + 1e-6)
        lp = 0
        for k, prm in (("p", mp_), ("a", ma), ("d", md), ("o", mo)):
            dims = tuple(range(1, e[k].dim()))
            lp = lp + (-0.5 * ((e[k] * S[k] - (prm - prm.detach()).unsqueeze(0)) / S[k]) ** 2).sum(dims)
        loss = -(A * lp).mean() / D
        opt.zero_grad(); loss.backward(); opt.step()

    fin = cur()
    out = dict(log=log, errs=errs())
    print("== FINAL: 40 TEST channels, designer uses estimated CSI (NMSE -10 dB), 200 steps, round to {0,pi}, rate on TRUE device+channel ==", flush=True)
    rows = (("nominal-model gradient", nominal, test_h), ("amplitude-aware gradient", ampaware, test_h),
            ("CALIBRATED (RL) gradient", fin, test_h), ("oracle-device gradient (CSI error remains)", true, test_h),
            ("oracle-device gradient + PERFECT CSI (upper ref)", true, test))
    for nm, mdl, Hh in rows:
        x, r = deploy(mdl, test, Hh, 200)
        pp = polish(true, x, test, 10)
        out[nm] = r.numpy(); out[nm + "+polish"] = pp.numpy()
        print(f"  {nm:50s} {r.mean():.3f}   +polish(10, exact meas.) {pp.mean():.3f}", flush=True)
    for b in ("amplitude-aware gradient", "nominal-model gradient"):
        d = out["CALIBRATED (RL) gradient"] - out[b]
        print(f"  calibrated - {b}: {d.mean():+.3f} +- {1.96*d.std()/np.sqrt(len(d)):.3f}, wins on {100*(d>0).mean():.0f}%", flush=True)
    np.save(f"calibration_full_{tag}.npy", np.array(out, dtype=object))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
