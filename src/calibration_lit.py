"""Calibration GRPO against LITERATURE-STYLE SIM impairments (K=10, M=64, L=6, 1-bit, commanded phases exactly {0,pi}).
True device (unknown to the designer):
  (1) phase-tuning error  psi_{l,m} ~ von Mises(0, kappa), i.i.d. zero-mean      [Li et al., SIM phase-tuning error model]
  (2) phase-dependent amplitude (practical RIS/SIM element, Abeywickrama-type):
        beta(phi) = (1-bmin) ((sin(phi-phi0)+1)/2)^kap + bmin ,  bmin=.2, phi0=.43pi, kap=1.6
        -> state 0 transmits with amplitude a0=beta(0), state pi with api=beta(pi)    [RIS hardware-impairment literature]
  (3) inter-layer spacing deviation: every layer spacing scaled by (1+delta)         [interlayer-error-calibration motivation]
Realised coefficient of element with command x=+-1:  d = amp(x) * x * exp(j psi),  amp(+1)=a0, amp(-1)=api.
Learned latent (policy N(m, s^2)): psi_hat (L x M), per-layer amplitudes (a0_hat, api_hat), delta_hat.  Reward = true-device
min-rate of round(relaxed-gradient solution under the candidate model) averaged over fresh channels (rates only).
Usage: python3 calibration_lit.py <scenario: amp|amp_spacing> <updates> <G> <lr_psi> <tag>"""
import sys, time, numpy as np, torch
torch.set_num_threads(3)
from big_sys import make_env, channels, Sys, sgn, rate, polish_true, relax
K, M, L = 10, 64, 6
D = L * M
PI = float(np.pi)


def beta(phi, bmin=0.2, phi0=0.43 * PI, kap=1.6):
    return (1 - bmin) * ((np.sin(phi - phi0) + 1) / 2) ** kap + bmin


# operating point: the two 1-bit states sit at phi0+0.2 and phi0+0.2+pi of the curve (a global phase per layer does not change any rate),
# amplitudes normalised so the stronger state has unit amplitude (transmit power compensates the common loss)
_phi1 = 0.43 * PI + 0.2
_b = np.array([beta(_phi1), beta(_phi1 + PI)])
A0, API = float(_b[0] / _b.max()), float(_b[1] / _b.max())


class Sys2:
    """Torch physics with per-state amplitudes, element phase offsets and (interpolated) layer-spacing-dependent Ws."""
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


class World:
    def __init__(self):
        self.env = make_env(K, M, L, 0)
        e = self.env
        self.P = torch.tensor(e.P, dtype=torch.float32); self.var = float(e.var)
        self.grid = np.linspace(-0.12, 0.12, 49)
        d0 = e.d_layer
        self.Wgrid = []
        for dl in self.grid:
            e.d_layer = d0 * (1 + dl)
            self.Wgrid.append([torch.tensor(w, dtype=torch.complex64) for w in e._build_Ws()])
        e.d_layer = d0

    def Ws(self, delta):                         # linear interpolation on the grid
        delta = float(np.clip(delta, self.grid[0], self.grid[-1]))
        i = min(int((delta - self.grid[0]) / (self.grid[1] - self.grid[0])), len(self.grid) - 2)
        t = (delta - self.grid[i]) / (self.grid[i + 1] - self.grid[i])
        return [(1 - t) * a + t * b for a, b in zip(self.Wgrid[i], self.Wgrid[i + 1])]

    def model(self, psi, a0, api, delta):
        return Sys2(self.Ws(delta), psi, a0, api, self.P, self.var)


def main():
    scen, U, G, lr_psi, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    mode = sys.argv[6] if len(sys.argv) > 6 else "pool"     # pool: calibrate on 200 training channels | one: on ONE fixed channel
    world = World()
    delta_true = 0.05 if scen == "amp_spacing" else 0.0
    rng = np.random.default_rng(7)
    psi_true = torch.tensor(rng.vonmises(0.0, 11.0, size=(L, M)), dtype=torch.float32)           # circular std ~0.3 rad
    one = torch.ones(L)
    true = world.model(psi_true, A0 * one, API * one, delta_true)
    nominal = world.model(torch.zeros(L, M), one, one, 0.0)
    ampaware = world.model(torch.zeros(L, M), A0 * one, API * one, 0.0)      # informed designer: knows beta(phi) curve, not psi/spacing
    oracle = true
    print(f"scenario {scen}: a0={A0:.3f} api={API:.3f} delta={delta_true} psi_std={psi_true.std():.3f}", flush=True)
    pool = channels(K, M, L, list(range(1000, 1020)), n_steps=100, stride=10)
    dev = channels(K, M, L, [4000, 4001], n_steps=100, stride=10)[:10]
    test = channels(K, M, L, [5000, 5001, 5002, 5003], n_steps=100, stride=10)[:40]
    H0 = pool[:1]                                            # the single calibration channel in "one" mode
    print(f"calibration mode: {mode}", flush=True)

    def deploy(model, H, steps=50, seed=1):
        w0 = torch.randn(H.shape[0], D, generator=torch.Generator().manual_seed(seed)) * 0.5
        return rate(true, sgn(relax(model, H, w0, steps)), H)

    for nm, mdl in (("nominal", nominal), ("amplitude-aware", ampaware), ("oracle", oracle)):
        print(f"  dev reference {nm:16s} {deploy(mdl, dev).mean():.3f}", flush=True)
    torch.manual_seed(0)
    m_psi = torch.zeros(L, M, requires_grad=True); m_a = torch.ones(L, 2, requires_grad=True); m_d = torch.zeros(1, requires_grad=True)
    s_psi, s_a, s_d = 0.15, 0.15, 0.02
    opt = torch.optim.Adam([dict(params=[m_psi], lr=lr_psi), dict(params=[m_a], lr=0.01), dict(params=[m_d], lr=0.003)])
    t0 = time.time(); log = []

    def cur_model():
        with torch.no_grad():
            return world.model(m_psi.detach(), m_a.detach()[:, 0].clamp(0.05, 1.5), m_a.detach()[:, 1].clamp(0.05, 1.5), m_d.item())

    for u in range(U + 1):
        if u % 25 == 0:
            dr = deploy(cur_model(), dev).mean().item()
            log.append((u, dr))
            print(f"u={u:4d} dev deploy {dr:.3f} | a0_hat {m_a.detach()[:,0].mean():.2f} api_hat {m_a.detach()[:,1].mean():.2f} delta_hat {m_d.item():+.3f} ({time.time()-t0:.0f}s)", flush=True)
        if u == U: break
        Hb = H0.expand(6, *H0.shape[1:]).contiguous() if mode == "one" else pool[torch.randperm(pool.shape[0])[:6]]
        w0 = torch.randn(6, D) * 0.5
        e1 = torch.randn(G // 2, L, M); e1 = torch.cat([e1, -e1], 0)
        e2 = torch.randn(G // 2, L, 2); e2 = torch.cat([e2, -e2], 0)
        e3 = torch.randn(G // 2, 1); e3 = torch.cat([e3, -e3], 0)
        R = torch.zeros(G)
        for g in range(G):
            mdl = world.model(m_psi.detach() + s_psi * e1[g], (m_a.detach() + s_a * e2[g])[:, 0].clamp(0.05, 1.5),
                              (m_a.detach() + s_a * e2[g])[:, 1].clamp(0.05, 1.5), (m_d.detach() + s_d * e3[g]).item())
            R[g] = rate(true, sgn(relax(mdl, Hb, w0, 20)), Hb).mean()
        A = (R - R.mean()) / (R.std() + 1e-6)
        lp = (-0.5 * e1 ** 2).sum((1, 2)) * 0 + (-0.5 * ((e1 * s_psi - (m_psi - m_psi.detach()).unsqueeze(0)) / s_psi) ** 2).sum((1, 2)) \
            + (-0.5 * ((e2 * s_a - (m_a - m_a.detach()).unsqueeze(0)) / s_a) ** 2).sum((1, 2)) \
            + (-0.5 * ((e3 * s_d - (m_d - m_d.detach()).unsqueeze(0)) / s_d) ** 2).sum(1)
        loss = -(A * lp).mean() / D
        opt.zero_grad(); loss.backward(); opt.step()
    fin = cur_model()
    out = dict(scen=scen, mode=mode, log=log, m_psi=m_psi.detach().numpy(), m_a=m_a.detach().numpy(), m_d=m_d.item(), psi_true=psi_true.numpy())
    H0r = H0.expand(20, *H0.shape[1:]).contiguous()
    print("== on the ONE calibration channel H0 (20 random starts, 200 steps) ==", flush=True)
    for nm, mdl in (("nominal-model gradient", nominal), ("amplitude-aware gradient (knows beta(phi))", ampaware), ("CALIBRATED (RL) gradient", fin), ("oracle-model gradient", oracle)):
        r = deploy(mdl, H0r, 200); out["H0 " + nm] = r.numpy()
        print(f"  {nm:44s} mean {r.mean():.3f}  best-of-20 {r.max():.3f}", flush=True)
    rm = lambda a, b: torch.angle(torch.exp(1j * (a - b))).pow(2).mean().sqrt().item()
    print(f"  phase-offset error rmse(m,psi_true) = {rm(m_psi.detach(), psi_true):.3f} rad (start {rm(torch.zeros(L, M), psi_true):.3f})", flush=True)
    print("== FINAL on 40 TEST channels (200 relaxed steps, then round to {0,pi}; rate on the true device) ==", flush=True)
    for nm, mdl in (("nominal-model gradient", nominal), ("amplitude-aware gradient (knows beta(phi))", ampaware), ("CALIBRATED (RL) gradient", fin), ("oracle-model gradient", oracle)):
        r = deploy(mdl, test, 200); out[nm] = r.numpy()
        w0 = torch.randn(test.shape[0], D, generator=torch.Generator().manual_seed(1)) * 0.5
        x = sgn(relax(mdl, test, w0, 200))
        pp = polish_true(true, x, test, 10)[1]
        out[nm + "+polish"] = pp.numpy()
        print(f"  {nm:44s} {r.mean():.3f}   +polish(10 sweeps) {pp.mean():.3f}", flush=True)
    d = out["CALIBRATED (RL) gradient"] - out["amplitude-aware gradient (knows beta(phi))"]
    print(f"  calibrated - amplitude-aware: {d.mean():+.3f} +- {1.96*d.std()/np.sqrt(len(d)):.3f}, calibrated wins on {100*(d>0).mean():.0f}% of channels", flush=True)
    d = out["CALIBRATED (RL) gradient"] - out["nominal-model gradient"]
    print(f"  calibrated - nominal:         {d.mean():+.3f} +- {1.96*d.std()/np.sqrt(len(d)):.3f}, wins on {100*(d>0).mean():.0f}%", flush=True)
    np.save(f"calibration_lit_{tag}.npy", np.array(out, dtype=object))
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
