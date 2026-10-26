"""HARD configuration: 1-bit SIM whose elements have FIXED, UNKNOWN phase offsets psi_{l,m} ~ N(0, sigma^2)
(fabrication / calibration error). Commanded phases are still exactly {0, pi}; the realised coefficient is
d = x * exp(j psi), x = +-1. The designer's NOMINAL model has psi = 0, so exact-gradient methods that rely on the model
optimise the wrong function; only reward (measured rate) feedback sees the true device.
Absorption: diag(d e^{j psi}) Ws_l = diag(d) (diag(e^{j psi}) Ws_l)  => true device = nominal physics with Ws'_l.
"""
import numpy as np, torch
torch.set_num_threads(1)
from unrolled_meta import Phys, softmin, L, M, K
from diag_relax import rates_x
from meta_offline import make_tasks


def device(sigma, seed=0, psi=None):
    ph = Phys()
    g = torch.Generator().manual_seed(seed)
    if psi is None:
        psi = torch.randn(L, M, generator=g) * sigma
    ph.Ws = [torch.polar(torch.ones(M), psi[l]).unsqueeze(1) * ph.Ws[l] for l in range(L)]
    return ph, psi


def relax(phys_model, H, w0, N=200, alpha=0.1, beta_end=5.0):
    w = w0.clone()
    for n in range(N):
        beta = 1.0 + (beta_end - 1.0) * n / N
        with torch.enable_grad():
            t = w.detach().requires_grad_(True)
            g, = torch.autograd.grad(softmin(rates_x(phys_model, torch.tanh(beta * t), H)).sum(), t)
        w = w + alpha * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    return w


def rate(phys, x, H):
    with torch.no_grad():
        return rates_x(phys, x, H).min(-1).values


def sgn(w):
    return torch.where(w < 0, -1.0, 1.0)


def polish_true(phys, x, H, sweeps=20):
    """first-improvement bit-flip sweeps scored by the TRUE device (black-box queries)"""
    x = x.clone()
    best = rate(phys, x, H)
    for _ in range(sweeps):
        changed = False
        for j in torch.randperm(L * M).tolist():
            x2 = x.clone(); x2[:, j] = -x2[:, j]
            r2 = rate(phys, x2, H)
            up = r2 > best + 1e-9
            if up.any():
                x[up] = x2[up]; best = torch.where(up, r2, best); changed = True
        if not changed:
            break
    return x, best


def test_set(n=100):
    te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:n]
    return torch.tensor(np.stack(te), dtype=torch.complex64)


if __name__ == "__main__":
    import sys
    H = test_set()
    nom = Phys()
    gen = torch.Generator().manual_seed(1)
    w0 = torch.randn(100, L * M, generator=gen) * 0.5
    print("sigma | nominal-grad | +polish(true) | oracle-grad | +polish | random+polish | oracle multi-start S=8")
    for sig in (0.0, 0.3, 0.5, 0.8):
        true, psi = device(sig, 0)
        xn = sgn(relax(nom, H, w0)); xo = sgn(relax(true, H, w0))
        rn, ro = rate(true, xn, H).mean(), rate(true, xo, H).mean()
        pn = polish_true(true, xn, H)[1].mean(); po = polish_true(true, xo, H)[1].mean()
        pr = polish_true(true, sgn(w0), H)[1].mean()
        best = rate(true, xo, H)
        for s in range(1, 8):
            ws = torch.randn(100, L * M, generator=torch.Generator().manual_seed(100 + s)) * 0.5
            best = torch.maximum(best, rate(true, sgn(relax(true, H, ws)), H))
        print(f"{sig:.1f}   | {rn:.3f}        | {pn:.3f}         | {ro:.3f}       | {po:.3f}   | {pr:.3f}         | {best.mean():.3f}", flush=True)
