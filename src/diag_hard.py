"""Where does the relaxed-gradient method break? Optimizer sees (H_est, Ws_est); the rate is earned on the TRUE (H, Ws), 1-bit only.
 (a) CSI error: H_est = H + relative error (env_ris.add_csi_estimation_error)
 (b) SYSTEMATIC model mismatch: Ws_est = Ws * exp(j*delta), delta fixed (one draw per 'device') ~ N(0, sigma^2)
 (c) multi-start: best of S random starts by exact true rate (perfect info)  -> exploration headroom
Reference rows: perfect info (S=1), random config."""
import math, numpy as np, torch
torch.set_num_threads(1)
from unrolled_meta import Phys, softmin, L, M, K
from diag_relax import rates_x
from meta_offline import make_tasks
from env_ris import add_csi_estimation_error

def relax(phys_est, H_est, w0, N=200, alpha=0.1, beta_end=5.0):
    w = w0.clone()
    for n in range(N):
        beta = 1.0 + (beta_end - 1.0) * n / N
        with torch.enable_grad():
            t = w.detach().requires_grad_(True)
            g, = torch.autograd.grad(softmin(rates_x(phys_est, torch.tanh(beta * t), H_est)).sum(), t)
        w = w + alpha * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    return w

def true_rate(phys, w, H):
    with torch.no_grad():
        return rates_x(phys, torch.where(w < 0, -1.0, 1.0), H).min(-1).values

def main():
    phys = Phys()
    te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:100]
    H = torch.tensor(np.stack(te), dtype=torch.complex64)
    gen = torch.Generator().manual_seed(1)
    w0 = torch.randn(100, L * M, generator=gen) * 0.5
    ref = true_rate(phys, relax(phys, H, w0), H)
    print(f"perfect information, S=1:                 {ref.mean():.3f}", flush=True)
    print(f"random 1-bit configuration:               {true_rate(phys, w0, H).mean():.3f}", flush=True)
    rng = np.random.default_rng(0)
    for std in (0.1, 0.3, 0.6, 1.0):
        He = torch.tensor(np.stack([add_csi_estimation_error(h, std, rng) for h in te]), dtype=torch.complex64)
        r = true_rate(phys, relax(phys, He, w0), H)
        print(f"(a) CSI error std={std:<4}                    {r.mean():.3f}  ({100*r.mean()/ref.mean():.0f}% of perfect)", flush=True)
    for sig in (0.1, 0.3, 0.6, 1.0):
        gen2 = torch.Generator().manual_seed(7)
        pe = Phys()
        pe.Ws = [w * torch.polar(torch.ones(w.shape), torch.randn(w.shape, generator=gen2) * sig) for w in phys.Ws]
        r = true_rate(phys, relax(pe, H, w0), H)
        print(f"(b) fixed Ws phase mismatch sigma={sig:<4} rad  {r.mean():.3f}  ({100*r.mean()/ref.mean():.0f}% of perfect)", flush=True)
    for S in (4, 16):
        best = ref.clone()
        for s in range(1, S):
            g3 = torch.Generator().manual_seed(100 + s)
            ws = torch.randn(100, L * M, generator=g3) * 0.5
            best = torch.maximum(best, true_rate(phys, relax(phys, H, ws), H))
        print(f"(c) multi-start S={S:<2} (perfect info, best of S)  {best.mean():.3f}  ({100*best.mean()/ref.mean():.0f}% of S=1)", flush=True)

if __name__ == "__main__":
    main()
