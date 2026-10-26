"""Scaled-up, harder SIM system (generic K=N users/antennas, M elements/layer, L layers) + optional fixed unknown phase
offsets psi (device mismatch). Torch physics identical to env_ris (verified against numpy eval_minr_batch)."""
import sys, numpy as np, torch
torch.set_num_threads(1)
from moving_user_env import MovingUserEnv

TAU = 8.0
T_S = 0.5


def make_env(K, M, L, seed=0, rho=0.99, v=1.4):
    return MovingUserEnv(seed=seed, temporal_rho=rho, user_speed=v, T_s=T_S, direction_seed=seed,
                         num_antennas=K, num_RIS_elements=M, num_users=K, num_layers=L, dBS=30.0, radius=50.0)


def channels(K, M, L, seeds, n_steps=200, stride=20, rho=0.99, v=1.4):
    out = []
    for sd in seeds:
        e = make_env(K, M, L, sd, rho, v)
        H = [e.H_true.copy()]
        for _ in range(n_steps - 1):
            e.advance_channel(); H.append(e.H_true.copy())
        out += [H[t] for t in range(0, n_steps, stride)]
    return torch.tensor(np.stack(out), dtype=torch.complex64)


class Sys:
    def __init__(self, env, psi=None):
        self.K, self.M, self.L = env.K, env.M, env.L
        self.Ws = [torch.tensor(w, dtype=torch.complex64) for w in env.Ws]
        self.P = torch.tensor(env.P, dtype=torch.float32)
        self.var = float(env.var)
        if psi is not None:                       # absorb phase offsets into the layer matrices
            self.Ws = [torch.polar(torch.ones(self.M), psi[l]).unsqueeze(1) * self.Ws[l] for l in range(self.L)]

    def rates(self, x, H):                         # x real +-1 (or relaxed) of shape (B, L*M)
        B, M, L = x.shape[0], self.M, self.L
        d = [torch.complex(x[:, l*M:(l+1)*M], torch.zeros_like(x[:, l*M:(l+1)*M])) for l in range(L)]
        Gv = torch.eye(M, dtype=torch.complex64).unsqueeze(0).expand(B, M, M)
        for idx in range(L - 1, 0, -1):
            Gv = torch.matmul(Gv, d[idx].unsqueeze(-1) * self.Ws[idx].unsqueeze(0))
        Gv = Gv * d[0].unsqueeze(1)
        Full = torch.matmul(torch.matmul(H.conj().transpose(1, 2), Gv), self.Ws[0])
        power = Full.abs() ** 2 * self.P.view(1, 1, self.K)
        sig = torch.diagonal(power, dim1=-2, dim2=-1)
        return torch.log2(1 + sig / (power.sum(-1) - sig + self.var))


def softmin(r):
    return -(1.0 / TAU) * torch.logsumexp(-TAU * r, dim=-1)


def relax(model, H, w0, N=200, alpha=0.1, beta_end=5.0):
    w = w0.clone()
    for n in range(N):
        beta = 1.0 + (beta_end - 1.0) * n / N
        with torch.enable_grad():
            t = w.detach().requires_grad_(True)
            g, = torch.autograd.grad(softmin(model.rates(torch.tanh(beta * t), H)).sum(), t)
        w = w + alpha * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    return w


def sgn(w):
    return torch.where(w < 0, -1.0, 1.0)


def rate(sysm, x, H):
    with torch.no_grad():
        return sysm.rates(x, H).min(-1).values


def polish_true(sysm, x, H, sweeps=20):
    x = x.clone(); best = rate(sysm, x, H)
    for _ in range(sweeps):
        ch = False
        for j in torch.randperm(x.shape[1]).tolist():
            x2 = x.clone(); x2[:, j] = -x2[:, j]
            r2 = rate(sysm, x2, H); up = r2 > best + 1e-9
            if up.any():
                x[up] = x2[up]; best = torch.where(up, r2, best); ch = True
        if not ch: break
    return x, best


if __name__ == "__main__":
    K, M, L = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    env = make_env(K, M, L, 0)
    # physics check vs numpy evaluator
    from physics import eval_minr_batch
    H = channels(K, M, L, [5000, 5001], n_steps=100, stride=10)[:50]
    nom = Sys(env)
    phi = np.random.default_rng(0).integers(0, 2, (3, L * M)) * np.pi
    x = torch.tensor(np.where(phi > 1, -1.0, 1.0), dtype=torch.float32)
    env.H_true = H[0].numpy().astype(complex)
    print("physics check (diff):", np.abs(eval_minr_batch(phi, env) - nom.rates(x, H[:1].expand(3, -1, -1)).min(-1).values.numpy()).max(), flush=True)
    B = H.shape[0]
    w0 = torch.randn(B, L * M, generator=torch.Generator().manual_seed(1)) * 0.5
    print(f"system K={K} M={M} L={L} bits={L*M}; channels {B}; mean|H|={H.abs().mean():.2e}")
    print(f"random 1-bit: {rate(nom, sgn(w0), H).mean():.3f}", flush=True)
    for N in (50, 200):
        print(f"relaxed gradient (exact model) N={N}: {rate(nom, sgn(relax(nom, H, w0, N)), H).mean():.3f}", flush=True)
    xo = sgn(relax(nom, H, w0, 200)); base = rate(nom, xo, H)
    best = base.clone()
    for s in range(1, 8):
        ws = torch.randn(B, L * M, generator=torch.Generator().manual_seed(100 + s)) * 0.5
        best = torch.maximum(best, rate(nom, sgn(relax(nom, H, ws, 200)), H))
    print(f"relaxed multi-start best-of-8 (exact model): {best.mean():.3f}  ({100*best.mean()/base.mean():.0f}% of S=1)", flush=True)
    print(f"relaxed S=1 + polish(20, black-box): {polish_true(nom, xo, H, 20)[1].mean():.3f}", flush=True)
    for sig in (0.3, 0.6):
        psi = torch.randn(L, M, generator=torch.Generator().manual_seed(7)) * sig
        true = Sys(env, psi)
        xn = sgn(relax(nom, H, w0, 200)); xr = sgn(relax(true, H, w0, 200))
        print(f"mismatch sigma={sig}: nominal-model gradient {rate(true, xn, H).mean():.3f} | +polish {polish_true(true, xn, H, 20)[1].mean():.3f} | oracle-model gradient {rate(true, xr, H).mean():.3f}", flush=True)
