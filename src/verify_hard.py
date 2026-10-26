"""Independent end-to-end check: relaxed optimization -> sign() -> phases in {0, pi} ONLY -> scored by three different evaluators."""
import math, numpy as np, torch
torch.set_num_threads(1)
from unrolled_meta import Phys, softmin, L, M
from diag_relax import rates_x
from meta_offline import make_tasks, env as get_env, minr as np_minr

def main():
    phys = Phys()
    te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:20]
    H = torch.tensor(np.stack(te), dtype=torch.complex64)
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(20, L * M, generator=gen) * 0.5
    N, alpha, beta_end = 200, 0.1, 5.0
    relaxed_rate = None
    for n in range(N):
        beta = 1.0 + (beta_end - 1.0) * n / N
        with torch.enable_grad():
            t = w.detach().requires_grad_(True)
            g, = torch.autograd.grad(softmin(rates_x(phys, torch.tanh(beta * t), H)).sum(), t)
        w = w + alpha * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
    with torch.no_grad():
        relaxed_rate = rates_x(phys, torch.tanh(beta_end * w), H).min(-1).values.numpy()   # what the RELAXED surface would give
    bits = (w < 0).numpy()
    phi = bits.astype(float) * np.pi                       # phases are exactly 0 or pi
    print("distinct phase values used:", np.unique(np.round(phi, 12)), "(0 and pi only)")
    a = rates_x(phys, torch.where(w < 0, -1.0, 1.0), H).min(-1).values.numpy()         # (a) my torch evaluator on d = +-1
    b = np.array([np_minr(phi[i], te[i]) for i in range(20)])                          # (b) numpy evaluator used by the local search / GRPO eval
    e = get_env(); c = []
    for i in range(20):                                                                  # (c) the ORIGINAL environment step/reward path used by the GRPO pipeline
        e.H_true = te[i]
        act = np.concatenate([np.concatenate([np.cos(phi[i, l*M:(l+1)*M]), np.sin(phi[i, l*M:(l+1)*M])]) for l in range(L)])[None, :]
        c.append(e.step(act[0])[2])
    c = np.array(c)
    print(f"mean min-rate over 20 held-out channels, 200 relaxed steps then sign():")
    print(f"  (a) torch, d=+-1            : {a.mean():.4f}")
    print(f"  (b) numpy evaluator (0/pi)  : {b.mean():.4f}   max|a-b| = {np.abs(a-b).max():.1e}")
    print(f"  (c) original env.step path  : {c.mean():.4f}   max|b-c| = {np.abs(b-c).max():.1e}")
    print(f"  for contrast, the RELAXED (soft, infeasible) surface would score: {relaxed_rate.mean():.4f}")

if __name__ == "__main__":
    main()
