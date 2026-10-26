"""Real-valued relaxation of the 1-bit problem: coefficient d = tanh(beta*w) in [-1,1] (1-bit <=> d = +-1), beta annealed.
Exact physical gradient on the smooth-min rate; hard evaluation = sign(w). Random init, 100 held-out channels."""
import math, numpy as np, torch
torch.set_num_threads(1)
from unrolled_meta import Phys, softmin, L, M, K
from meta_offline import make_tasks, minr as np_minr

def rates_x(phys, x, H):
    B = x.shape[0]
    d = [torch.complex(x[:, l * M:(l + 1) * M], torch.zeros_like(x[:, l * M:(l + 1) * M])) for l in range(L)]
    Gv = torch.eye(M, dtype=torch.complex64).unsqueeze(0).expand(B, M, M)
    for idx in range(L - 1, 0, -1):
        Gv = torch.matmul(Gv, d[idx].unsqueeze(-1) * phys.Ws[idx].unsqueeze(0))
    Gv = Gv * d[0].unsqueeze(1)
    Full = torch.matmul(torch.matmul(H.conj().transpose(1, 2), Gv), phys.Ws[0])
    power = Full.abs() ** 2 * phys.P.view(1, 1, K)
    sig = torch.diagonal(power, dim1=-2, dim2=-1)
    return torch.log2(1 + sig / (power.sum(-1) - sig + phys.var))

def main():
    phys = Phys()
    te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:100]
    H = torch.tensor(np.stack(te), dtype=torch.complex64)
    # cross-check: sign(w) -> bits -> numpy evaluator
    w = torch.randn(5, L * M); ref = [np_minr((w[i] < 0).numpy().astype(float) * np.pi, te[i]) for i in range(5)]
    mine = rates_x(phys, torch.where(w < 0, -1.0, 1.0), H[:5]).min(-1).values.numpy()
    print("check vs numpy evaluator: max diff %.2e" % np.abs(np.array(ref) - mine).max(), flush=True)
    gen = torch.Generator().manual_seed(1)
    w0 = torch.randn(100, L * M, generator=gen) * 0.5
    N = 200
    for alpha in (0.02, 0.05, 0.1):
        for beta_end in (5.0, 20.0):
            w = w0.clone(); out = {}
            for n in range(N + 1):
                if n in (0, 5, 10, 20, 50, 100, 200):
                    hard = torch.where(w < 0, -1.0, 1.0)
                    out[n] = float(rates_x(phys, hard, H).min(-1).values.mean())
                beta = 1.0 + (beta_end - 1.0) * n / N
                with torch.enable_grad():
                    t = w.detach().requires_grad_(True)
                    g, = torch.autograd.grad(softmin(rates_x(phys, torch.tanh(beta * t), H)).sum(), t)
                w = w + alpha * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
            print(f"  alpha={alpha:<5} beta 1->{beta_end:<4}: " + " ".join(f"n={n}:{out[n]:.3f}" for n in out), flush=True)

if __name__ == "__main__":
    main()
