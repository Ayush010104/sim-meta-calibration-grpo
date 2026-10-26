"""Can exact physical gradient ascent optimize the 1-bit min-rate at all? Random init, 100 held-out channels."""
import math, numpy as np, torch
torch.set_num_threads(1)
from unrolled_meta import Phys, softmin, hard_minrate, L, M, K
from meta_offline import make_tasks
from grpo_ris import quantize_phase_ste, quantize_phase_hard_torch

def main():
    phys = Phys()
    te = make_tasks([5000 + i for i in range(5)], [(0.99, 1.4)], stride=20)[:100]
    H = torch.tensor(np.stack(te), dtype=torch.complex64)
    g0 = torch.Generator().manual_seed(1)
    th0 = (torch.rand(100, L * M, generator=g0) * 2 - 1) * math.pi
    checkpoints = (0, 1, 3, 10, 30, 100, 300)
    print("min-rate (hard 1-bit) after n normalized-gradient steps from RANDOM init:")
    for mode in ("ste", "continuous->round"):
        for alpha in (0.01, 0.03, 0.1, 0.3, 0.6):
            th = th0.clone(); row = {}
            for n in range(max(checkpoints) + 1):
                if n in checkpoints:
                    row[n] = float(hard_minrate(phys, th, H).mean())
                with torch.enable_grad():
                    t = th.detach().requires_grad_(True)
                    q = quantize_phase_ste(t, 1) if mode == "ste" else t
                    g, = torch.autograd.grad(softmin(phys.rates(q, H)).sum(), t)
                th = th + alpha * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
            cont = float(phys.rates(th, H).min(-1).values.mean()) if mode != "ste" else float("nan")
            print(f"  {mode:18s} alpha={alpha:<5} " + " ".join(f"n={n}:{row[n]:.3f}" for n in checkpoints)
                  + (f"   [continuous-phase min-rate at the end: {cont:.3f}]" if mode != "ste" else ""))
if __name__ == "__main__":
    main()
