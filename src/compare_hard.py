"""GRPO (hybrid) vs normal gradient method on the HARD system (K=10, M=64, L=6, 384 bits, fixed unknown phase offsets).
40 TEST channels (never used for tuning; GRPO lr=0.05 sigma0=0.5 chosen on dev channels). All rates: x=+-1 <=> phases {0,pi}, true device.
Methods:  grad        nominal-model relaxed gradient, 200 steps, 0 device queries
          grad+pol    + 10-sweep black-box bit-flip polish (3840 queries)
          grpo        hybrid GRPO, 100 updates x 16 samples = 1600 queries, deploy best queried sample
          grpo+pol    + same polish
          restart     same sampling/refinement with mu fixed (no learning), 1600 queries, best sample
          restart+pol
Usage: python3 compare_hard.py <mismatch_sigma>"""
import sys, json, numpy as np, torch
torch.set_num_threads(3)
from big_sys import make_env, Sys, sgn, rate, polish_true, relax
from hybrid_grpo import refine, get_channels, K, M, L, D, G

def search(nom, true, H, method, U=100, lr=0.05, sigma0=0.5, seed=0):
    B = H.shape[0]
    Hrep = H.unsqueeze(1).expand(B, G, *H.shape[1:]).reshape(B * G, *H.shape[1:])
    torch.manual_seed(seed)
    mu = torch.zeros(B, D, requires_grad=(method == "grpo")); ls = torch.full((B, 1), float(np.log(sigma0)), requires_grad=(method == "grpo"))
    opt = torch.optim.Adam([mu, ls], lr=lr) if method == "grpo" else None
    best = torch.full((B,), -1.0); bx = torch.ones(B, D); curve = []
    for u in range(U):
        w0 = mu.detach().unsqueeze(1) + ls.exp().detach().unsqueeze(1) * torch.randn(B, G, D)
        x = sgn(refine(nom, Hrep, w0.reshape(B * G, D)))
        r = rate(true, x, Hrep).reshape(B, G)
        top, idx = r.max(1)
        xb = x.reshape(B, G, D)[torch.arange(B), idx]
        imp = top > best
        best = torch.where(imp, top, best); bx[imp] = xb[imp]
        curve.append(best.mean().item())
        if method == "grpo":
            A = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
            logp = (-0.5 * ((w0 - mu.unsqueeze(1)) / ls.exp().unsqueeze(1)) ** 2 - ls.unsqueeze(1)).sum(-1)
            loss = -(A * logp).mean() / D
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): ls.clamp_(np.log(0.15), np.log(2.0))
    return bx, best, curve

def main():
    sig = float(sys.argv[1])
    env = make_env(K, M, L, 0); nom = Sys(env)
    psi = torch.randn(L, M, generator=torch.Generator().manual_seed(7)) * sig if sig > 0 else None
    true = Sys(env, psi)
    H = get_channels("test"); B = H.shape[0]
    w0 = torch.randn(B, D, generator=torch.Generator().manual_seed(1)) * 0.5
    res = {}
    xg = sgn(relax(nom, H, w0, 200)); res["grad"] = rate(true, xg, H).numpy()
    res["grad+pol"] = polish_true(true, xg, H, 10)[1].numpy()
    xo = sgn(relax(true, H, w0, 200)); res["oracle-model grad (reference)"] = rate(true, xo, H).numpy()
    for m in ("grpo", "restart"):
        bx, best, curve = search(nom, true, H, m)
        res[m] = best.numpy(); res[m + "_curve"] = np.array(curve)
        res[m + "+pol"] = polish_true(true, bx, H, 10)[1].numpy()
    out = {k: float(np.mean(v)) for k, v in res.items() if not k.endswith("_curve")}
    out["B"] = B
    # paired grpo - grad and grpo - restart
    for a, b in (("grpo", "grad"), ("grpo", "restart"), ("grpo+pol", "grad+pol"), ("grpo+pol", "restart+pol")):
        d = res[a] - res[b]; out[f"{a}-{b}"] = (float(d.mean()), float(d.std() / np.sqrt(B) * 1.96), float((d > 0).mean()))
    print(json.dumps({"mismatch_sigma": sig, **out}, indent=1), flush=True)
    np.save(f"compare_hard_{sig}.npy", np.array(res, dtype=object))

if __name__ == "__main__":
    main()
