"""PART B: Calibration-GRPO on any configuration (K,M,L) and impairment level (impair_lib.LEVELS), one device draw.
Calibration latent = modeled class only (psi, per-layer amplitudes, spacing, lateral offset); unmodeled extras stay unknown.
Usage: python3 calibration_cfg.py K M L <level> <device_seed> <updates> <G> <tag>"""
import sys, time, numpy as np, torch
torch.set_num_threads(2)
from big_sys import sgn, relax
from impair_lib import System, Dev, LEVELS, minrate, design_rate

S_ = dict(p=0.15, a=0.15, d=0.02, o=0.05)
LR = dict(p=0.012, a=0.01, d=0.003, o=0.005)


def main():
    K, M, L = map(int, sys.argv[1:4]); lvname, dseed, U, G, tag = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
    lv = LEVELS[lvname]
    S = System(K, M, L); D = L * M
    th_t, ae, st = S.sample(lv, dseed)
    true = Dev(S, th_t, ae, st)
    pool = S.channels(list(range(1000, 1020)), 200); pool_h = S.estimate(pool, lv["csi_db"], 11)
    dev = S.channels([4000, 4001], 10); dev_h = S.estimate(dev, lv["csi_db"], 12)
    test = S.channels([5000, 5001, 5002, 5003], 40); test_h = S.estimate(test, lv["csi_db"], 13)
    nom = S.nominal()
    ampaware = dict(nom); ampaware["a"] = th_t["a"].clone()
    print(f"[{tag}] K={K} M={M} L={L} level={lvname} device={dseed} | unmodeled: amp_elem={'yes' if ae is not None else 'no'}, "
          f"stuck={int((st!=0).sum()) if st is not None else 0}", flush=True)
    torch.manual_seed(0)
    prm = {k: v.clone().requires_grad_(True) for k, v in nom.items()}
    opt = torch.optim.Adam([dict(params=[prm[k]], lr=LR[k]) for k in "pado"])
    log, t0 = [], time.time()
    for u in range(U + 1):
        if u % 25 == 0:
            r = design_rate(Dev(S, {k: v.detach() for k, v in prm.items()}), true, dev, dev_h, 50).mean().item()
            log.append((u, r))
            print(f"u={u:4d} dev {r:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if u == U: break
        idx = torch.randperm(pool.shape[0])[:6]; Hb, Hbh = pool[idx], pool_h[idx]
        w0 = torch.randn(6, D) * 0.5
        e = {k: torch.randn((G // 2,) + tuple(nom[k].shape)) for k in "pado"}
        e = {k: torch.cat([v, -v], 0) for k, v in e.items()}
        R = torch.zeros(G)
        for g in range(G):
            cand = {k: prm[k].detach() + S_[k] * e[k][g] for k in "pado"}
            r = minrate(true, sgn(relax(Dev(S, cand), Hbh, w0, 20)), Hb)
            R[g] = (r * (1 + lv["noise"] * torch.randn_like(r))).mean()
        A = (R - R.mean()) / (R.std() + 1e-6)
        lp = 0
        for k in "pado":
            dims = tuple(range(1, e[k].dim()))
            lp = lp + (-0.5 * ((e[k] * S_[k] - (prm[k] - prm[k].detach()).unsqueeze(0)) / S_[k]) ** 2).sum(dims)
        loss = -(A * lp).mean() / D
        opt.zero_grad(); loss.backward(); opt.step()
    th = {k: v.detach() for k, v in prm.items()}
    rows = [("ideal-model gradient", Dev(S, nom), test_h), ("amplitude-aware gradient", Dev(S, ampaware), test_h),
            ("CALIBRATED (RL) gradient", Dev(S, th), test_h), ("oracle, modeled class only", Dev(S, th_t), test_h),
            ("oracle, full device", true, test_h), ("oracle, full device + perfect CSI", true, test)]
    out = dict(cfg=(K, M, L), level=lvname, device=dseed, log=log)
    print(f"== [{tag}] FINAL on 40 test channels ==", flush=True)
    for nm, mdl, Hh in rows:
        r = design_rate(mdl, true, test, Hh, 200); out[nm] = r.numpy()
        print(f"  {nm:36s} {r.mean():.3f}", flush=True)
    for b in ("amplitude-aware gradient", "ideal-model gradient"):
        d = out["CALIBRATED (RL) gradient"] - out[b]
        print(f"  calibrated - {b}: {d.mean():+.3f} +- {1.96*d.std()/np.sqrt(len(d)):.3f}, wins {100*(d>0).mean():.0f}%", flush=True)
    out["err"] = dict(psi=torch.angle(torch.exp(1j * (th["p"] - th_t["p"]))).pow(2).mean().sqrt().item(),
                      psi_start=torch.angle(torch.exp(1j * th_t["p"])).pow(2).mean().sqrt().item(),
                      amp=(th["a"][:, 1] - th_t["a"][:, 1]).abs().mean().item(), amp_start=(1 - th_t["a"][:, 1]).abs().mean().item(),
                      spacing=(th["d"] - th_t["d"]).abs().mean().item(), spacing_start=th_t["d"].abs().mean().item(),
                      offset=(th["o"] - th_t["o"]).abs().mean().item(), offset_start=th_t["o"].abs().mean().item())
    print("  parameter errors (start -> learned):", {k: round(v, 4) for k, v in out["err"].items()}, flush=True)
    np.save(f"calcfg_{tag}.npy", np.array(out, dtype=object))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
