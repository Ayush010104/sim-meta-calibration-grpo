"""
Exact MINLP formulation of the 1-/2-bit RIS/SIM phase-quantization problem
(the notebook's "GRPO vs Direct" max-min-rate task), solved with SCIP --
a genuine global mixed-integer nonlinear solver (nonconvex spatial
branch-and-bound with a certified dual bound / optimality gap), not a
heuristic.

------------------------------------------------------------------------
Model
------------------------------------------------------------------------
Decision variables: one discrete phase level per (layer l, RIS element m),
chosen from levels = 2**n_bits uniformly spaced angles on the unit circle
(exactly the notebook's quantization grid). Encoded as one-hot binaries:

    x[l,m,v] in {0,1},  sum_v x[l,m,v] = 1                 (l=0..L-1, m=0..M-1)
    d_re[l,m] = sum_v x[l,m,v]*cos(v*2*pi/levels)   (const * binary, LINEAR)
    d_im[l,m] = sum_v x[l,m,v]*sin(v*2*pi/levels)

Physics (identical to env._get_Gv / env.calculate_reward, re-derived as a
left-to-right row-vector recursion so we never need the full MxM Gv
matrix as decision variables -- only a (K x M) running vector per stage,
since K << M):

    u^(0)          = H^H                                    (K x M, CONSTANT)
    for l = L-1, L-2, ..., 1, 0:
        A^(l) = u^(prev) * diag(d_re[l,:] + i d_im[l,:])     (BILINEAR: prev-stage var * d)
        u^(next) = A^(l) @ Ws[l]      (l>0, M x M, linear -- Ws is data)
                 = A^(l) @ Ws[0]      (l==0, this IS Full, K x N)

Every "diag-scale" step is bilinear (a continuous running-vector entry
times a {0,1}-weighted sum of constants); every "propagate through Ws"
step is linear because Ws is fixed data. This mirrors exactly what
env._get_Gv / env.calculate_reward computes, just without ever forming
the (M x M) Gv explicitly.

Rates: since rate_k = log2(1 + SINR_k) is strictly increasing in SINR_k,
   argmax_x  min_k rate_k(x)   ==   argmax_x  min_k SINR_k(x)
so the MINLP maximizes min-SINR directly (avoids a transcendental log2 in
the model; the reported number is converted back to a rate with log2 for
comparison with the notebook's metric).

    power[k,o] = (Full_re[k,o]^2 + Full_im[k,o]^2) * P[o]        (QUADRATIC)
    signal[k]  = power[k,k]
    interf[k]  = sum_{o != k} power[k,o]
    maximize t   s.t.   t * (interf[k] + noise_var) <= signal[k]  (BILINEAR)  for all k

This is an exact, nonconvex MIQCP -- SCIP solves it via spatial
branch-and-bound and reports a primal (best found) and dual (proven)
bound, i.e. a real optimality gap, not just a locally-improved heuristic.
"""
import time
import numpy as np
from pyscipopt import Model, quicksum

from physics import phi_to_action, eval_minr_batch, greedy_bitflip_local_search, quantize_phase_hard


def _interval_bound(env, h_scale=1.0):
    """L-infinity bound on every entry of every intermediate (K x M)
    running-vector stage, via exact triangle-inequality propagation
    (diag-scale by a unit-modulus number never increases magnitude;
    multiplying by a fixed matrix Ws scales the bound by its max
    column-abs-sum). Used only as variable bounds to help SCIP's
    presolve/relaxations -- not required for correctness."""
    HH = env.H_true.conj().T * h_scale
    B = float(np.max(np.abs(HH)))
    bounds = {}
    cur = B
    for l in range(env.L - 1, -1, -1):
        bounds[('diag', l)] = cur
        Wl = env.Ws[l] if l > 0 else env.Ws[0]
        col_abs_sum = float(np.abs(Wl).sum(axis=0).max())
        cur = cur * col_abs_sum
        bounds[('ws', l)] = cur
    return bounds


def _numeric_scale(env):
    """The channel gains here are physically tiny (large path loss) and
    the noise variance tinier still (e.g. ~1e-12), so unscaled
    signal/interference/power live around 1e-8..1e-5. SCIP's default
    feasibility tolerance is an ABSOLUTE tolerance (~1e-6..1e-7): applied
    to values that small it is not tight at all, and across the many
    chained equality constraints in this model that slack compounds into
    materially wrong decoded solutions (verified empirically -- without
    this rescale the solver reports solutions whose claimed objective
    does not match independent re-evaluation of the same phases).

    Fix: solve an EXACTLY EQUIVALENT rescaled problem. Multiply H by a
    constant h_scale and var by h_scale**2: every power/signal/interference
    term scales by h_scale**2 while their RATIO (the SINR objective t) is
    unchanged, so the optimal x (and therefore the optimal phi) is
    identical -- only the model's internal numerics improve. h_scale is
    chosen so the largest achievable |Full| entry, and hence power, lands
    near O(1).
    """
    bnd0 = _interval_bound(env, h_scale=1.0)
    bnd_full = max(bnd0[('ws', 0)], 1e-300)
    h_scale = 1.0 / (bnd_full * np.sqrt(max(float(env.P.max()), 1e-300)))
    return float(h_scale)


def build_minlp(env, n_bits, warm_phi=None, h_scale=None):
    """Builds the SCIP model. Returns (model, x, t, Full_re, Full_im, meta)."""
    L, M, K, N = env.L, env.M, env.K, env.N
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    level_re = [float(np.cos(v * step)) for v in range(levels)]
    level_im = [float(np.sin(v * step)) for v in range(levels)]

    if h_scale is None:
        h_scale = _numeric_scale(env)
    bnd = _interval_bound(env, h_scale=h_scale)

    m = Model("RIS_SIM_quant_MINLP")
    m.setParam("numerics/feastol", 1e-7)

    # --- one-hot phase-level selection -----------------------------------
    x = {}
    for l in range(L):
        for mm in range(M):
            for v in range(levels):
                x[l, mm, v] = m.addVar(vtype="B", name=f"x_{l}_{mm}_{v}")
            m.addCons(quicksum(x[l, mm, v] for v in range(levels)) == 1, name=f"onehot_{l}_{mm}")

    def d_re(l, mm):
        return quicksum(x[l, mm, v] * level_re[v] for v in range(levels))

    def d_im(l, mm):
        return quicksum(x[l, mm, v] * level_im[v] for v in range(levels))

    # --- row-vector propagation u^(0)=H^H -> ... -> Full ------------------
    HH = env.H_true.conj().T * h_scale  # (K, M) constant, rescaled (see _numeric_scale)
    prev_re, prev_im = HH.real, HH.imag  # numpy constants for stage 0
    prev_is_const = True

    for l in range(L - 1, -1, -1):
        Bd = bnd[('diag', l)] * 1.05  # small safety margin on the bound
        A_re = {}
        A_im = {}
        for k in range(K):
            for mm in range(M):
                A_re[k, mm] = m.addVar(vtype="C", lb=-Bd, ub=Bd, name=f"Are_{l}_{k}_{mm}")
                A_im[k, mm] = m.addVar(vtype="C", lb=-Bd, ub=Bd, name=f"Aim_{l}_{k}_{mm}")
                dre, dim = d_re(l, mm), d_im(l, mm)
                if prev_is_const:
                    pr, pi = float(prev_re[k, mm]), float(prev_im[k, mm])
                    m.addCons(A_re[k, mm] == pr * dre - pi * dim, name=f"Are_def_{l}_{k}_{mm}")
                    m.addCons(A_im[k, mm] == pr * dim + pi * dre, name=f"Aim_def_{l}_{k}_{mm}")
                else:
                    pr, pi = prev_re[k, mm], prev_im[k, mm]
                    m.addCons(A_re[k, mm] == pr * dre - pi * dim, name=f"Are_def_{l}_{k}_{mm}")
                    m.addCons(A_im[k, mm] == pr * dim + pi * dre, name=f"Aim_def_{l}_{k}_{mm}")

        Wl = env.Ws[l] if l > 0 else env.Ws[0]
        out_dim = M if l > 0 else N
        Bnext = bnd[('ws', l)] * 1.05
        next_re = {}
        next_im = {}
        for k in range(K):
            for j in range(out_dim):
                next_re[k, j] = m.addVar(vtype="C", lb=-Bnext, ub=Bnext, name=f"Ure_{l}_{k}_{j}")
                next_im[k, j] = m.addVar(vtype="C", lb=-Bnext, ub=Bnext, name=f"Uim_{l}_{k}_{j}")
                re_expr = quicksum(A_re[k, mm] * float(Wl[mm, j].real) - A_im[k, mm] * float(Wl[mm, j].imag)
                                    for mm in range(M))
                im_expr = quicksum(A_re[k, mm] * float(Wl[mm, j].imag) + A_im[k, mm] * float(Wl[mm, j].real)
                                    for mm in range(M))
                m.addCons(next_re[k, j] == re_expr, name=f"Ure_def_{l}_{k}_{j}")
                m.addCons(next_im[k, j] == im_expr, name=f"Uim_def_{l}_{k}_{j}")

        prev_re, prev_im = next_re, next_im
        prev_is_const = False

    Full_re, Full_im = prev_re, prev_im  # (K, N) SCIP vars

    # --- power / signal / interference / max-min-SINR epigraph ------------
    P = env.P
    var = env.var * h_scale ** 2  # rescaled consistently with H -- SINR (t) is unchanged
    power_ub = (bnd[('ws', 0)] * 1.05) ** 2 * float(P.max())
    power = {}
    for k in range(K):
        for o in range(N):
            power[k, o] = m.addVar(vtype="C", lb=0.0, ub=power_ub, name=f"power_{k}_{o}")
            m.addCons(power[k, o] == (Full_re[k, o] * Full_re[k, o] + Full_im[k, o] * Full_im[k, o]) * float(P[o]),
                      name=f"power_def_{k}_{o}")

    signal = {}
    interf = {}
    for k in range(K):
        signal[k] = power[k, k]
        interf[k] = quicksum(power[k, o] for o in range(N) if o != k)

    # Physically-grounded SINR upper bound (worst case zero interference),
    # from the same interval bound used for the power variables -- NOT an
    # arbitrary constant. A loose t bound wrecks the McCormick relaxation
    # of the t*interf[k] bilinear epigraph constraint below and makes the
    # dual bound meaningless.
    t_ub = power_ub / float(var)
    t = m.addVar(vtype="C", lb=0.0, ub=t_ub, name="t")
    for k in range(K):
        m.addCons(t * (interf[k] + float(var)) <= signal[k], name=f"epigraph_{k}")

    m.setObjective(t, sense="maximize")

    meta = dict(L=L, M=M, K=K, N=N, levels=levels, step=step)

    return m, x, t, Full_re, Full_im, meta


def _add_warm_start(m, x, L, M, levels, step, warm_phi):
    """Register a full feasible solution (e.g. from greedy_bitflip_local_search)
    as an initial incumbent so SCIP's B&B starts with a strong primal bound
    and spends its time budget closing the GAP instead of searching for
    feasibility."""
    sol = m.createSol()
    for l in range(L):
        for mm in range(M):
            cur_level = int(round(np.mod(warm_phi[l * M + mm], 2 * np.pi) / step)) % levels
            for v in range(levels):
                m.setSolVal(sol, x[l, mm, v], 1.0 if v == cur_level else 0.0)
    accepted = m.trySol(sol)
    return accepted


def extract_phi(m, x, meta):
    """Decode the chosen phase level per (layer, element) from a solved
    SCIP model. pyscipopt variables have no .X accessor (that's a
    Gurobi-ism) -- values must be read via model.getVal(var)."""
    L, M, levels, step = meta["L"], meta["M"], meta["levels"], meta["step"]
    phi = np.zeros(L * M)
    for l in range(L):
        for mm in range(M):
            best_v, best_val = 0, -1.0
            for v in range(levels):
                val = m.getVal(x[l, mm, v])
                if val > best_val:
                    best_val, best_v = val, v
            phi[l * M + mm] = best_v * step
    return phi


def solve_minlp(env, n_bits, time_limit=120.0, warm_start=True, verbose=True, gap_target=None,
                 heuristics_emphasis=True, warm_phi_override=None):
    """End-to-end: build model, (optionally) warm-start from the greedy
    bit-flip local search, solve with SCIP under a wall-clock budget,
    return a result dict comparable to the notebook's direct/GRPO results."""
    warm_phi = None
    warm_minr = None
    if warm_phi_override is not None:
        warm_phi = warm_phi_override
        warm_minr = float(eval_minr_batch(warm_phi[None, :], env)[0])
        if verbose:
            print(f"  [warm start] externally supplied incumbent: min_rate={warm_minr:.4f}")
    elif warm_start:
        rng = np.random.default_rng(0)
        phi0 = rng.uniform(0, 2 * np.pi, size=env.L * env.M)
        phi0 = quantize_phase_hard(phi0, n_bits)
        warm_phi, warm_minr, n_evals, n_sweeps = greedy_bitflip_local_search(phi0, env, n_bits, max_sweeps=50)
        if verbose:
            print(f"  [warm start] greedy bit-flip local search: min_rate={warm_minr:.4f} "
                  f"({n_sweeps} sweeps, {n_evals} evals)")

    m, x, t, Full_re, Full_im, meta = build_minlp(env, n_bits, warm_phi=warm_phi)
    m.setParam("limits/time", float(time_limit))
    if gap_target is not None:
        m.setParam("limits/gap", float(gap_target))
    if heuristics_emphasis:
        try:
            from pyscipopt import SCIP_PARAMSETTING
            m.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
        except Exception:
            pass
    m.hideOutput(not verbose)

    if warm_phi is not None:
        try:
            m.presolve()
            _add_warm_start(m, x, meta["L"], meta["M"], meta["levels"], meta["step"], warm_phi)
        except Exception as e:
            if verbose:
                print(f"  [warm start] could not register incumbent with SCIP: {e}")

    t0 = time.time()
    m.optimize()
    wall = time.time() - t0

    status = m.getStatus()
    n_sols = m.getNSols()
    if n_sols == 0:
        return dict(status=status, wall_time=wall, warm_minr=warm_minr, warm_phi=warm_phi,
                     best_minr=warm_minr, best_phi=warm_phi, sinr_t=None, gap=None)

    phi_sol = extract_phi(m, x, meta)
    minr_sol = float(eval_minr_batch(phi_sol[None, :], env)[0])
    t_val = m.getVal(t)

    try:
        gap = m.getGap()
    except Exception:
        gap = None
    try:
        dual_bound = m.getDualbound()
    except Exception:
        dual_bound = None

    if verbose:
        print(f"  [SCIP] status={status}  wall={wall:.1f}s  primal SINR(t)={t_val:.4f}  "
              f"gap={gap}  dual_bound={dual_bound}")
        print(f"  [SCIP] decoded solution min_rate={minr_sol:.4f} bits/s/Hz "
              f"(vs warm-start {warm_minr:.4f})")

    best_minr, best_phi = minr_sol, phi_sol
    if warm_minr is not None and warm_minr > minr_sol:
        best_minr, best_phi = warm_minr, warm_phi

    return dict(status=status, wall_time=wall, warm_minr=warm_minr, warm_phi=warm_phi,
                 minlp_minr=minr_sol, minlp_phi=phi_sol, minlp_t=t_val,
                 best_minr=best_minr, best_phi=best_phi, gap=gap, dual_bound=dual_bound,
                 n_sols=n_sols)
