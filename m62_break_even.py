"""M62 LP in-window overhead break-even model (OFFLINE, never shipped).

M54 judged in-window LP RED, but its beta=0 kill test only zeroed t_solve:
tp = t_build + beta*t_solve + t_guard (m54_lp_rf_model.py:305) -- build+guard
stay at Python speed forever, so the "vectorized assembly / C++ build+guard"
axis was never killed. M62 answers: how much combined speedup
    tp = alpha*(t_build + t_guard) + beta*t_solve + eps
is required before SOME gate (band x cap x iters x abort/skip) becomes a
strict weak-win over ALL M in {6,8,11,14,20} x cores in {4,8,12,16} cells
(no cell worse, at least one cell clearly better).

Pure analysis: everything comes from m54_cache.pkl (per-pass t_build/t_solve/
t_guard/kept/cost_after). No new LP, no official eval, no shipped file touched.
m54_lp_rf_model is imported and reused verbatim (load_walls / cache paths /
cell constants); its module import loads the l3 dataset once (~1 min).

Assumption (also in M62_REPORT.md): a sped-up implementation reproduces the
scipy chain's keep decisions and post-LP quality exactly -- only time scales.
alpha scales build+guard together (both are Python overhead, same C++ batch);
beta scales solve (already HiGHS C++ under scipy, so beta<<1 is NOT free).
"""
import argparse
import json
import pickle

import numpy as np

import m54_lp_rf_model as m54                  # noqa: E402 (loads l3 dataset)

GAMMA, FLOOR, MS, CS = m54.GAMMA, m54.FLOOR, m54.MS, m54.CS
ALPHAS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0)
BETAS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.0)
CAPS_M54 = m54.CAPS                            # (None, 4, 2, 1, 0.5)
CAPS_EXT = (None, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1)
BAND_LOS = m54.BAND_LOS                        # (0, 60, 100, 110)
WIN_EPS = 1e-9                                 # "no cell worse" tolerance (%)
CELLS = [(M, cs) for cs in CS for M in MS]     # 20 cells


def load():
    c = pickle.load(open(m54.CACHE, "rb"))
    db = c["db"]
    assert len(db) == 100, f"dep cache incomplete ({len(db)}/100)"
    W, TOTW = m54.l3.W, m54.l3.TOTW
    anch_total = json.load(open(m54.ANCHOR_JSON))["total_score"]
    tot_s = sum(W[ci] * db[ci]["q_ship"] for ci in db) / TOTW
    assert abs(tot_s - anch_total) < 1e-9, "anchor total mismatch"
    wall = m54.load_walls()
    cis = sorted(db)
    T = {cs: np.array([db[ci]["t12"] * wall(ci, cs) / wall(ci, 12)
                       for ci in cis]) for cs in CS}
    return db, cis, np.array([W[ci] for ci in cis]), TOTW, T, tot_s


DB, CIS, WARR, TOTW, TARR, TOT_S = load()
NARR = np.array([DB[ci]["n"] for ci in CIS])
QSHIP = np.array([DB[ci]["q_ship"] for ci in CIS])


def lp_dt_q(r, alpha, beta, iters, cap, skip, eps):
    """(added seconds, resulting quality); semantics verbatim from
    m54.lp_time_q except tp = alpha*(build+guard) + beta*solve + eps."""
    chain = r["passes"][:iters]

    def tp(p):
        return alpha * (p["t_build"] + p["t_guard"]) + beta * p["t_solve"] + eps

    if skip and cap is not None and sum(tp(p) for p in chain) > cap:
        return 0.0, r["q_ship"]
    t, q = 0.0, r["q_ship"]
    for p in chain:
        tpp = tp(p)
        if cap is not None and t + tpp > cap:
            return cap, q                       # abort: budget burned
        t += tpp
        if p["kept"]:
            q = p["cost_after"]
        else:
            break
    return t, q


def arrays(alpha, beta, iters, cap, skip, eps):
    dt = np.empty(100)
    q = np.empty(100)
    for i, ci in enumerate(CIS):
        dt[i], q[i] = lp_dt_q(DB[ci], alpha, beta, iters, cap, skip, eps)
    return dt, q


def cell_total(dt, q, M, cs):
    rf = np.maximum(FLOOR, ((TARR[cs] + dt) / M) ** GAMMA)
    return float((WARR * q * rf).sum() / TOTW)


BASE = {(M, cs): cell_total(np.zeros(100), QSHIP, M, cs) for (M, cs) in CELLS}


def gate_deltas(dt, q, lo):
    m = NARR > lo
    dt_g = np.where(m, dt, 0.0)
    q_g = np.where(m, q, QSHIP)
    return [100.0 * (cell_total(dt_g, q_g, M, cs) - BASE[(M, cs)])
            / BASE[(M, cs)] for (M, cs) in CELLS]


def analyze(alpha, beta, eps, caps, win_min):
    """-> (wins, allres); each item = (tag, worst, best, ds)."""
    wins, allres = [], []
    for cap in caps:
        for it in (1, 2):
            for sk in ((False, True) if cap else (False,)):
                dt, q = arrays(alpha, beta, it, cap, sk, eps)
                for lo in BAND_LOS:
                    ds = gate_deltas(dt, q, lo)
                    tag = (f"n>{lo} cap={cap if cap else '-'} it{it}"
                           f"{' skip' if sk else ''}")
                    rec = (tag, max(ds), min(ds), ds)
                    allres.append(rec)
                    if max(ds) <= WIN_EPS and min(ds) <= -win_min:
                        wins.append(rec)
    return wins, allres


def best_of(wins):
    return min(wins, key=lambda r: r[2])        # most negative best cell


def sanity(win_min):
    print("== sanity ==")
    print(f"anchor total (RF=1.0) {TOT_S:.10f}  [assert passed]")
    # (b) alpha=1 beta=1, m54 gate set: must reproduce "no clear weak-win"
    w11, a11 = analyze(1.0, 1.0, 0.0, CAPS_M54, win_min)
    near = min(a11, key=lambda r: r[1])
    print(f"(b) a=1 b=1 (m54 caps): clear weak-wins = {len(w11)} "
          f"(expect 0); closest {near[0]} worst {near[1]:+.3f}% "
          f"best {near[2]:+.3f}%")
    assert not w11, "a=1,b=1 found a clear weak-win -- contradicts M54"
    # (c) alpha=1 beta=0 = M54 kill test: best gate worst cell ~ +0.48%
    _, a10 = analyze(1.0, 0.0, 0.0, CAPS_M54, win_min)
    near = min(a10, key=lambda r: r[1])
    print(f"(c) a=1 b=0 (M54 kill test): best gate {near[0]} worst "
          f"{near[1]:+.3f}% (M54 reported ~+0.48%)")
    # (d) alpha=0 beta=0: pure quality gain, all-it2 all cells ~ -0.87%
    dt, q = arrays(0.0, 0.0, 2, None, False, 0.0)
    ds = gate_deltas(dt, q, 0)
    tot_d = float((WARR * q).sum() / TOTW)
    print(f"(d) a=0 b=0 gate='n>0 cap=- it2': cells [{min(ds):+.3f}%, "
          f"{max(ds):+.3f}%]; RF=1 quality {TOT_S:.6f}->{tot_d:.6f} "
          f"({100 * (TOT_S - tot_d) / TOT_S:+.4f}%, M54 reported +0.891%)")
    assert max(ds) < 0, "a=0,b=0 must win every cell"


def grid(eps, win_min):
    print(f"\n== WIN grid (eps={eps * 1000:g}ms, win_min={win_min}%): "
          f"best-cell% if a clear weak-win gate exists, else '.' with "
          f"closest worst-cell% ==")
    hdr = "alpha\\beta"
    print(f"{hdr:>10} " + "".join(f"{b:>12g}" for b in BETAS))
    frontier = {}                               # beta -> max winning alpha
    cells_txt = {}
    for a in ALPHAS:
        row = f"{a:>10g} "
        for b in BETAS:
            wins, allres = analyze(a, b, eps, CAPS_EXT, win_min)
            if wins:
                bst = best_of(wins)
                row += f"  W {bst[2]:+7.3f}"
                cells_txt[(a, b)] = (len(wins), bst)
                if b not in frontier or a > frontier[b]:
                    frontier[b] = a
            else:
                near = min(allres, key=lambda r: r[1])
                row += f"  . {near[1]:+7.3f}"
        print(row)
    return frontier, cells_txt


def detail(eps, win_min):
    """Scrutiny of the surprise finding: wins already exist at a=1,b=1 with
    caps < 0.5s (a region M54's CAPS grid never scanned). Sections: full win
    list (are any abort-only i.e. deployable-without-predictor?), gate
    composition, and timing robustness (machine speed / per-case jitter --
    the skip criterion uses TRUE chain time = oracle predictor)."""
    print("\n== detail: ALL clear weak-win gates at alpha=1 beta=1 ==")
    wins, _ = analyze(1.0, 1.0, eps, CAPS_EXT, win_min)
    for tag, w, b, _ds in sorted(wins, key=lambda r: r[1]):
        print(f"  {tag:<30} worst {w:+.3f}% best {b:+.3f}%")
    nsk = [r for r in wins if "skip" not in r[0]]
    print(f"  abort-only (predictor-free) wins: {len(nsk)}")

    for (cap, it) in ((0.25, 1), (0.1, 1)):
        dt, q = arrays(1.0, 1.0, it, cap, True, eps)
        ran, kept = dt > 0, q < QSHIP - 1e-15
        wq = 100 * float((WARR * (QSHIP - q)).sum() / (WARR * QSHIP).sum())
        print(f"\ngate n>0 cap={cap} it{it} skip @a=1,b=1: runs "
              f"{int(ran.sum())} cases, {int(kept.sum())} improve; RF=1 "
              f"weighted quality gain {wq:+.4f}%  (top contributors:)")
        idx = np.where(kept)[0]
        idx = idx[np.argsort(-WARR[idx] * (QSHIP[idx] - q[idx]))][:10]
        for i in idx:
            print(f"  case {CIS[i]:3d} n={NARR[i]:3d} "
                  f"{QSHIP[i]:.4f}->{q[i]:.4f} dt={dt[i]:.3f}s")

    print("\n== machine-speed robustness: tp scaled by s (grader speed "
          "unknown), gate FIXED; cell shows worst-cell%; W=clear weak-win, "
          "w=weak only ==")
    ss = (0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 3.0)
    print(f"{'gate (n>0, skip)':<20}" + "".join(f"{s:>10g}" for s in ss))
    for cap, it in ((0.5, 1), (0.25, 1), (0.25, 2), (0.1, 1), (0.1, 2)):
        row = f"cap={cap} it{it}".ljust(20)
        for s in ss:
            dt, q = arrays(s, s, it, cap, True, eps)
            ds = gate_deltas(dt, q, 0)
            mk = ("W" if max(ds) <= WIN_EPS and min(ds) <= -win_min
                  else ("w" if max(ds) <= WIN_EPS else "."))
            row += f" {mk}{max(ds):>+8.3f}"
        print(row)
    row = "#win gates (any)".ljust(20)
    for s in ss:
        ws, _ = analyze(s, s, eps, CAPS_EXT, win_min)
        row += f" {len(ws):>9d}"
    print(row)

    print("\n== per-case timing jitter (each case's pass times x U[0.7,1.3], "
          "40 draws): does the FIXED gate stay a clear weak-win? ==")
    rng = np.random.default_rng(62)
    for cap, it in ((0.5, 1), (0.25, 1), (0.1, 1)):
        wc, worsts = 0, []
        for _ in range(40):
            f = rng.uniform(0.7, 1.3, 100)
            dt, q = np.empty(100), np.empty(100)
            for i, ci in enumerate(CIS):
                dt[i], q[i] = lp_dt_q(DB[ci], f[i], f[i], it, cap, True, eps)
            ds = gate_deltas(dt, q, 0)
            worsts.append(max(ds))
            wc += (max(ds) <= WIN_EPS and min(ds) <= -win_min)
        print(f"gate n>0 cap={cap} it{it} skip: clear weak-win {wc}/40 "
              f"draws; worst-cell range [{min(worsts):+.3f}%, "
              f"{max(worsts):+.3f}%]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.0,
                    help="additive per-pass floor, seconds")
    ap.add_argument("--win-min", type=float, default=0.01,
                    help="'clearly better' threshold, %% (default 0.01)")
    ap.add_argument("--no-sens", action="store_true")
    args = ap.parse_args()

    sanity(args.win_min)
    frontier, cells_txt = grid(args.eps, args.win_min)

    print("\n== frontier: per beta, the LARGEST alpha with a clear "
          "weak-win gate (= minimum required build+guard speedup) ==")
    for b in BETAS:
        a = frontier.get(b)
        if a is None:
            print(f"beta={b:g}: no winning alpha in grid (even alpha=0)")
        else:
            nw, bst = cells_txt[(a, b)]
            spd = "inf" if a == 0 else f"{1 / a:g}x"
            print(f"beta={b:g}: alpha*={a:g} ({spd} speedup)  wins={nw}  "
                  f"best gate {bst[0]} worst {bst[1]:+.3f}% "
                  f"best {bst[2]:+.3f}%")

    # worst-cell matrices at the deployable-region frontier points
    for b in (1.0, 0.5, 0.2):
        a = frontier.get(b)
        if a is None:
            continue
        _, bst = cells_txt[(a, b)]
        print(f"\n== cell matrix, alpha={a:g} beta={b:g}, gate {bst[0]} ==")
        print(f"{'cores':>5} " + "".join(f"M={M:>3} {'':>4}" for M in MS))
        for cs in CS:
            row = f"{cs:>5} "
            for M in MS:
                d = bst[3][CELLS.index((M, cs))]
                row += f"{d:>+8.3f} "
            print(row)

    detail(args.eps, args.win_min)

    # per-case dt budget: floor headroom vs LP chain time
    kfl = FLOOR ** (1.0 / GAMMA)                # 0.30459: t/M below this=floor
    print(f"\n== per-case dt budget (top 16 by weight; kfl={kfl:.4f}) ==")
    print("dt_free(M,c) = max(0, kfl*M - T[c]): free seconds under the RF "
          "floor;\ndt_tax = T12*((Qs/Qlp)^(1/0.3)-1): above-floor break-even "
          "budget @12c\n(chain = full it2 tLP at a=1,b=1; a,b scale it "
          "linearly)")
    print(f"{'ci':>3} {'n':>4} {'Qs':>7} {'Qlp':>7} {'dq%':>7} {'T12':>6} "
          f"{'T4':>6} {'chain':>6} {'free@6,12':>9} {'free@8,12':>9} "
          f"{'free@11,12':>10} {'free@6,4':>8} {'dt_tax':>7}")
    order = np.argsort(-WARR)
    for i in order[:16]:
        ci = CIS[i]
        r = DB[ci]
        dtf, qf = lp_dt_q(r, 1.0, 1.0, 2, None, False, 0.0)
        dq = 100 * (r["q_ship"] - qf) / r["q_ship"]
        tax = (TARR[12][i] * ((r["q_ship"] / qf) ** (1 / GAMMA) - 1)
               if qf < r["q_ship"] else 0.0)
        f = {(M, cs): max(0.0, kfl * M - TARR[cs][i])
             for (M, cs) in ((6, 12), (8, 12), (11, 12), (6, 4))}
        print(f"{ci:>3} {r['n']:>4} {r['q_ship']:>7.4f} {qf:>7.4f} "
              f"{dq:>+7.3f} {TARR[12][i]:>6.2f} {TARR[4][i]:>6.2f} "
              f"{dtf:>6.2f} {f[(6, 12)]:>9.2f} {f[(8, 12)]:>9.2f} "
              f"{f[(11, 12)]:>10.2f} {f[(6, 4)]:>8.2f} {tax:>7.3f}")

    # safe-set reference bound (per-case fixed subset, NOT deployable: needs
    # an input-side per-case predictor, which M56 proved unrealizable)
    print("\n== safe-set reference (cases whose full it2 chain is net<=0 in "
          "ALL 20 cells; upper bound for per-case-predicted gates) ==")
    hdr = "alpha\\beta"
    print(f"{hdr:>10} " + "".join(f"{b:>14g}" for b in BETAS))
    for a in ALPHAS:
        row = f"{a:>10g} "
        for b in BETAS:
            dt, q = arrays(a, b, 2, None, False, args.eps)
            safe = np.ones(100, dtype=bool)
            clear = np.zeros(100, dtype=bool)
            for (M, cs) in CELLS:
                rf1 = np.maximum(FLOOR, ((TARR[cs] + dt) / M) ** GAMMA)
                rf0 = np.maximum(FLOOR, (TARR[cs] / M) ** GAMMA)
                net = q * rf1 - QSHIP * rf0
                safe &= net <= 1e-12
                clear |= net < -1e-12
            safe &= clear
            best = 0.0
            for (M, cs) in CELLS:
                dt_g = np.where(safe, dt, 0.0)
                q_g = np.where(safe, q, QSHIP)
                d = 100 * (cell_total(dt_g, q_g, M, cs) - BASE[(M, cs)]) \
                    / BASE[(M, cs)]
                best = min(best, d)
            row += f"  {int(safe.sum()):>3}:{best:+7.3f}"
        print(row)

    if not args.no_sens:
        for eps in (0.002, 0.005):
            print(f"\n== eps sensitivity: frontier at eps={eps * 1000:g}ms "
                  f"per pass ==")
            fr, ct = grid(eps, args.win_min)
            for b in BETAS:
                a = fr.get(b)
                print(f"  beta={b:g}: alpha*="
                      f"{a if a is not None else 'none'}")

    print("\n== verdict inputs ==")
    a1, a05 = frontier.get(1.0), frontier.get(0.5)
    print(f"alpha* at beta=1.0 (solver untouched): "
          f"{a1 if a1 is not None else 'none'}")
    print(f"alpha* at beta=0.5 (credible 2x solver): "
          f"{a05 if a05 is not None else 'none'}")
    anyw = [(b, frontier[b]) for b in BETAS if b in frontier]
    print(f"full frontier (beta, alpha*): {anyw if anyw else 'EMPTY'}")
    print("thresholds: alpha*>=0.2 GREEN / <0.05 RED / else YELLOW "
          "(judged on the credible beta in {1,0.5} columns; scipy linprog "
          "is already HiGHS C++, beta<<1 is not obtainable by porting)")


if __name__ == "__main__":
    main()
