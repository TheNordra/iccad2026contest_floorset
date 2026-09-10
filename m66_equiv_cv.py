"""M66: equivalence-class patch of the M56 per-case CV (cache-only, OFFLINE).

M56's Phase B had a methodological asymmetry: the train-LOO J* search and the
kNN keep-set union both used the SINGLETON winner index, while the OOS gate
accepts any cost-equal (rel 1e-9) substitute. M66 replaces the single label
with the equivalence class
  E(case, regime) = {k in chain_pool(case, cores) :
                     |cost(ci,k) - cost(ci,winner)| <= 1e-9*max(1,|cost_w|)}
in BOTH the LOO acceptance test and the keep-set union, then re-runs the same
5-fold CV. Everything else (folds, features, kNN, variants, fail-open, OOS
cost gate) is unchanged. Cache-only: positions come from audit_cache via the
m56 module; official costs persist into m56_cache.pkl's cost dict (resume-
safe, no new cache file). m56_percase_oracle.py itself is NOT modified.

Pre-registered verdict rules (user-set):
  - no variant reaches 0 OOS strict breaks           -> RED (ledger stands)
  - 0 breaks but realizable max-cell gain < 0.05%    -> RED (pool inflation)
  - 0 breaks and gain >= 0.05%                       -> GREEN, report-only

Run:  python -u m66_equiv_cv.py selfcheck
      python -u m66_equiv_cv.py cv [--seed N]
      python -u m66_equiv_cv.py diag      (per-break: winner class absent vs
                                           hmin-flip; costs must be cached)
Importing m56_percase_oracle re-runs its original cv mode (deterministic,
cache-hot) and re-dumps results_M56_percase.json bit-identically; that rerun
doubles as the gate0 reproduction anchor. Artifacts: results_M66_equiv.json.
Never shipped; touches no shipped file.
"""
import argparse
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["selfcheck", "cv", "diag"])
ap.add_argument("--seed", type=int, default=None,
                help="passed through to m56's fold shuffle (default None = "
                     "original folds; gate0 stored-value asserts need None)")
A = ap.parse_args()

# m56_percase_oracle is a top-level script: it parses sys.argv and executes
# its whole mode at import time. Point it at its own cv mode so the original
# CV re-runs (costs cached -> minutes) and all cv-branch objects (FOLDS, D,
# nearest, model_pool, Jstar, results, ...) become module attributes.
sys.argv = ["m56_percase_oracle.py", "cv"] + \
    ([] if A.seed is None else ["--seed", str(A.seed)])
print(f"=== M66 [{A.mode}] importing m56_percase_oracle "
      f"(re-runs original cv mode) ===", flush=True)
import m56_percase_oracle as m          # noqa: E402
import optimizer_constructive as oc     # noqa: E402

TOL = 1e-9        # identical to the m56 OOS gate tolerance
GREEN_BAR = 0.05  # % max-cell realizable gain required for GREEN (house bar)
OUTJSON = _DIR / "results_M66_equiv.json"

# -- gate0: the import above re-ran the original CV; pin its numbers ----------
print("\n" + "=" * 72)
print("GATE0: original-CV reproduction anchors (from the import-time rerun)")
print("=" * 72)
assert m.args.mode == "cv"
n_none = sum(1 for v_ in m.Jstar.values() if v_ is None)
print(f"original Jstar: {n_none}/{len(m.Jstar)} None")
if A.seed is None:
    assert n_none == len(m.Jstar) == 20, "Jstar not all-None (M56 anchor)"
    for v_ in m.VARIANTS:
        assert len(m.results[v_]["recs"]) == 200
        assert m.results[v_]["breaks"] == 30, \
            f"breaks {m.results[v_]['breaks']} != stored 30 ({v_})"
        assert abs(m.results[v_]["mean_pool"] - 22.86) < 0.01
    ref = json.load(open(_DIR / "results_M56_percase.json"))
    wh, wl = ref["oracle"]["winner_hi"], ref["oracle"]["winner_lo"]
    for ci in range(100):
        assert m.winner(ci, 12) == wh[str(ci)], f"winner_hi drift ci={ci}"
        assert m.winner(ci, 4) == wl[str(ci)], f"winner_lo drift ci={ci}"
    print("gate0 OK: Jstar 20/20 None; breaks 30+30; mean_pool 22.86; "
          "winner_hi/lo == stored oracle JSON (100/100 x2)")
else:
    print(f"gate0 (seed={A.seed}): stored-value asserts skipped "
          f"(non-default folds)")


# -- equivalence classes ------------------------------------------------------
_EQ = {}


def equiv(ci, cores):
    """E over the winner's own domain, chain_pool(ci, cores), per regime."""
    reg = "lo" if cores <= oc._M45_CORES_MAX else "hi"
    if (ci, reg) not in _EQ:
        w = m.winner(ci, cores)
        qw = m.cost(ci, w)
        E = frozenset(k for k in m.chain_pool(ci, cores)
                      if abs(m.cost(ci, k) - qw) <= TOL * max(1.0, abs(qw)))
        assert w in E and len(E) >= 1
        _EQ[(ci, reg)] = E
    return _EQ[(ci, reg)]


def eq_single(ci, cores):
    """Selfcheck stand-in: singleton winner = the original M56 semantics."""
    return frozenset((m.winner(ci, cores),))


def build_equiv():
    print("\nbuilding E(case, regime) over both chain pools "
          "(official costs persist into m56_cache.pkl) ...", flush=True)
    n0 = len(m._cost)
    for i, c_ in enumerate(m.CASES):
        for cores_ in (12, 4):   # lo pool is a subset of hi -> no extra evals
            equiv(c_["idx"], cores_)
        if (i + 1) % 10 == 0:
            m.flush_cost()
            print(f"  {i + 1}/100 cases  (cost cache {len(m._cost)} entries)",
                  flush=True)
    m.flush_cost()
    print(f"E done: cost cache {n0} -> {len(m._cost)} entries")


# -- E-substituted mirrors of the m56 cv-branch functions ---------------------
def keep_set_e(ci, train, J, cores, variant, eq):
    ks = set()
    for t in m.nearest(ci, train, J):
        ks |= eq(t, cores)
    if variant == "knn+band":
        bt = m.band_tag(m.CASES[ci]["n"])
        for t in train:
            if m.band_tag(m.CASES[t]["n"]) == bt:
                ks |= eq(t, cores)
    return ks


def model_pool_e(ci, train, J, cores, variant, eq):
    ks = keep_set_e(ci, train, J, cores, variant, eq)
    pool = [k for k in m.chain_pool(ci, cores) if k in ks]
    return pool or m.chain_pool(ci, cores)   # fail-open (mirrors m56)


def loo_J_e(train, cores, variant, eq):
    """Smallest J with 100% train-LOO preservation, acceptance = membership
    in E (cost-equality) instead of index identity."""
    for J in range(1, len(train) + 1):
        ok = True
        for j in train:
            others = [t for t in train if t != j]
            sel = m.select(j, model_pool_e(j, others, J, cores, variant, eq))
            if sel not in eq(j, cores):
                ok = False
                break
        if ok:
            return J
    return None


def run_oos(Js, eq, check_super=False):
    """OOS pass over all folds x regimes; the preserved check is byte-for-byte
    the m56 one (index match short-circuit, else cost equality)."""
    results = {}
    for v in m.VARIANTS:
        recs = []
        for f, (tr, te) in enumerate(m.FOLDS):
            for reg, cores_ in m.REGIMES:
                J = Js[(f, reg, v)] or len(tr)
                for ci in te:
                    if check_super:
                        assert keep_set_e(ci, tr, J, cores_, v, eq) >= \
                            m.keep_set(ci, tr, J, cores_, v), \
                            f"keep-set not a superset ci={ci}"
                    pool = model_pool_e(ci, tr, J, cores_, v, eq)
                    sel = m.select(ci, pool)
                    w = m.winner(ci, cores_)
                    if sel == w:
                        pres, dq = True, 0.0
                    else:
                        qs, qn = m.cost(ci, w), m.cost(ci, sel)
                        pres = abs(qn - qs) <= TOL * max(1.0, abs(qs))
                        dq = (qn - qs) / qs * 100 if qs > 0 else 0.0
                    recs.append(dict(fold=f, ci=ci, reg=reg,
                                     n=m.CASES[ci]["n"], poolsz=len(pool),
                                     preserved=pres, dq=dq))
        nb = sum(1 for r in recs if not r["preserved"])
        results[v] = dict(recs=recs, breaks=nb,
                          mean_pool=sum(r["poolsz"] for r in recs) / len(recs))
    return results


# =============================================================================
if A.mode == "diag":
    # For each residual OOS break: is the winner's WHOLE equivalence class
    # absent from the kept pool (LABEL-ABSENT), or present but out-selected
    # via subset-pool hmin coupling (HMIN-FLIP)? Uses J = len(tr), the actual
    # OOS fallback when J* is None; at that J the knn+band variant collapses
    # onto knn (nearest already covers the whole train set).
    print("\n" + "=" * 72)
    print("DIAG: residual-break decomposition (variant knn, J = |train|)")
    print("=" * 72)
    nla = nhf = 0
    for f, (tr, te) in enumerate(m.FOLDS):
        for reg, cores_ in m.REGIMES:
            for ci in te:
                pool = model_pool_e(ci, tr, len(tr), cores_, "knn", equiv)
                sel = m.select(ci, pool)
                w = m.winner(ci, cores_)
                qs, qn = m.cost(ci, w), m.cost(ci, sel)
                if sel != w and abs(qn - qs) > TOL * max(1.0, abs(qs)):
                    E = equiv(ci, cores_)
                    inp = set(E) & set(pool)
                    kind = "HMIN-FLIP" if inp else "LABEL-ABSENT"
                    nhf += bool(inp)
                    nla += not inp
                    print(f"  f{f} case {ci:>3} @{reg}: |E|={len(E)} "
                          f"E-in-pool={len(inp)} {kind}  "
                          f"dq={(qn - qs) / qs * 100:+.3f}%")
    m.flush_cost()
    print(f"\ntotal: {nla} LABEL-ABSENT, {nhf} HMIN-FLIP")
    sys.exit(0)

# =============================================================================
if A.mode == "selfcheck":
    print("\n" + "=" * 72)
    print("SELFCHECK: singleton-E through the new plumbing must reproduce M56")
    print("=" * 72)
    tr0 = m.FOLDS[0][0]
    for reg, cores_ in m.REGIMES:
        for v in m.VARIANTS:
            jn = loo_J_e(tr0, cores_, v, eq_single)
            jo = m.Jstar[(0, reg, v)]
            assert jn == jo, f"LOO parity fail f0 {reg}/{v}: {jn} != {jo}"
            print(f"  loo_J_e singleton f0 {reg}/{v} = {jn}  == original OK")
    res_s = run_oos(m.Jstar, eq_single)
    for v in m.VARIANTS:
        ro, rn = m.results[v]["recs"], res_s[v]["recs"]
        assert len(ro) == len(rn) == 200
        for a_, b_ in zip(ro, rn):
            assert a_ == b_, f"OOS rec mismatch ({v}): {a_} vs {b_}"
        print(f"  OOS parity {v}: 200/200 recs identical "
              f"(breaks {res_s[v]['breaks']} == {m.results[v]['breaks']})")
    m.flush_cost()
    print("\nSELFCHECK PASS")
    sys.exit(0)

# =============================================================================
print("\n" + "=" * 72)
print("M66 MAIN: equivalence-class CV (E replaces the singleton winner label)")
print("=" * 72)
build_equiv()

print("\n|E| distribution:")
esz = {}
for reg, cores_ in m.REGIMES:
    sizes = [len(equiv(ci, cores_)) for ci in range(100)]
    esz[reg] = sizes
    print(f"  {reg}: mean {sum(sizes) / 100:.2f}  median {sorted(sizes)[50]}  "
          f"max {max(sizes)}  |E|>1 in {sum(1 for s in sizes if s > 1)}/100")
    for lo_, hi_ in m.BANDS:
        bs = [len(equiv(c_["idx"], cores_))
              for c_ in m.CASES if lo_ < c_["n"] <= hi_]
        if bs:
            hs = "inf" if hi_ >= 10 ** 9 else str(hi_)
            print(f"    ({lo_},{hs}]: mean |E| {sum(bs) / len(bs):.2f}  "
                  f"max {max(bs)}  ({len(bs)} cases)")
for sz, ci, reg in sorted(((len(E), ci, reg) for (ci, reg), E in _EQ.items()),
                          reverse=True)[:5]:
    mem = " ".join(f"#{k}" for k in sorted(_EQ[(ci, reg)]))
    print(f"  largest: case {ci} @{reg} |E|={sz}: {mem}")

Jn = {}
print("\nper-fold J* with E-acceptance (original: all None):")
for f, (tr, te) in enumerate(m.FOLDS):
    for reg, cores_ in m.REGIMES:
        for v in m.VARIANTS:
            Jn[(f, reg, v)] = loo_J_e(tr, cores_, v, equiv)
    print(f"  fold {f}: " + "  ".join(
        f"{reg}/{v}={Jn[(f, reg, v)]}"
        for reg, _ in m.REGIMES for v in m.VARIANTS), flush=True)

res = run_oos(Jn, equiv, check_super=True)
mcp = sum(len(m.chain_pool(ci, c2)) for _, c2 in m.REGIMES
          for ci in range(100)) / 200
orig_bk = {v: {(r["fold"], r["ci"], r["reg"])
               for r in m.results[v]["recs"] if not r["preserved"]}
           for v in m.VARIANTS}
for v in m.VARIANTS:
    nb, mps = res[v]["breaks"], res[v]["mean_pool"]
    nbk = {(r["fold"], r["ci"], r["reg"])
           for r in res[v]["recs"] if not r["preserved"]}
    print(f"\nvariant {v}: OOS strict-breaks {nb}/200 (mean kept pool "
          f"{mps:.1f})   [original: 30/200 @ 22.86; mean chain pool {mcp:.1f}]")
    print(f"  break overlap vs original: {len(nbk & orig_bk[v])} persist, "
          f"{len(nbk - orig_bk[v])} new, {len(orig_bk[v] - nbk)} healed")
    for r in sorted(res[v]["recs"], key=lambda r: -abs(r["dq"])):
        if not r["preserved"]:
            print(f"    fold {r['fold']} case {r['ci']:>3} n={r['n']:>3} "
                  f"@{r['reg']}  pool {r['poolsz']}  dq {r['dq']:+.3f}%")
m.flush_cost()

out = dict(seed=A.seed, tol=TOL, green_bar=GREEN_BAR,
           mean_chain_pool=mcp,
           esize={reg: esz[reg] for reg in esz},
           jstar_new={f"f{f}_{reg}_{v}": Jn[(f, reg, v)]
                      for f in range(m.NFOLD)
                      for reg, _ in m.REGIMES for v in m.VARIANTS},
           breaks={v: res[v]["breaks"] for v in m.VARIANTS},
           mean_pool={v: res[v]["mean_pool"] for v in m.VARIANTS},
           breaks_orig={v: m.results[v]["breaks"] for v in m.VARIANTS},
           oos={v: res[v]["recs"] for v in m.VARIANTS})

green = [v for v in m.VARIANTS if res[v]["breaks"] == 0]
if not green:
    out["verdict"] = "RED (OOS preservation < 100% even with equivalence classes)"
    print(f"\nVERDICT: {out['verdict']}")
else:
    v = min(green, key=lambda v: res[v]["mean_pool"])
    print(f"\nvariant '{v}' reaches 0 breaks -> realizable OOS projection:")

    def cvfn_e(ci, cores):
        f = m.fold_of[ci]
        reg = "lo" if cores <= oc._M45_CORES_MAX else "hi"
        tr = m.FOLDS[f][0]
        J = Jn[(f, reg, v)] or len(tr)
        return model_pool_e(ci, tr, J, cores, v, equiv)

    print(f"{'cores':>5} {'M':>3} {'shipped':>9} {'cvOOS':>9} {'gain%':>8}")
    mat, g_max = {}, -1e9
    for cores_ in m.CORES_GRID:
        for M in m.M_GRID:
            a = m.rf_total_pool(M, cores_, m.shipped_fn)
            b = m.rf_total_pool(M, cores_, cvfn_e)
            g = 100 * (a - b) / a
            mat[(cores_, M)] = (a, b, g)
            g_max = max(g_max, g)
            print(f"{cores_:>5} {M:>3} {a:>9.4f} {b:>9.4f} {g:>+7.3f}%")
    out["realizable"] = {f"c{c2}_M{M}": dict(shipped=x[0], cv=x[1], gain_pct=x[2])
                         for (c2, M), x in mat.items()}
    out["g_max"] = g_max
    if g_max < GREEN_BAR:
        out["verdict"] = (f"RED (pool inflation: 0 breaks but max-cell gain "
                          f"{g_max:+.3f}% < {GREEN_BAR}% bar)")
    else:
        out["verdict"] = f"GREEN ({v}, g_max {g_max:+.3f}%) - report-only, never ship"
    print(f"\nVERDICT: {out['verdict']}")

json.dump(out, open(OUTJSON, "w"), indent=1)
m.flush_cost()
print(f"\ndumped {OUTJSON.name}")
print("M66 DONE")
