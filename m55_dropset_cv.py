"""M55: 5-fold out-of-sample validation of the M42/M45 drop sets (OFFLINE).

Question: _BIG_REDUNDANT_IDX (M42, n>100) and _M45_BAND_DROP / _M45_LOWCORE_DROP
(M45 tiers) were derived IN-SAMPLE on the local 100 cases ("profile wins no case
in band"). On a hidden test set the winner structure may differ: a profile that
never wins locally may be the needed winner elsewhere. This tool re-derives the
drop sets with the SAME derivation code (copied from rf_score_model.py,
parameterized by a train-case subset) on 5 band-stratified folds of 80 cases,
then on the 20 held-out cases:
  - checks strict selection preservation (cost equality, rel 1e-9) and the
    weaker median-independent WIN inequality (Qc/Qb <= (tB/tC)^0.3);
  - quantifies quality regret (dq, dq%, weighted contribution);
  - projects the real RF-aware total over an M x cores grid for
    base(M41) / shipped(in-sample) / CV-composite(fully OOS) chains;
  - sweeps the M42 threshold T in {100,105,110} for the FREE_N recommendation.

Known limitation (per CLAUDE.md): audit_cache.pkl is the n>100 K=12
counterfactual (dt conservative); conclusions target selection/redundancy
structure, orthogonal to the M49/M50 REFINE axis.

Run:  python -u m55_dropset_cv.py [--seed N]
Artifacts: m55_cv_cache.pkl (PM + lazy-cost persistence), results_M55_cv.json.
Never shipped; touches no shipped file.
"""
import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc

RH = 1.4
CORES = 12
GAMMA = 0.3
FLOOR = 0.7
NFOLD = 5
CACHE = _DIR / "audit_cache.pkl"
CVCACHE = _DIR / "m55_cv_cache.pkl"
OUTJSON = _DIR / "results_M55_cv.json"

# Must match profile_audit.py's cache key exactly (its PROFILES = live + OM16).
OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
# M72: exclude the appended, gated-off _M55_EXTRA tier (see profile_audit.py).
_SHIPPED = list(oc._PROFILES[:getattr(oc, "_M55_BASE_LEN", len(oc._PROFILES))])
PROFILES = _SHIPPED + [OM16]
N_LIVE = len(_SHIPPED)
FPR = repr(PROFILES)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=None,
                help="shuffle within bands before round-robin fold assignment "
                     "(default: deterministic sorted order)")
args = ap.parse_args()

# ── dataset prep (mirrors rf_score_model.py / profile_audit.py) ──────────────
print("loading dataset ...", flush=True)
ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
CASES = []
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    sumA = sum(max(0.0, float(at[i])) for i in range(n))
    CASES.append(dict(idx=idx, n=n, A_hat=1.035 * max(sumA, 1e-9),
                      w=math.exp(n / 12.0), base=base, tp=tp,
                      at=at, b2b=b2b, p2b=p2b, pins=pins, cons=cons))
totW = sum(c_["w"] for c_ in CASES)

# ── load audit cache (signature gate) ────────────────────────────────────────
if not CACHE.exists():
    sys.exit("audit_cache.pkl missing -> run profile_audit.py first")
c = pickle.load(open(CACHE, "rb"))
if c.get("profiles") != FPR:
    sys.exit("cache profile signature != current pool -> re-run profile_audit.py")
data = c["data"]
missing = [(ci, k) for ci in range(100) for k in range(len(PROFILES)) if (ci, k) not in data]
if missing:
    sys.exit(f"cache incomplete ({len(missing)} combos missing)")
print(f"audit cache OK: {len(data)} combos, {len(PROFILES)} profiles "
      f"({N_LIVE} live + OM16 stand-by)", flush=True)

# ── PM + lazy true cost, persisted in m55_cv_cache.pkl ───────────────────────
cvc = {}
if CVCACHE.exists():
    try:
        cvc = pickle.load(open(CVCACHE, "rb"))
        if cvc.get("profiles") != FPR:
            print("m55 cache signature stale -> recomputing"); cvc = {}
    except Exception:
        cvc = {}
PM = cvc.get("pm")
if PM is None:
    print("computing proxy metrics for all combos (one-time, ~2-3 min) ...", flush=True)
    PM = {}
    for c_ in CASES:
        ci = c_["idx"]
        for k in range(len(PROFILES)):
            ps, _ = data[(ci, k)]
            m = oc._proxy_metrics(ps, c_["at"], c_["b2b"], c_["p2b"],
                                  c_["pins"], c_["cons"], c_["n"])
            PM[(ci, k)] = (m["area"], m["hpwl"], m["vrel"])
    cvc = {"profiles": FPR, "pm": PM, "cost": cvc.get("cost", {})}
    pickle.dump(cvc, open(CVCACHE, "wb"))
    print("proxy metrics cached", flush=True)
_cost = cvc.setdefault("cost", {})
_cost_dirty = [False]


def cost(ci, k):
    if (ci, k) not in _cost:
        c_ = CASES[ci]; ps, _ = data[(ci, k)]
        tc = evaluate_solution({'positions': ps, 'runtime': 1.0}, c_["base"],
                               c_["cons"][:c_["n"]], c_["b2b"], c_["p2b"], c_["pins"],
                               c_["at"][:c_["n"]], target_positions=c_["tp"][:c_["n"]],
                               median_runtime=1.0)
        _cost[(ci, k)] = tc.cost
        _cost_dirty[0] = True
    return _cost[(ci, k)]


def flush_cost():
    if _cost_dirty[0]:
        pickle.dump(cvc, open(CVCACHE, "wb"))
        _cost_dirty[0] = False


def select(ci, pool):
    """Deployed _RH=1.4 proxy selector (wrapper parity)."""
    hmin = min(PM[(ci, k)][1] for k in pool) or 1.0
    A = CASES[ci]["A_hat"]
    return min(pool, key=lambda k: (PM[(ci, k)][0] / A + RH * PM[(ci, k)][1] / hmin)
               * math.exp(2 * PM[(ci, k)][2]))


def wall(ci, pool, cores=CORES):
    ts = [data[(ci, k)][1] for k in pool]
    return max(max(ts), sum(ts) / cores)


live = list(range(N_LIVE))
SWAP = [k for k in live if "ICCAD_ORDER_SWAP" in PROFILES[k] or "ICCAD_ORDER_MOVE" in PROFILES[k]]
SWAPset = set(SWAP)
BUILD = [k for k in live if k not in SWAPset]
BIGSET = set(oc._BIG_REDUNDANT_IDX)
T3_SHIP = set(oc._M45_BAND_DROP[0][2])
T4_SHIP = {(lo, hi): set(d) for lo, hi, d in oc._M45_LOWCORE_DROP}
T4_BANDS = [(lo, hi) for lo, hi, _ in oc._M45_LOWCORE_DROP]


def post_swap(ci):
    """M41 shipped pool (drop swap for all n)."""
    return [k for k in live if k not in SWAPset]


def shipped_pool(ci):
    """M41+M42 shipped chain (the M45 tier baseline)."""
    n = CASES[ci]["n"]
    return [k for k in live if k not in SWAPset and not (n > 100 and k in BIGSET)]


# ── M42 derivation (verbatim logic from rf_score_model.py, train-subset aware) ─
def redundant_at(T, cs):
    """BUILD profiles that win NO post-swap case with n>T among cs."""
    winners = set(select(c_["idx"], post_swap(c_["idx"])) for c_ in cs if c_["n"] > T)
    return [k for k in BUILD if k not in winners], winners


def m42cap(T, R):
    Rs = set(R)
    def f(ci):
        n = CASES[ci]["n"]
        return [k for k in live if k not in SWAPset and not (n > T and k in Rs)]
    return f


def m42_rows(R, T, cs):
    """(ci, n, qb, qc, tb, tc, ratio, qrel, win, strictEq) over cs with n>T,
    capped-vs-post_swap. win = M42 WIN inequality; strictEq = M45 strict gate."""
    capfn = m42cap(T, R); rows = []
    for c_ in cs:
        if c_["n"] <= T:
            continue
        ci = c_["idx"]
        pb, pc = post_swap(ci), capfn(ci)
        qb, qc = cost(ci, select(ci, pb)), cost(ci, select(ci, pc))
        tb, tc = wall(ci, pb), wall(ci, pc)
        ratio = (tb / tc) ** GAMMA if tc > 0 else 1.0
        qrel = qc / qb if qb > 0 else 1.0
        win = qrel <= ratio + 1e-12
        seq = abs(qc - qb) <= 1e-9 * max(1.0, abs(qb))
        rows.append((ci, c_["n"], qb, qc, tb, tc, ratio, qrel, win, seq))
    return rows


def refine_m42(T, cs, gate):
    """Greedy keep-back until all train cases with n>T pass the gate.
    gate='win' reproduces rf_score_model refine(); gate='strict' uses the M45
    selection-preserving criterion instead."""
    ok_of = (lambda r: r[8]) if gate == "win" else (lambda r: r[9])
    R, _ = redundant_at(T, cs)
    while R:
        rows = m42_rows(R, T, cs)
        if all(ok_of(r) for r in rows):
            break
        best_k, best = None, sum(ok_of(r) for r in rows)
        for k in list(R):
            rows2 = m42_rows([x for x in R if x != k], T, cs)
            w2 = sum(ok_of(r) for r in rows2)
            if w2 > best:
                best, best_k = w2, k
        if best_k is None:
            break
        R.remove(best_k)
    return sorted(R)


# ── M45 derivation (verbatim logic, train-subset aware) ──────────────────────
def band_cases_of(cs, lo, hi):
    return [c_ for c_ in cs if lo < c_["n"] <= hi]


def strict_rows_band(drop, lo, hi, cs):
    rows = []
    for c_ in band_cases_of(cs, lo, hi):
        ci = c_["idx"]
        pb = shipped_pool(ci)
        pc = [k for k in pb if k not in drop]
        qb, qc = cost(ci, select(ci, pb)), cost(ci, select(ci, pc))
        rows.append((ci, c_["n"], qb, qc, abs(qc - qb) <= 1e-9 * max(1.0, abs(qb))))
    return rows


def strict_refine_band(lo, hi, cs):
    bc = band_cases_of(cs, lo, hi)
    winners = set(select(c_["idx"], shipped_pool(c_["idx"])) for c_ in bc)
    pool0 = shipped_pool(bc[0]["idx"])
    R = [k for k in pool0 if k not in winners]
    while R:
        rows = strict_rows_band(set(R), lo, hi, cs)
        if all(r[-1] for r in rows):
            break
        best_k, best = None, sum(r[-1] for r in rows)
        for k in list(R):
            w2 = sum(r[-1] for r in strict_rows_band(set(R) - {k}, lo, hi, cs))
            if w2 > best:
                best, best_k = w2, k
        if best_k is None:
            break
        R.remove(best_k)
    return sorted(R)


# ── full-sample self-check: train=all 100 must reproduce shipped constants ───
print("\n== full-sample derivation gates (train = all 100) ==", flush=True)
R100 = refine_m42(100, CASES, "win")
assert set(R100) == BIGSET, (
    f"M42 derivation drift: model {R100} vs shipped {sorted(BIGSET)}")
print(f"refine_m42(100, all, win) == _BIG_REDUNDANT_IDX ({len(BIGSET)} idx) OK")
t3_full = strict_refine_band(60, 100, CASES)
assert set(t3_full) == T3_SHIP, f"tier-3 drift: {t3_full} vs {sorted(T3_SHIP)}"
print(f"strict_refine_band(60,100, all) == _M45_BAND_DROP ({len(T3_SHIP)} idx) OK")
for lo, hi in T4_BANDS:
    r = strict_refine_band(lo, hi, CASES)
    assert set(r) == T4_SHIP[(lo, hi)], (
        f"tier-4 drift ({lo},{hi}]: {r} vs {sorted(T4_SHIP[(lo, hi)])}")
    print(f"strict_refine_band({lo},{hi}, all) == _M45_LOWCORE_DROP "
          f"({len(T4_SHIP[(lo, hi)])} idx) OK")
loc_full = sum(c_["w"] * cost(c_["idx"], select(c_["idx"], live)) for c_ in CASES) / totW
print(f"local RF=1.0 full-pool total = {loc_full:.4f}  (M51 cache reference ~1.3248)")
flush_cost()

# ── fold assignment: deterministic, band-stratified round-robin ──────────────
FOLD_BANDS = [(0, 40), (40, 60), (60, 100), (100, 110), (110, 10 ** 9)]
fold_of = {}
rng = None
if args.seed is not None:
    import random
    rng = random.Random(args.seed)
for lo, hi in FOLD_BANDS:
    ids = sorted(c_["idx"] for c_ in CASES if lo < c_["n"] <= hi)
    if rng is not None:
        rng.shuffle(ids)
    for r, ci in enumerate(ids):
        fold_of[ci] = r % NFOLD
assert len(fold_of) == 100
FOLDS = []
for f in range(NFOLD):
    test = [c_ for c_ in CASES if fold_of[c_["idx"]] == f]
    train = [c_ for c_ in CASES if fold_of[c_["idx"]] != f]
    FOLDS.append((train, test))
assert sorted(ci for _, te in FOLDS for c_ in te for ci in [c_["idx"]]) == list(range(100))
print(f"\n== fold assignment (seed={args.seed}) ==")
for f, (tr, te) in enumerate(FOLDS):
    cnt = {f"({lo},{'inf' if hi >= 10**9 else hi}]":
           len(band_cases_of(te, lo, hi)) for lo, hi in FOLD_BANDS}
    print(f"  fold {f}: test {len(te)}  {cnt}")

# ── per-fold derivations ─────────────────────────────────────────────────────
TS = (100, 105, 110)
GATES = ("win", "strict")
M45_BANDS = [(60, 100)] + T4_BANDS
M42_SETS = {}          # (fold, T, gate) -> sorted list
M45_SETS = {}          # (fold, (lo,hi)) -> sorted list
print("\n== per-fold re-derivation (80 train cases each) ==", flush=True)
for f, (tr, te) in enumerate(FOLDS):
    for T in TS:
        for g in GATES:
            M42_SETS[(f, T, g)] = refine_m42(T, tr, g)
    for lo, hi in M45_BANDS:
        M45_SETS[(f, (lo, hi))] = strict_refine_band(lo, hi, tr)
    m = M42_SETS[(f, 100, 'win')]
    extra = sorted(set(m) - BIGSET); miss = sorted(BIGSET - set(m))
    print(f"  fold {f}: M42@100(win) |R|={len(m)}  extra-vs-shipped={extra}  "
          f"missing-vs-shipped={miss}")
    for lo, hi in M45_BANDS:
        r = M45_SETS[(f, (lo, hi))]
        ship = T3_SHIP if (lo, hi) == (60, 100) else T4_SHIP[(lo, hi)]
        ex = sorted(set(r) - ship); ms = sorted(ship - set(r))
        hs = "inf" if hi >= 10 ** 9 else hi
        print(f"           M45({lo},{hs}] |R|={len(r)}  extra={ex}  missing={ms}")
    flush_cost()

# ── holdout evaluation ───────────────────────────────────────────────────────
TOL = 1e-9


def rel(qb, qc):
    return (qc - qb) / qb * 100.0 if qb > 0 else 0.0


print("\n" + "=" * 72)
print("HOLDOUT: M42 tier (capped-vs-post_swap on held-out cases with n>T)")
print("=" * 72)
m42_hold = {}   # (T, gate) -> list of row dicts
for T in TS:
    for g in GATES:
        recs = []
        for f, (tr, te) in enumerate(FOLDS):
            R = M42_SETS[(f, T, g)]
            for row in m42_rows(R, T, te):
                ci, n, qb, qc, tb, tc, ratio, qrel, win, seq = row
                recs.append(dict(fold=f, ci=ci, n=n, qb=qb, qc=qc,
                                 dq=qc - qb, dqrel=rel(qb, qc),
                                 wcontr=CASES[ci]["w"] * (qc - qb) / totW,
                                 win=win, strict=seq))
        m42_hold[(T, g)] = recs
        nb = sum(1 for r in recs if not r["strict"])
        nw = sum(1 for r in recs if not r["win"])
        wsum = sum(r["wcontr"] for r in recs)
        wpos = sum(max(0.0, r["wcontr"]) for r in recs)
        mx = max((r["dqrel"] for r in recs), default=0.0)
        print(f"\nT={T} gate={g}: {len(recs)} held-out big cases, "
              f"strict-broken {nb}, WIN-broken {nw}, "
              f"dLocal {wsum:+.6f} (regret-only {wpos:+.6f}, "
              f"{wpos / loc_full * 100:+.4f}% of total), max dq {mx:+.3f}%")
        for r in sorted(recs, key=lambda r: -abs(r["dq"])):
            if r["strict"]:
                continue
            print(f"    fold {r['fold']} case {r['ci']:>3} n={r['n']:>3}  "
                  f"q {r['qb']:.4f} -> {r['qc']:.4f}  dq {r['dqrel']:+.3f}%  "
                  f"wContr {r['wcontr']:+.6f}  {'WIN' if r['win'] else 'LOSE'}")

print("\n" + "=" * 72)
print("HOLDOUT: M45 tiers (capped-vs-shipped_pool on held-out band cases)")
print("=" * 72)
m45_hold = {}   # (lo,hi) -> list of row dicts
for lo, hi in M45_BANDS:
    recs = []
    for f, (tr, te) in enumerate(FOLDS):
        R = set(M45_SETS[(f, (lo, hi))])
        for ci, n, qb, qc, seq in strict_rows_band(R, lo, hi, te):
            # WIN diagnostic at the cores where the tier applies
            cores_diag = 12 if (lo, hi) == (60, 100) else 4
            pb = shipped_pool(ci); pc = [k for k in pb if k not in R]
            tb, tc = wall(ci, pb, cores_diag), wall(ci, pc, cores_diag)
            ratio = (tb / tc) ** GAMMA if tc > 0 else 1.0
            qrel = qc / qb if qb > 0 else 1.0
            recs.append(dict(fold=f, ci=ci, n=n, qb=qb, qc=qc,
                             dq=qc - qb, dqrel=rel(qb, qc),
                             wcontr=CASES[ci]["w"] * (qc - qb) / totW,
                             strict=seq, win=qrel <= ratio + 1e-12,
                             cores_diag=cores_diag))
    m45_hold[(lo, hi)] = recs
    hs = "inf" if hi >= 10 ** 9 else hi
    nb = sum(1 for r in recs if not r["strict"])
    nw = sum(1 for r in recs if not r["win"])
    wpos = sum(max(0.0, r["wcontr"]) for r in recs)
    mx = max((r["dqrel"] for r in recs), default=0.0)
    tier = "tier-3 UNIVERSAL" if (lo, hi) == (60, 100) else "tier-4 lowcore"
    print(f"\nband ({lo},{hs}] {tier}: {len(recs)} held-out cases, "
          f"strict-broken {nb}, WIN@{recs[0]['cores_diag'] if recs else '-'}c-broken {nw}, "
          f"regret dLocal {wpos:+.6f} ({wpos / loc_full * 100:+.4f}% of total), "
          f"max dq {mx:+.3f}%")
    for r in sorted(recs, key=lambda r: -abs(r["dq"])):
        if r["strict"]:
            continue
        print(f"    fold {r['fold']} case {r['ci']:>3} n={r['n']:>3}  "
              f"q {r['qb']:.4f} -> {r['qc']:.4f}  dq {r['dqrel']:+.3f}%  "
              f"wContr {r['wcontr']:+.6f}  "
              f"{'WIN' if r['win'] else 'LOSE'}@{r['cores_diag']}c")
flush_cost()

# ── drop-set stability across folds ──────────────────────────────────────────
print("\n" + "=" * 72)
print("DROP-SET STABILITY (folds dropping profile k, 0-5; * = risky:"
      " dropped by some fold but KEPT by shipped)")
print("=" * 72)


def stab_line(tag, ship, fold_sets):
    cnt = defaultdict(int)
    for s in fold_sets:
        for k in s:
            cnt[k] += 1
    parts = []
    for k in sorted(set(cnt) | ship):
        c5 = cnt.get(k, 0)
        risky = "*" if (c5 > 0 and k not in ship) else ""
        inship = "S" if k in ship else "-"
        parts.append(f"#{k}:{c5}{inship}{risky}")
    print(f"  {tag:<16} {' '.join(parts)}")
    risky_ks = sorted(k for k in cnt if k not in ship)
    fragile = sorted(k for k in ship if cnt.get(k, 0) < NFOLD)
    return risky_ks, fragile


risky_m42, fragile_m42 = stab_line(
    "M42@100(win)", BIGSET, [M42_SETS[(f, 100, "win")] for f in range(NFOLD)])
print(f"    -> fold-drop & shipped-keep (holdout-break sources): {risky_m42}")
print(f"    -> shipped-drop but not dropped by all folds: {fragile_m42}")
for lo, hi in M45_BANDS:
    ship = T3_SHIP if (lo, hi) == (60, 100) else T4_SHIP[(lo, hi)]
    hs = "inf" if hi >= 10 ** 9 else hi
    rk, fg = stab_line(f"M45({lo},{hs}]", ship,
                       [M45_SETS[(f, (lo, hi))] for f in range(NFOLD)])
    if rk:
        print(f"    -> fold-drop & shipped-keep: {rk}")

# ── real-score projection: base / shipped / CV-composite chains ──────────────
print("\n" + "=" * 72)
print("PROJECTED REAL TOTAL (M x cores): base=M41 | shipped=in-sample chain | "
      "CV=OOS composite (each case uses its held-out fold's sets)")
print("=" * 72)


def chain_pool(ci, cores, big_of, t3_of, t4_of, T=100):
    """M41 + M42(T) + M45 tier-3 (universal) + tier-4 (cores<=max) chain."""
    n = CASES[ci]["n"]
    pool = [k for k in live if k not in SWAPset]
    if n > T:
        B = big_of(ci)
        pool = [k for k in pool if k not in B]
    D3 = t3_of(ci)
    if 60 < n <= 100:
        pool = [k for k in pool if k not in D3]
    if cores <= oc._M45_CORES_MAX:
        for lo, hi in T4_BANDS:
            if lo < n <= hi:
                D4 = t4_of(ci, (lo, hi))
                pool = [k for k in pool if k not in D4]
    return pool


SHIP_big = lambda ci: BIGSET
SHIP_t3 = lambda ci: T3_SHIP
SHIP_t4 = lambda ci, band: T4_SHIP[band]


def cv_fns(gate):
    big = lambda ci: set(M42_SETS[(fold_of[ci], 100, gate)])
    t3 = lambda ci: set(M45_SETS[(fold_of[ci], (60, 100))])
    t4 = lambda ci, band: set(M45_SETS[(fold_of[ci], band)])
    return big, t3, t4


def rf_total_chain(M, cores, big_of, t3_of, t4_of, T=100):
    s = 0.0
    for c_ in CASES:
        ci = c_["idx"]
        pool = chain_pool(ci, cores, big_of, t3_of, t4_of, T)
        q = cost(ci, select(ci, pool))
        rf = max(FLOOR, (wall(ci, pool, cores) / M) ** GAMMA)
        s += c_["w"] * q * rf
    return s / totW


def rf_total_pool(M, cores, pool_fn):
    s = 0.0
    for c_ in CASES:
        ci = c_["idx"]
        pool = pool_fn(ci)
        q = cost(ci, select(ci, pool))
        rf = max(FLOOR, (wall(ci, pool, cores) / M) ** GAMMA)
        s += c_["w"] * q * rf
    return s / totW


# plumbing self-check: chain with full-sample-derived sets == shipped-constant
# chain pools for every case at low/high cores
full_big = lambda ci: set(R100)
full_t3 = lambda ci: set(t3_full)
_t4_full = {b: set(strict_refine_band(b[0], b[1], CASES)) for b in T4_BANDS}
full_t4 = lambda ci, band: _t4_full[band]
for cores_ in (4, 12):
    for c_ in CASES:
        ci = c_["idx"]
        assert chain_pool(ci, cores_, full_big, full_t3, full_t4) == \
            chain_pool(ci, cores_, SHIP_big, SHIP_t3, SHIP_t4), \
            f"chain plumbing mismatch ci={ci} cores={cores_}"
print("plumbing check: full-sample chain == shipped-constant chain, all cases OK")

CVW = cv_fns("win")
CVS = cv_fns("strict")
proj = {}
print(f"\n{'cores':>5} {'M':>3} {'base':>9} {'shipped':>9} {'dS%':>7} "
      f"{'CVwin':>9} {'dC%':>7} {'tax%':>7} {'CVstrict':>9} {'taxS%':>7}")
for cores_ in (4, 6, 8, 12, 16):
    for M in (6, 8, 11, 14, 20):
        b = rf_total_pool(M, cores_, post_swap)
        sh = rf_total_chain(M, cores_, SHIP_big, SHIP_t3, SHIP_t4)
        cw = rf_total_chain(M, cores_, *CVW)
        cs_ = rf_total_chain(M, cores_, *CVS)
        proj[(cores_, M)] = (b, sh, cw, cs_)
        print(f"{cores_:>5} {M:>3} {b:>9.4f} {sh:>9.4f} {100*(sh-b)/b:>+6.2f}% "
              f"{cw:>9.4f} {100*(cw-b)/b:>+6.2f}% {100*(cw-sh)/sh:>+6.3f}% "
              f"{cs_:>9.4f} {100*(cs_-sh)/sh:>+6.3f}%")
wtax = max((100 * (proj[k][2] - proj[k][1]) / proj[k][1], k) for k in proj)
print(f"\nworst-cell overfit tax (CVwin vs shipped): {wtax[0]:+.3f}% @ "
      f"cores={wtax[1][0]} M={wtax[1][1]}")
flush_cost()

# ── FREE_N sweep: M42-only OOS projection per threshold T ────────────────────
print("\n" + "=" * 72)
print("FREE_N SWEEP (M42 tier only: post_swap + fold M42 set at n>T, fully OOS)")
print("=" * 72)


def m42_only_pool(T, gate):
    def f(ci):
        n = CASES[ci]["n"]
        pool = [k for k in live if k not in SWAPset]
        if n > T:
            R = set(M42_SETS[(fold_of[ci], T, gate)])
            pool = [k for k in pool if k not in R]
        return pool
    return f


freen = {}
for T in TS:
    recs = m42_hold[(T, "win")]
    nb = sum(1 for r in recs if not r["strict"])
    nw = sum(1 for r in recs if not r["win"])
    wpos = sum(max(0.0, r["wcontr"]) for r in recs)
    cells = {}
    worst = (-1e9, None)
    for cores_ in (4, 8, 12):
        for M in (6, 8, 11, 14, 20):
            b = rf_total_pool(M, cores_, post_swap)
            v = rf_total_pool(M, cores_, m42_only_pool(T, "win"))
            d = 100 * (v - b) / b
            cells[(cores_, M)] = d
            if d > worst[0]:
                worst = (d, (cores_, M))
    freen[T] = dict(strict_breaks=nb, win_breaks=nw, regret=wpos, cells=cells,
                    worst=worst)
    shown = "  ".join(f"@{c}c,M={M}:{cells[(c, M)]:+.2f}%"
                      for c, M in ((12, 11), (12, 6), (8, 8), (4, 6)))
    print(f"T={T}: strict-broken {nb}/{len(recs)}, WIN-broken {nw}, "
          f"OOS regret dLocal {wpos:+.6f}")
    print(f"        OOS proj vs M41: {shown}  worst {worst[0]:+.2f}% "
          f"@cores={worst[1][0]},M={worst[1][1]}")

b100 = freen[100]
reco = "KEEP ICCAD_ADAPTIVE_FREE_N=100"
why = "no strict holdout break at T=100" if b100["strict_breaks"] == 0 else None
if b100["strict_breaks"] > 0:
    # do the breaks vanish at a higher T without hurting the OOS projection?
    for T in (105, 110):
        dworst = freen[T]["worst"][0] - b100["worst"][0]
        if freen[T]["strict_breaks"] < b100["strict_breaks"] and dworst <= 0.1:
            reco = f"CONSIDER RELAXING FREE_N -> {T}"
            why = (f"T={T} cuts strict breaks {b100['strict_breaks']} -> "
                   f"{freen[T]['strict_breaks']} with worst-cell OOS delta "
                   f"{dworst:+.3f}%")
            break
    else:
        why = (f"breaks persist or projection worsens at higher T "
               f"(105: {freen[105]['strict_breaks']}, 110: {freen[110]['strict_breaks']})")
        if b100["win_breaks"] == 0:
            why += "; all breaks still median-independent WINs (RF-safe)"
print(f"\nRECOMMENDATION: {reco}  [{why}]")

# ── dump json ────────────────────────────────────────────────────────────────
out = dict(
    seed=args.seed,
    fold_of={str(k): v for k, v in fold_of.items()},
    m42_sets={f"f{f}_T{T}_{g}": M42_SETS[(f, T, g)]
              for f in range(NFOLD) for T in TS for g in GATES},
    m45_sets={f"f{f}_{lo}_{hi if hi < 10**9 else 'inf'}": M45_SETS[(f, (lo, hi))]
              for f in range(NFOLD) for lo, hi in M45_BANDS},
    m42_holdout={f"T{T}_{g}": m42_hold[(T, g)] for T in TS for g in GATES},
    m45_holdout={f"{lo}_{hi if hi < 10**9 else 'inf'}": m45_hold[(lo, hi)]
                 for lo, hi in M45_BANDS},
    projection={f"c{c}_M{M}": dict(base=v[0], shipped=v[1], cv_win=v[2],
                                   cv_strict=v[3])
                for (c, M), v in proj.items()},
    freen={str(T): dict(strict_breaks=freen[T]["strict_breaks"],
                        win_breaks=freen[T]["win_breaks"],
                        regret=freen[T]["regret"],
                        worst=dict(delta=freen[T]["worst"][0],
                                   cell=list(freen[T]["worst"][1])))
           for T in TS},
    recommendation=reco,
)
json.dump(out, open(OUTJSON, "w"), indent=1)
flush_cost()
print(f"\ndumped {OUTJSON.name}")
print("M55 CV DONE")
