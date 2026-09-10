"""M56: feature-conditioned PER-CASE profile pruning - oracle ceiling probe (OFFLINE).

M42/M45 prune by n-band only. Question: if a model could pick the kept pool
PER CASE (conditioned on input features), what is the ceiling? For any
selection-preserving per-case pruning the ceiling is the WINNER-ONLY pool:
  - the proxy is oracle-min (M31: zero leakage), so pruning cannot improve Q;
  - a singleton pool trivially selects its only member -> Q bit-identical;
  - wall drops to max(t_w, t_w/cores) = t_w, the minimum with Q preserved.
So the oracle pool is exactly {select(ci, shipped_chain_pool)} per (case, cores)
- no "necessary support set" ambiguity exists.

Phase A (mode "oracle"): project the real RF-aware total over the M x cores
grid, shipped chain (M41+M42+M45 tiers) vs winner-only oracle. Decision rule
(user-set): max cell gain < 0.5% -> dead-end ledger, stop. Else Phase B.

Phase B (mode "cv"): 5-fold band-stratified CV of a simple feature model
(kNN winner-union; J chosen in-fold by train-LOO INDEX preservation, no OOS
leakage). Gate (user-set): out-of-fold STRICT selection preservation (cost
equality, rel 1e-9) must be 100% over all held-out cases and both cores
regimes, else RED.

Known limitation (same as M55): audit_cache dt is the n>100 K=12
counterfactual (M49/M50 REFINE cuts not reflected; dt conservative). Both
sides (shipped/oracle) use the same cache dt; conclusions target selection
structure, orthogonal to the REFINE axis.

Run:  python -u m56_percase_oracle.py oracle
      python -u m56_percase_oracle.py cv [--seed N]
Artifacts: m56_cache.pkl (cost persistence; PM reused read-only from
m55_cv_cache.pkl when fresh), results_M56_percase.json.
Never shipped; touches no shipped file.
"""
import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc

RH = 1.4
GAMMA = 0.3
FLOOR = 0.7
NFOLD = 5
CORES_GRID = (4, 6, 8, 12, 16)
M_GRID = (6, 8, 11, 14, 20)
CEIL_BAR = 0.5            # % - user-set Phase A bar (max cell gain)
CACHE = _DIR / "audit_cache.pkl"
M55CACHE = _DIR / "m55_cv_cache.pkl"
M56CACHE = _DIR / "m56_cache.pkl"
OUTJSON = _DIR / "results_M56_percase.json"

# Must match profile_audit.py's cache key exactly (its PROFILES = live + OM16).
OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
# M72: exclude the appended, gated-off _M55_EXTRA tier (see profile_audit.py).
_SHIPPED = list(oc._PROFILES[:getattr(oc, "_M55_BASE_LEN", len(oc._PROFILES))])
PROFILES = _SHIPPED + [OM16]
N_LIVE = len(_SHIPPED)
# M74: signature pins the binary and the overlay mode too (see profile_audit.py).
_MODE_KEY = repr(("base", repr(sorted(oc._m71_env().items()))))
_EXE_MD5 = hashlib.md5((_DIR / "constructive.exe").read_bytes()).hexdigest()
FPR = repr((repr(PROFILES), _MODE_KEY, _EXE_MD5))

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["oracle", "cv"])
ap.add_argument("--seed", type=int, default=None,
                help="cv: shuffle within bands before round-robin folds")
args = ap.parse_args()


def pname(prof):
    if not prof:
        return "base"
    short = {"ICCAD_WIRE_MULT": "W", "ICCAD_ANCHOR_W": "anc", "ICCAD_LR_ASPECT": "LR",
             "ICCAD_TB_ASPECT": "TB", "ICCAD_FRAME_ASPECTS": "fa", "ICCAD_FRAME_SCALES": "fs",
             "ICCAD_WIRE_TIEBREAK": "WT", "ICCAD_WIRE_BFS": "BFS", "ICCAD_BFS_PIN": "PIN",
             "ICCAD_ORDER_SWAP": "OS", "ICCAD_ORDER_MOVE": "OM", "ICCAD_FREE_ASPECT": "FREE",
             "ICCAD_GUIDE_MED": "GM", "ICCAD_FREE_CLUSTER": "FC", "ICCAD_FREE_ANCHORED": "FA",
             "ICCAD_FREE_ANCHORED_BND": "FAbnd", "ICCAD_MIB_ASPECT": "MIB",
             "ICCAD_CLUSTER_ASPECT": "CA"}
    parts = []
    for k, v in prof.items():
        s = short.get(k, k)
        if s in ("WT", "BFS", "PIN", "FREE", "GM", "FC", "FA", "FAbnd"):
            parts.append(s)
        elif s in ("fa", "fs"):
            parts.append(f"{s}{v.split(',')[0]}")
        elif "RATIOS" in k:
            continue
        else:
            parts.append(f"{s}{v}")
    return "+".join(parts)


# -- dataset prep (mirrors m55_dropset_cv.py, + per-case input features) ------
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
    # input-side features (available BEFORE any profile runs; cv mode only)
    nfix = npre = nbnd = nclu = nmib = 0
    if cons.dim() > 1 and cons.shape[1] > 4:
        col0 = cons[:n, 0].tolist(); col1 = cons[:n, 1].tolist()
        mib_l = cons[:n, 2].tolist(); clu_l = cons[:n, 3].tolist()
        bnd_l = cons[:n, 4].tolist()
        nfix = sum(1 for v in col0 if int(v) != 0)
        npre = sum(1 for v in col1 if int(v) != 0)
        nbnd = sum(1 for v in bnd_l if int(v) != 0)
        nclu = int(max(clu_l)) if clu_l else 0
        nmib = int(max(mib_l)) if mib_l else 0
    nb2b = int((b2b[:, 0] >= 0).sum().item()) if b2b.dim() > 1 and b2b.shape[0] else 0
    np2b = int((p2b[:, 0] >= 0).sum().item()) if p2b.dim() > 1 and p2b.shape[0] else 0
    npin = int(pins.shape[0]) if pins.dim() >= 1 else 0
    feat = [float(n), float(nfix), float(npre), float(nbnd), float(nclu),
            float(nmib), math.log(max(sumA, 1e-9)), float(nb2b), float(np2b),
            float(npin)]
    CASES.append(dict(idx=idx, n=n, A_hat=1.035 * max(sumA, 1e-9),
                      w=math.exp(n / 12.0), base=base, tp=tp,
                      at=at, b2b=b2b, p2b=p2b, pins=pins, cons=cons,
                      feat=feat))
totW = sum(c_["w"] for c_ in CASES)

# -- load audit cache (signature gate) ----------------------------------------
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

# -- PM: reuse m55_cv_cache.pkl read-only when fresh; cost persisted in m56 ---
PM = None
seed_cost = {}
if M55CACHE.exists():
    try:
        m55 = pickle.load(open(M55CACHE, "rb"))
        if m55.get("profiles") == FPR:
            PM = m55.get("pm")
            seed_cost = dict(m55.get("cost", {}))
            print(f"PM reused from m55_cv_cache.pkl ({len(PM)} combos, "
                  f"{len(seed_cost)} cost entries seeded)", flush=True)
    except Exception:
        PM = None
cvc = {}
if M56CACHE.exists():
    try:
        cvc = pickle.load(open(M56CACHE, "rb"))
        if cvc.get("profiles") != FPR:
            print("m56 cache signature stale -> recomputing"); cvc = {}
    except Exception:
        cvc = {}
if PM is None:
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
    cvc["pm"] = PM
_cost = cvc.setdefault("cost", {})
for k_, v_ in seed_cost.items():
    _cost.setdefault(k_, v_)
cvc["profiles"] = FPR
_cost_dirty = [True]


def flush_cost():
    if _cost_dirty[0]:
        pickle.dump(cvc, open(M56CACHE, "wb"))
        _cost_dirty[0] = False


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


def select(ci, pool):
    """Deployed _RH=1.4 proxy selector (wrapper parity)."""
    hmin = min(PM[(ci, k)][1] for k in pool) or 1.0
    A = CASES[ci]["A_hat"]
    return min(pool, key=lambda k: (PM[(ci, k)][0] / A + RH * PM[(ci, k)][1] / hmin)
               * math.exp(2 * PM[(ci, k)][2]))


def wall_of(ci, pool, cores):
    ts = [data[(ci, k)][1] for k in pool]
    return max(max(ts), sum(ts) / cores)


live = list(range(N_LIVE))
SWAP = [k for k in live if "ICCAD_ORDER_SWAP" in PROFILES[k] or "ICCAD_ORDER_MOVE" in PROFILES[k]]
SWAPset = set(SWAP)
BIGSET = set(oc._BIG_REDUNDANT_IDX)


def chain_pool(ci, cores):
    """The ACTUAL shipped pool at a given cores count (M41+M42+M45 tier-3/4;
    mirrors rf_score_model.m46_pool / wrapper _pool_indices)."""
    n = CASES[ci]["n"]
    pool = [k for k in live if k not in SWAPset and not (n > 100 and k in BIGSET)]
    if cores <= oc._M45_MID_CORES_MAX:            # M74: tier-3 is cores-gated
        for lo, hi, D in oc._M45_BAND_DROP:
            if lo < n <= hi:
                pool = [k for k in pool if k not in D]
    if cores <= oc._M45_CORES_MAX:
        for lo, hi, D in oc._M45_LOWCORE_DROP:
            if lo < n <= hi:
                pool = [k for k in pool if k not in D]
    return pool


# winner is cores-regime-dependent (tier-4 flips the pool at cores<=8);
# M45 strict gates keep the COST equal across regimes, the index may differ.
_winner = {}


def winner(ci, cores):
    reg = "lo" if cores <= oc._M45_CORES_MAX else "hi"
    if (ci, reg) not in _winner:
        _winner[(ci, reg)] = select(ci, chain_pool(ci, cores))
    return _winner[(ci, reg)]


def rf_total_pool(M, cores, pool_fn):
    s = 0.0
    for c_ in CASES:
        ci = c_["idx"]
        pool = pool_fn(ci, cores)
        q = cost(ci, select(ci, pool))
        rf = max(FLOOR, (wall_of(ci, pool, cores) / M) ** GAMMA)
        s += c_["w"] * q * rf
    return s / totW


def shipped_fn(ci, cores):
    return chain_pool(ci, cores)


def oracle_fn(ci, cores):
    return [winner(ci, cores)]


BANDS = [(0, 40), (40, 60), (60, 100), (100, 110), (110, 10 ** 9)]


def band_tag(n):
    for lo, hi in BANDS:
        if lo < n <= hi:
            return f"({lo},{'inf' if hi >= 10**9 else hi}]"
    return "?"


# -- shared gates -------------------------------------------------------------
loc_full = sum(c_["w"] * cost(c_["idx"], select(c_["idx"], live)) for c_ in CASES) / totW
print(f"local RF=1.0 full-pool total = {loc_full:.4f}  (M51 cache reference ~1.3248)")
assert 1.320 < loc_full < 1.330, "full-pool sanity out of range"
for cores_ in (4, 12):
    for c_ in CASES:
        ci = c_["idx"]
        qs = cost(ci, select(ci, chain_pool(ci, cores_)))
        qo = cost(ci, winner(ci, cores_))
        assert abs(qo - qs) <= 1e-9 * max(1.0, abs(qs)), \
            f"oracle Q mismatch ci={ci} cores={cores_}"
print("gate: oracle Q == shipped Q, 100/100 cases @4c and @12c OK")
flush_cost()

out = dict(mode=args.mode)

# =============================================================================
if args.mode == "oracle":
    print("\n" + "=" * 72)
    print("PHASE A: winner-only oracle ceiling  (shipped chain vs per-case oracle)")
    print("=" * 72)

    proj = {}
    print(f"\n{'cores':>5} {'M':>3} {'shipped':>9} {'oracle':>9} {'gain%':>8}")
    g_max, g_argmax = -1e9, None
    for cores_ in CORES_GRID:
        for M in M_GRID:
            a = rf_total_pool(M, cores_, shipped_fn)
            b = rf_total_pool(M, cores_, oracle_fn)
            assert b <= a + 1e-12, f"oracle worse than shipped @c={cores_},M={M}"
            g = 100 * (a - b) / a
            proj[(cores_, M)] = (a, b, g)
            if g > g_max:
                g_max, g_argmax = g, (cores_, M)
            print(f"{cores_:>5} {M:>3} {a:>9.4f} {b:>9.4f} {g:>+7.3f}%")
    print(f"\nG_max = {g_max:+.3f}%  @ cores={g_argmax[0]} M={g_argmax[1]}   "
          f"(bar = {CEIL_BAR}%)")

    # band mean wall shipped -> oracle
    print(f"\nband mean wall (s), shipped -> winner-only:")
    for lo, hi in BANDS:
        bc = [c_ for c_ in CASES if lo < c_["n"] <= hi]
        if not bc:
            continue
        hs = "inf" if hi >= 10 ** 9 else str(hi)
        row = f"  ({lo},{hs}]"
        for cores_ in (4, 8, 12):
            ws = sum(wall_of(c_["idx"], chain_pool(c_["idx"], cores_), cores_)
                     for c_ in bc) / len(bc)
            wo = sum(data[(c_["idx"], winner(c_["idx"], cores_))][1]
                     for c_ in bc) / len(bc)
            row += f"   @{cores_}c {ws:6.2f} -> {wo:5.2f}"
        print(row)

    # winner == max-setter tally (@12c the big-band wall is max-setter-bound:
    # if the winner IS the max setter, winner-only gains nothing there)
    print(f"\nwinner==max-setter tally @12c (per band):")
    for lo, hi in BANDS:
        bc = [c_ for c_ in CASES if lo < c_["n"] <= hi]
        if not bc:
            continue
        hs = "inf" if hi >= 10 ** 9 else str(hi)
        hit = 0
        for c_ in bc:
            ci = c_["idx"]
            pool = chain_pool(ci, 12)
            ms = max(pool, key=lambda k: data[(ci, k)][1])
            if ms == winner(ci, 12):
                hit += 1
        print(f"  ({lo},{hs}]: {hit}/{len(bc)} cases winner is the max-setter")

    # per-case table for n>100 @12c
    print(f"\nper-case n>100 @12c:  (floor line: wall <= 0.305*M -> RF=0.7)")
    print(f"{'idx':>3} {'n':>4} {'win#':>4} {'tWin':>6} {'wallShip':>8} "
          f"{'(wS/tW)^.3':>10}  winner")
    for c_ in sorted((c_ for c_ in CASES if c_["n"] > 100), key=lambda x: -x["n"]):
        ci = c_["idx"]
        w = winner(ci, 12)
        tw = data[(ci, w)][1]
        ws = wall_of(ci, chain_pool(ci, 12), 12)
        print(f"{ci:>3} {c_['n']:>4} {w:>4} {tw:>6.2f} {ws:>8.2f} "
              f"{(ws / tw) ** GAMMA if tw > 0 else 1.0:>10.3f}  {pname(PROFILES[w])}")

    # keep-J ceiling-decay diagnostic (IN-SAMPLE frequency; bounds Phase B):
    # pool_J = ({winner} U top-(J-1) most frequent winners) ^ chain pool.
    # Subset pools can flip selection via hmin coupling -> report break count.
    print(f"\nkeep-J ceiling decay (pool = winner + top-(J-1) frequent winners):")
    keepj = {}
    for cores_ in (12, 4):
        tal = Counter(winner(c_["idx"], cores_) for c_ in CASES)
        top = [k for k, _ in tal.most_common()]
        print(f"  @{cores_}c winner-frequency top10: "
              + " ".join(f"#{k}x{tal[k]}" for k in top[:10]))
        for J in (1, 2, 4, 8, 16):
            base_set = set(top[:J - 1])
            breaks = 0

            def pj(ci, cores2, _bs=base_set, _c=cores_):
                keep = _bs | {winner(ci, _c)}
                pool = [k for k in chain_pool(ci, _c) if k in keep]
                return pool or chain_pool(ci, _c)

            for c_ in CASES:
                ci = c_["idx"]
                sel = select(ci, pj(ci, cores_))
                if sel != winner(ci, cores_):
                    qs = cost(ci, winner(ci, cores_)); qn = cost(ci, sel)
                    if abs(qn - qs) > 1e-9 * max(1.0, abs(qs)):
                        breaks += 1
            row = f"  @{cores_}c J={J:>2}  strict-breaks {breaks:>2}  gain%:"
            for M in (6, 8, 11):
                a = proj[(cores_, M)][0]
                v = rf_total_pool(M, cores_, pj)
                row += f"  M={M}:{100 * (a - v) / a:+.3f}%"
            keepj[(cores_, J)] = dict(breaks=breaks)
            print(row)
    flush_cost()

    verdict = "RED (below bar)" if g_max < CEIL_BAR else "ABOVE BAR -> Phase B (cv)"
    print(f"\nVERDICT (Phase A): G_max {g_max:+.3f}% vs bar {CEIL_BAR}% -> {verdict}")
    out.update(
        projection={f"c{c2}_M{M}": dict(shipped=v[0], oracle=v[1], gain_pct=v[2])
                    for (c2, M), v in proj.items()},
        g_max=g_max, g_argmax=list(g_argmax), bar=CEIL_BAR, verdict=verdict,
        winner_hi={c_["idx"]: winner(c_["idx"], 12) for c_ in CASES},
        winner_lo={c_["idx"]: winner(c_["idx"], 4) for c_ in CASES},
    )

# =============================================================================
else:  # cv
    print("\n" + "=" * 72)
    print("PHASE B: 5-fold CV of kNN winner-union (OOS strict preservation gate)")
    print("=" * 72)

    # z-scored features + distance matrix
    F = [c_["feat"] for c_ in CASES]
    nf = len(F[0])
    mu = [sum(f[j] for f in F) / 100 for j in range(nf)]
    sd = [max(1e-9, math.sqrt(sum((f[j] - mu[j]) ** 2 for f in F) / 100))
          for j in range(nf)]
    Z = [[(f[j] - mu[j]) / sd[j] for j in range(nf)] for f in F]
    D = [[sum((Z[a][j] - Z[b][j]) ** 2 for j in range(nf)) for b in range(100)]
         for a in range(100)]
    print(f"features ({nf}): n nFix nPre nBnd nClu nMib logSumA nB2B nP2B nPin")

    # folds: deterministic band-stratified round-robin (mirrors m55)
    fold_of = {}
    rng = None
    if args.seed is not None:
        import random
        rng = random.Random(args.seed)
    for lo, hi in BANDS:
        ids = sorted(c_["idx"] for c_ in CASES if lo < c_["n"] <= hi)
        if rng is not None:
            rng.shuffle(ids)
        for r, ci in enumerate(ids):
            fold_of[ci] = r % NFOLD
    assert len(fold_of) == 100
    FOLDS = [([ci for ci in range(100) if fold_of[ci] != f],
              [ci for ci in range(100) if fold_of[ci] == f]) for f in range(NFOLD)]

    REGIMES = (("hi", 12), ("lo", 4))

    def nearest(ci, cand, J):
        return sorted(cand, key=lambda t: (D[ci][t], t))[:J]

    def keep_set(ci, train, J, cores, variant):
        ks = {winner(t, cores) for t in nearest(ci, train, J)}
        if variant == "knn+band":
            bt = band_tag(CASES[ci]["n"])
            ks |= {winner(t, cores) for t in train
                   if band_tag(CASES[t]["n"]) == bt}
        return ks

    def model_pool(ci, train, J, cores, variant):
        ks = keep_set(ci, train, J, cores, variant)
        pool = [k for k in chain_pool(ci, cores) if k in ks]
        return pool or chain_pool(ci, cores)   # fail-open (empty -> full chain)

    def loo_J(train, cores, variant):
        """Smallest J with 100% train-LOO INDEX preservation (stricter than the
        cost gate, zero cost evals; conservative -> only inflates J)."""
        for J in range(1, len(train) + 1):
            ok = True
            for j in train:
                others = [t for t in train if t != j]
                if select(j, model_pool(j, others, J, cores, variant)) != winner(j, cores):
                    ok = False
                    break
            if ok:
                return J
        return None

    VARIANTS = ("knn", "knn+band")
    Jstar = {}
    print("\nper-fold J* (smallest J with 100% train-LOO index preservation):")
    for f, (tr, te) in enumerate(FOLDS):
        for reg, cores_ in REGIMES:
            for v in VARIANTS:
                Jstar[(f, reg, v)] = loo_J(tr, cores_, v)
        print(f"  fold {f}: " + "  ".join(
            f"{reg}/{v}={Jstar[(f, reg, v)]}" for reg, _ in REGIMES for v in VARIANTS))

    # OOS strict preservation over all held-out cases x regimes
    results = {}
    for v in VARIANTS:
        recs = []
        for f, (tr, te) in enumerate(FOLDS):
            for reg, cores_ in REGIMES:
                J = Jstar[(f, reg, v)] or len(tr)
                for ci in te:
                    pool = model_pool(ci, tr, J, cores_, v)
                    sel = select(ci, pool)
                    w = winner(ci, cores_)
                    if sel == w:
                        pres, dq = True, 0.0
                    else:
                        qs, qn = cost(ci, w), cost(ci, sel)
                        pres = abs(qn - qs) <= 1e-9 * max(1.0, abs(qs))
                        dq = (qn - qs) / qs * 100 if qs > 0 else 0.0
                    recs.append(dict(fold=f, ci=ci, reg=reg, n=CASES[ci]["n"],
                                     poolsz=len(pool), preserved=pres, dq=dq))
        nb = sum(1 for r in recs if not r["preserved"])
        mps = sum(r["poolsz"] for r in recs) / len(recs)
        results[v] = dict(recs=recs, breaks=nb, mean_pool=mps)
        print(f"\nvariant {v}: OOS strict-breaks {nb}/{len(recs)} "
              f"(mean kept pool {mps:.1f})")
        for r in sorted(recs, key=lambda r: -abs(r["dq"])):
            if not r["preserved"]:
                print(f"    fold {r['fold']} case {r['ci']:>3} n={r['n']:>3} "
                      f"@{r['reg']}  pool {r['poolsz']}  dq {r['dq']:+.3f}%")
        flush_cost()

    green = [v for v in VARIANTS if results[v]["breaks"] == 0]
    if not green:
        print(f"\nVERDICT (Phase B): NO variant reaches 100% OOS preservation -> RED")
        out["verdict"] = "RED (OOS preservation < 100%)"
    else:
        # realizable OOS matrix for the leanest green variant
        v = min(green, key=lambda v: results[v]["mean_pool"])
        print(f"\nvariant '{v}' is GREEN -> realizable OOS projection "
              f"(each case uses its held-out fold's model):")

        def cvfn(ci, cores):
            f = fold_of[ci]
            reg = "lo" if cores <= oc._M45_CORES_MAX else "hi"
            tr = FOLDS[f][0]
            J = Jstar[(f, reg, v)] or len(tr)
            return model_pool(ci, tr, J, cores, v)

        print(f"{'cores':>5} {'M':>3} {'shipped':>9} {'cvOOS':>9} {'gain%':>8}")
        mat = {}
        for cores_ in CORES_GRID:
            for M in M_GRID:
                a = rf_total_pool(M, cores_, shipped_fn)
                b = rf_total_pool(M, cores_, cvfn)
                mat[(cores_, M)] = (a, b)
                print(f"{cores_:>5} {M:>3} {a:>9.4f} {b:>9.4f} "
                      f"{100 * (a - b) / a:>+7.3f}%")
        out["realizable"] = {f"c{c2}_M{M}": dict(shipped=x[0], cv=x[1])
                             for (c2, M), x in mat.items()}
        out["verdict"] = f"GREEN ({v})"
        print(f"\nVERDICT (Phase B): {out['verdict']}")

    out.update(
        seed=args.seed,
        jstar={f"f{f}_{reg}_{v}": Jstar[(f, reg, v)]
               for f in range(NFOLD) for reg, _ in REGIMES for v in VARIANTS},
        breaks={v: results[v]["breaks"] for v in VARIANTS},
        mean_pool={v: results[v]["mean_pool"] for v in VARIANTS},
        oos={v: [dict(fold=r["fold"], ci=r["ci"], reg=r["reg"], n=r["n"],
                      poolsz=r["poolsz"], preserved=r["preserved"], dq=r["dq"])
                 for r in results[v]["recs"]] for v in VARIANTS},
    )

# -- dump ---------------------------------------------------------------------
prev = {}
if OUTJSON.exists():
    try:
        prev = json.load(open(OUTJSON))
    except Exception:
        prev = {}
prev[args.mode] = out
json.dump(prev, open(OUTJSON, "w"), indent=1)
flush_cost()
print(f"\ndumped {OUTJSON.name} [{args.mode}]")
print("M56 DONE")
