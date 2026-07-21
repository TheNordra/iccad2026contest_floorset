"""RF-aware projected-score model (OFFLINE, never shipped).

The official Cost multiplies quality by max(0.7, (runtime/median)^0.3) PER CASE
(iccad2026_evaluate.py:552), but the local harness forces RuntimeFactor=1.0
(:924-940). So every A/B we ever ran is blind to the runtime term, and the
1.3269 "score" is the RF=1.0 fiction. This tool restores RF.

It reuses profile_audit.py's cache (audit_cache.pkl: per (case,profile) ->
(positions, dt)) and its EXACT selection/wall model, so the FULL-pool RF=1.0
total reproduces the official 1.3269 (sanity gate). It then:
  - reports per-profile cpu (mean / max / max on big cases) to find the wall
    setters (expected: the ORDER_SWAP / ORDER_MOVE profiles);
  - defines size-adaptive CAP variants (for n > T drop the swap profiles, which
    are wall-dominant; keep the build-time aspect/cluster/MIB profiles that carry
    the M33-M37 big-case quality wall-free);
  - projects the REAL total  Sigma w_i Q_i max(0.7,(t_i/M)^0.3) / Sigma w_i  over
    a sweep of the unknown cross-submission median M, for FULL vs each CAP;
  - prints the per-big-case median-INDEPENDENT decision  Q_cap/Q_full  vs
    (t_full/t_cap)^0.3  (cap wins iff the former < the latter, no M needed).

Run (after profile_audit.py has refreshed the cache for the current pool):
  python -u rf_score_model.py
"""
import os, sys, math, pickle
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc

RH = 1.4
CORES = 12                       # physical cores -> wrapper wall = max(max_i, sum_i/CORES)
GAMMA = 0.3                      # official runtime damping
FLOOR = 0.7                      # official RF floor
CACHE = _DIR / "audit_cache.pkl"

# Must match profile_audit.py's cache key exactly (its PROFILES = live + OM16).
OM16 = {"ICCAD_ORDER_MOVE": "16", "ICCAD_WIRE_BFS": "1",
        "ICCAD_WIRE_TIEBREAK": "1", "ICCAD_WIRE_MULT": "2.0"}
PROFILES = list(oc._PROFILES) + [OM16]
N_LIVE = len(oc._PROFILES)       # OM16 (index N_LIVE) is a stand-by, NOT in the live pool
FPR = repr(PROFILES)


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


# ── dataset prep (mirrors profile_audit.py) ─────────────────────────────────
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

# ── load cache (must be fresh for the current pool) ─────────────────────────
if not CACHE.exists():
    sys.exit("audit_cache.pkl missing -> run profile_audit.py first")
c = pickle.load(open(CACHE, "rb"))
if c.get("profiles") != FPR:
    sys.exit("cache profile signature != current pool -> re-run profile_audit.py")
data = c["data"]
missing = [(ci, k) for ci in range(100) for k in range(len(PROFILES)) if (ci, k) not in data]
if missing:
    sys.exit(f"cache incomplete ({len(missing)} combos missing) -> profile_audit.py still running")

# ── proxy metrics + lazy true cost (RF=1.0) ─────────────────────────────────
PM = {}
for c_ in CASES:
    ci = c_["idx"]
    for k in range(len(PROFILES)):
        ps, _ = data[(ci, k)]
        m = oc._proxy_metrics(ps, c_["at"], c_["b2b"], c_["p2b"], c_["pins"], c_["cons"], c_["n"])
        PM[(ci, k)] = (m["area"], m["hpwl"], m["vrel"])

_cost = {}


def cost(ci, k):
    if (ci, k) not in _cost:
        c_ = CASES[ci]; ps, _ = data[(ci, k)]
        tc = evaluate_solution({'positions': ps, 'runtime': 1.0}, c_["base"],
                               c_["cons"][:c_["n"]], c_["b2b"], c_["p2b"], c_["pins"],
                               c_["at"][:c_["n"]], target_positions=c_["tp"][:c_["n"]],
                               median_runtime=1.0)
        _cost[(ci, k)] = tc.cost
    return _cost[(ci, k)]


def select(ci, pool):
    """Deployed _RH=1.4 proxy selector (wrapper parity)."""
    hmin = min(PM[(ci, k)][1] for k in pool) or 1.0
    A = CASES[ci]["A_hat"]
    return min(pool, key=lambda k: (PM[(ci, k)][0] / A + RH * PM[(ci, k)][1] / hmin)
               * math.exp(2 * PM[(ci, k)][2]))


def wall(ci, pool, cores=CORES):
    ts = [data[(ci, k)][1] for k in pool]
    return max(max(ts), sum(ts) / cores)


# ── pool variants (per-case profile subsets) ────────────────────────────────
live = list(range(N_LIVE))
SWAP = [k for k in live if "ICCAD_ORDER_SWAP" in PROFILES[k] or "ICCAD_ORDER_MOVE" in PROFILES[k]]


def full(ci):
    return live


def cap(T, drop):
    drop = set(drop)
    def f(ci):
        return [k for k in live if not (CASES[ci]["n"] > T and k in drop)]
    return f


totW = sum(c_["w"] for c_ in CASES)


def local_total(pool_fn):
    return sum(c_["w"] * cost(c_["idx"], select(c_["idx"], pool_fn(c_["idx"]))) for c_ in CASES) / totW


def rf_total(pool_fn, M, cores=CORES):
    s = 0.0
    for c_ in CASES:
        ci = c_["idx"]; pool = pool_fn(ci)
        q = cost(ci, select(ci, pool))
        rf = max(FLOOR, (wall(ci, pool, cores) / M) ** GAMMA)
        s += c_["w"] * q * rf
    return s / totW


# ── report ──────────────────────────────────────────────────────────────────
print(f"pool: {N_LIVE} live profiles + OM16 stand-by   cores={CORES}")
loc_full = local_total(full)
print(f"SANITY: full-pool RF=1.0 total = {loc_full:.4f}  (official M37 = 1.3269)\n")

big = [c_ for c_ in CASES if c_["n"] >= 110]
bigw = sum(c_["w"] for c_ in big) / totW
print(f"big cases n>=110: {len(big)}  weight share = {bigw*100:.1f}%")

# per-profile cpu, sorted by max-cpu on big cases (the wall setters)
print(f"\n{'#':>3} {'cpuMean':>7} {'cpuMax':>7} {'cpuMaxBig':>9}  name")
rows = []
for k in live:
    ts = [data[(c_['idx'], k)][1] for c_ in CASES]
    tbig = [data[(c_['idx'], k)][1] for c_ in big]
    rows.append((max(tbig), sum(ts)/len(ts), max(ts), k))
for mb, mean, mx, k in sorted(rows, reverse=True):
    tag = "  <- SWAP" if k in SWAP else ""
    print(f"{k:>3} {mean:>7.2f} {mx:>7.2f} {mb:>9.2f}  {pname(PROFILES[k])}{tag}")

print(f"\nSWAP profiles (drop candidates for big n): {SWAP}")
print(f"full-pool wall: n>=110 mean = {sum(wall(c_['idx'],live) for c_ in big)/len(big):.1f}s  "
      f"max = {max(wall(c_['idx'],live) for c_ in big):.1f}s")

# ── variant sweep over the unknown median M ─────────────────────────────────
variants = [("full", full)]
for T in (0, 80, 85, 90, 95, 100, 105, 110):
    variants.append((f"capSWAP>{T}", cap(T, SWAP)))
# surgical: drop only the 3 OS16 + OM8 (keep the light OS8 #34/#36) for n>100
OS16OM = [k for k in SWAP if "16" in PROFILES[k].get("ICCAD_ORDER_SWAP", "")
          or "ICCAD_ORDER_MOVE" in PROFILES[k]]
variants.append(("capOS16/OM>100", cap(100, OS16OM)))

print(f"\nlocal (RF=1.0) totals  [quality cost of capping]:")
for tag, fn in variants:
    print(f"  {tag:<14} {local_total(fn):.4f}")

print(f"\nprojected REAL total = Sigma w*Q*max(0.7,(t/M)^0.3)/Sigma w   (sweep median M):")
hdr = "  M(s) " + "".join(f"{tag:>12}" for tag, _ in variants)
print(hdr)
for M in (6, 8, 9, 10, 11, 12, 14, 16, 20, 25):
    row = f"  {M:>4} "
    vals = [rf_total(fn, M) for _, fn in variants]
    best = min(vals)
    for v in vals:
        mark = "*" if abs(v - best) < 1e-9 else " "
        row += f"{v:>11.4f}{mark}"
    print(row)

# ── per-big-case median-INDEPENDENT decision (cap = capSWAP>100) ─────────────
capfn = cap(100, SWAP)
print(f"\nper-case median-INDEPENDENT check (cap = drop SWAP for n>100):")
print(f"{'idx':>3} {'n':>4} {'Qfull':>7} {'Qcap':>7} {'tFull':>6} {'tCap':>6} "
      f"{'(tF/tC)^.3':>10} {'Qc/Qf':>7}  win?")
for c_ in sorted(big, key=lambda x: -x["w"]):
    ci = c_["idx"]
    pf, pc = full(ci), capfn(ci)
    qf, qc = cost(ci, select(ci, pf)), cost(ci, select(ci, pc))
    tf, tc = wall(ci, pf), wall(ci, pc)
    ratio = (tf / tc) ** GAMMA if tc > 0 else 1.0
    qrel = qc / qf if qf > 0 else 1.0
    win = "WIN" if qrel < ratio else "lose"
    print(f"{ci:>3} {c_['n']:>4} {qf:>7.4f} {qc:>7.4f} {tf:>6.1f} {tc:>6.1f} "
          f"{ratio:>10.4f} {qrel:>7.4f}  {win}")

# ── M42: 2nd-order RF lever — prune big-case-redundant BUILD-time profiles ────
# After M41 drops SWAP, the big-case wall is sum/cores-bound by the ~20 expensive
# (~8s) build-time FREE/CA/FC profiles. A build-time profile that is NEVER the
# selected winner on any case with n>T is pure wall dead-weight for those cases
# (the M41 swap argument, applied PER-BIG-N: the audit's GLOBAL LOO would wrongly
# keep profiles that only win small cases). Cap SWAP+them for n>T and project.
# Gate: every capped case must stay a per-case median-INDEPENDENT WIN
# (Qcap/Qfull < (tF/tC)^0.3) vs the post-swap (M41) pool -> robust over all M.
print("\n" + "=" * 64)
print("M42: 2nd-order RF lever - drop big-case-redundant BUILD profiles")
print("=" * 64)
MGRID = (6, 8, 9, 10, 11, 12, 14, 16, 20, 25)
SWAPset = set(SWAP)
BUILD = [k for k in live if k not in SWAPset]
post_swap = cap(0, SWAP)               # M41 shipped pool fn (drop swap for all n)


def redundant_at(T):
    """BUILD profiles that win NO post-swap case with n>T (per-big-n redundant)."""
    winners = set(select(c_["idx"], post_swap(c_["idx"]))
                  for c_ in CASES if c_["n"] > T)
    return [k for k in BUILD if k not in winners], winners


def m42cap(T, R):
    """Drop SWAP for ALL n (M41) AND additionally drop set R for n>T."""
    R = set(R)
    def f(ci):
        n = CASES[ci]["n"]
        return [k for k in live if k not in SWAPset and not (n > T and k in R)]
    return f


def win_check(R, T):
    """Per-case median-INDEPENDENT WIN of m42cap(T,R) vs post_swap, over n>T."""
    capfn = m42cap(T, R); rows = []; allwin = True
    for c_ in CASES:
        if c_["n"] <= T:
            continue
        ci = c_["idx"]
        pb, pc = post_swap(ci), capfn(ci)
        qf, qc = cost(ci, select(ci, pb)), cost(ci, select(ci, pc))
        tf, tc = wall(ci, pb), wall(ci, pc)
        ratio = (tf / tc) ** GAMMA if tc > 0 else 1.0
        qrel = qc / qf if qf > 0 else 1.0
        win = qrel <= ratio + 1e-12
        allwin = allwin and win
        rows.append((ci, c_["n"], qf, qc, tf, tc, ratio, qrel, win))
    return allwin, rows


def refine(T):
    """Greedy: drop all per-big-n-redundant BUILD profiles, then KEEP back any
    whose removal (via hmin coupling) broke a capped case's WIN, until all win."""
    R, _ = redundant_at(T)
    n0 = len(R)
    while R:
        ok, rows = win_check(R, T)
        if ok:
            break
        cur = sum(r[-1] for r in rows)
        best_k, best = None, cur
        for k in list(R):                       # try keeping k (remove from drop)
            _, rows2 = win_check([x for x in R if x != k], T)
            w2 = sum(r[-1] for r in rows2)
            if w2 > best:
                best, best_k = w2, k
        if best_k is None:                      # no single keep helps -> stop
            break
        R.remove(best_k)
    return R, n0


# baseline = M41 post-swap pool
base_real = {M: rf_total(post_swap, M) for M in MGRID}
base_local = local_total(post_swap)
print(f"\nbaseline (M41 post-swap)  local RF=1.0 = {base_local:.4f}")

best_pick = None
for T in (100, 105, 110):
    R, n0 = refine(T)
    naff = len([c_ for c_ in CASES if c_["n"] > T])
    capfn = m42cap(T, R)
    loc = local_total(capfn)
    ok, rows = win_check(R, T)
    print(f"\n-- T={T}  (n>{T}: {naff} cases)  redundant candidates {n0} -> "
          f"DROP {len(R)} for n>{T}  [all-win={ok}]")
    print(f"   drop idx {sorted(R)}")
    for k in sorted(R, key=lambda k: -max(data[(c_['idx'], k)][1] for c_ in big)):
        cb = max(data[(c_['idx'], k)][1] for c_ in big)
        print(f"     #{k:<3} cpuMaxBig {cb:>5.2f}  {pname(PROFILES[k])}")
    print(f"   local RF=1.0:  {loc:.4f}  (baseline {base_local:.4f}, "
          f"d={loc-base_local:+.4f})")
    print(f"   {'idx':>3} {'n':>4} {'Qbase':>7} {'Qcap':>7} {'tBase':>6} {'tCap':>6} "
          f"{'(tB/tC)^.3':>10} {'Qc/Qb':>7}  win?")
    for ci, n, qf, qc, tf, tc, ratio, qrel, win in sorted(rows, key=lambda r: -r[1]):
        print(f"   {ci:>3} {n:>4} {qf:>7.4f} {qc:>7.4f} {tf:>6.1f} {tc:>6.1f} "
              f"{ratio:>10.4f} {qrel:>7.4f}  {'WIN' if win else 'lose'}")
    print(f"   projected REAL total vs M41 post-swap baseline:")
    print(f"     {'M(s)':>5} {'base':>9} {'M42cap':>9} {'delta%':>8}")
    for M in MGRID:
        v = rf_total(capfn, M); b = base_real[M]
        print(f"     {M:>5} {b:>9.4f} {v:>9.4f} {100*(v-b)/b:>+7.2f}%")
    # pick T that all-wins and maximizes the @M=11 real gain
    g11 = (base_real[11] - rf_total(capfn, 11)) / base_real[11]
    if ok and (best_pick is None or g11 > best_pick[1]):
        best_pick = (T, g11, sorted(R), loc)

print("\n" + "-" * 64)
if best_pick:
    T, g11, R, loc = best_pick
    print(f"RECOMMEND: drop for n>{T}  ->  _BIG_REDUNDANT_IDX = {set(R)}")
    print(f"  ICCAD_ADAPTIVE_FREE_N = {T}   real gain @M=11 = {g11*100:+.2f}%   "
          f"local RF=1.0 = {loc:.4f} (vs M41 {base_local:.4f})")
else:
    print("NO all-win cap found -> 2nd-order lever below bar (converged).")

# ── M45: per-band pool tiers — band-scoped redundancy under a STRICT gate ─────
# M42's redundancy is CUMULATIVE ("wins no case with n>T"); a BAND-scoped one
# ("wins no case with lo<n<=hi") frees profiles that are big-case winners but
# mid-band dead weight (#2/#13/#17/#18 win n>110 yet nothing in 61..100). Mid
# cases stay sum-bound even at 12 cores (sum34/12 ~ 5.2s > max34 ~ 4.4s), so a
# mid-band drop is a UNIVERSAL wall cut; small-band and deeper big-band drops
# only pay when the machine is low-core (wall flips to sum/cores) -> tier-4,
# gated by detected cores in the wrapper.
# GATE (stricter than the M42 WIN inequality, which over-credits caps near the
# RF floor): per-case SELECTION-PRESERVING — cost(select(capped)) must EQUAL
# cost(select(shipped)) (rel 1e-9; cost-tied winner flips allowed). Q unchanged
# -> max(0.7,(t/M)^0.3) monotone in t -> a wall-only drop weakly wins for EVERY
# median M and EVERY cores count. Candidates exclude band winners, so an exact
# tie's INDEX winner (e.g. #19 on case 90, tied with #21) is always KEPT.
print("\n" + "=" * 64)
print("M45: per-band tiers - band-scoped redundancy, strict gate")
print("=" * 64)

BIGSET = set(oc._BIG_REDUNDANT_IDX)
R100r, _ = refine(100)
assert set(R100r) == BIGSET, (
    f"shipped _BIG_REDUNDANT_IDX drift: model {sorted(R100r)} vs {sorted(BIGSET)}")
print(f"shipped-chain check: refine(100) == _BIG_REDUNDANT_IDX ({len(BIGSET)} idx) OK")


def shipped_pool(ci):
    """M41+M42 shipped chain (the tier-3 baseline)."""
    n = CASES[ci]["n"]
    return [k for k in live if k not in SWAPset and not (n > 100 and k in BIGSET)]


def band_cases(lo, hi):
    return [c_ for c_ in CASES if lo < c_["n"] <= hi]


def strict_rows(drop, lo, hi):
    """Per-band-case (ci, n, q_shipped, q_capped, equal?) under drop set."""
    rows = []
    for c_ in band_cases(lo, hi):
        ci = c_["idx"]
        pb = shipped_pool(ci)
        pc = [k for k in pb if k not in drop]
        qb, qc = cost(ci, select(ci, pb)), cost(ci, select(ci, pc))
        rows.append((ci, c_["n"], qb, qc, abs(qc - qb) <= 1e-9 * max(1.0, abs(qb))))
    return rows


def strict_refine_band(lo, hi):
    """Drop = band non-winners, greedy keep-back until every band case keeps an
    EQUAL-cost selection (hmin coupling can shift selections)."""
    bc = band_cases(lo, hi)
    winners = set(select(c_["idx"], shipped_pool(c_["idx"])) for c_ in bc)
    pool0 = shipped_pool(bc[0]["idx"])
    R = [k for k in pool0 if k not in winners]
    n0 = len(R)
    while R:
        rows = strict_rows(set(R), lo, hi)
        if all(r[-1] for r in rows):
            break
        best_k, best = None, sum(r[-1] for r in rows)
        for k in list(R):
            w2 = sum(r[-1] for r in strict_rows(set(R) - {k}, lo, hi))
            if w2 > best:
                best, best_k = w2, k
        if best_k is None:
            break
        R.remove(best_k)
    return sorted(R), n0


# band edges must not straddle n=100 (pool composition changes there)
BANDS = [(40, 60), (60, 80), (80, 100), (60, 100), (100, 110), (110, 10**9),
         (100, 10**9)]
BAND_R = {}
for lo, hi in BANDS:
    bc = band_cases(lo, hi)
    if not bc:
        continue
    R, n0 = strict_refine_band(lo, hi)
    ok = all(r[-1] for r in strict_rows(set(R), lo, hi))
    poolsz = len(shipped_pool(bc[0]["idx"]))
    BAND_R[(lo, hi)] = R
    hs = "inf" if hi >= 10**9 else str(hi)
    print(f"\nband ({lo},{hs}]: {len(bc)} cases, pool {poolsz}, candidates {n0} "
          f"-> DROP {len(R)} [strict all-equal={ok}]  kept {poolsz - len(R)}")
    print(f"  drop idx {R}")
    for k in R:
        tb = [data[(c_['idx'], k)][1] for c_ in bc]
        print(f"    #{k:<3} bandCpu mean {sum(tb)/len(tb):5.2f} max {max(tb):5.2f}  "
              f"{pname(PROFILES[k])}")

# fine (60,80]+(80,100] union vs coarse (60,100]
fine_mid = set(BAND_R.get((60, 80), [])) | set(BAND_R.get((80, 100), []))
coarse_mid = set(BAND_R.get((60, 100), []))
print(f"\nmid banding: fine union {sorted(fine_mid)} vs coarse {sorted(coarse_mid)}")

# ── projection: shipped vs +universal(mid) vs +lowcore(all tiers) ─────────────


def m45_fn(band_drop):
    """band_drop: iterable of (lo, hi, set). Applied on top of shipped chain."""
    def f(ci):
        n = CASES[ci]["n"]
        pool = shipped_pool(ci)
        for lo, hi, D in band_drop:
            if lo < n <= hi:
                pool = [k for k in pool if k not in D]
        return pool
    return f


UNI = [(60, 100, coarse_mid)]
LOW = UNI + [(40, 60, set(BAND_R.get((40, 60), []))),
             (100, 110, set(BAND_R.get((100, 110), []))),
             (110, 10**9, set(BAND_R.get((110, 10**9), [])))]
V_ship = shipped_pool
V_uni = m45_fn(UNI)
V_low = m45_fn(LOW)

for tag, fn in (("shipped", V_ship), ("+uni(mid)", V_uni), ("+lowcore(all)", V_low)):
    print(f"  local RF=1.0 total {tag:<14} = {local_total(fn):.4f}")

print(f"\nprojected REAL total, rows = median M, cols = cores (shipped / +uni / +low):")
for cores_ in (4, 6, 8, 12, 16, 48):
    print(f"  cores={cores_}")
    for M in (6, 8, 11, 14, 20):
        a = rf_total(V_ship, M, cores_)
        b = rf_total(V_uni, M, cores_)
        d = rf_total(V_low, M, cores_)
        print(f"    M={M:>3}  {a:.4f}  {b:.4f} ({100*(b-a)/a:+.2f}%)  "
              f"{d:.4f} ({100*(d-a)/a:+.2f}%)")

# per-band mean wall before/after (uni and low) at each cores
print(f"\nband mean wall (s) shipped -> +uni -> +low:")
for lo, hi in ((40, 60), (60, 100), (100, 110), (110, 10**9)):
    bc = band_cases(lo, hi)
    hs = "inf" if hi >= 10**9 else str(hi)
    for cores_ in (4, 8, 12, 48):
        ws = sum(wall(c_["idx"], V_ship(c_["idx"]), cores_) for c_ in bc) / len(bc)
        wu = sum(wall(c_["idx"], V_uni(c_["idx"]), cores_) for c_ in bc) / len(bc)
        wl = sum(wall(c_["idx"], V_low(c_["idx"]), cores_) for c_ in bc) / len(bc)
        print(f"  ({lo},{hs}] cores={cores_:>2}: {ws:6.2f} -> {wu:6.2f} -> {wl:6.2f}")

# ── M67-E: 48-core structure (Beta runs on a dedicated 48c ICELAKE box) ───────
# At 48 cores every shipped pool (13 / 25 / 35 profiles) is smaller than the core
# count, so sum/cores <= (|P|/48)*max_i < max_i: the wall is the MAX-setter, not
# the sum. The per-case crossover c* = sum_i/max_i is the core count above which
# that holds; max c* over a band answers "does the assumption survive if the 48
# logical cores are really 24 physical?". Tier-4 / M50 low-core do NOT fire at 48
# detected cores (fail-open -> universal tiers only), so the 48c pool is exactly
# V_uni (== m46_pool(., cores>8) == _pool_indices() under ICCAD_ADAPTIVE_CORES=48;
# gated in m67e_rf48.py gate0). Detail + alpha-anchored projection: m67e_rf48.py.
print(f"\n48-core structure (shipped pool at 48c; wall = max_i iff cores >= c*):")
print(f"  {'band':>12} {'#':>3} {'|P|':>4} {'max_i':>7} {'sum/48':>7} {'c* p50':>7} "
      f"{'c* max':>7}  max-bound?")
for lo, hi in ((0, 40), (40, 60), (60, 100), (100, 110), (110, 10**9)):
    bc = band_cases(lo, hi)
    if not bc:
        continue
    hs = "inf" if hi >= 10**9 else str(hi)
    mx, s48, cstar = [], [], []
    for c_ in bc:
        ci = c_["idx"]; pool = V_uni(ci)
        ts = [data[(ci, k)][1] for k in pool]
        mx.append(max(ts)); s48.append(sum(ts) / 48.0)
        cstar.append(sum(ts) / max(ts))
    cs = sorted(cstar)
    ok = all(s <= m + 1e-12 for s, m in zip(s48, mx))
    print(f"  ({lo:>3},{hs:>4}] {len(bc):>3} {len(V_uni(bc[0]['idx'])):>4} "
          f"{sum(mx)/len(mx):>7.2f} {sum(s48)/len(s48):>7.2f} "
          f"{cs[len(cs)//2]:>7.1f} {max(cstar):>7.1f}  {'YES' if ok else 'NO'}")

# tie record for the docs (n=111 case; #19 vs #21 under shipped pool)
for c_ in CASES:
    if c_["n"] == 111:
        ci = c_["idx"]
        pb = shipped_pool(ci)
        hmin = min(PM[(ci, k)][1] for k in pb) or 1.0
        A = CASES[ci]["A_hat"]
        for k in (19, 21):
            if k in pb:
                px = (PM[(ci, k)][0] / A + RH * PM[(ci, k)][1] / hmin) * math.exp(2 * PM[(ci, k)][2])
                print(f"\ntie check case {ci} (n=111): #{k} proxy {px:.9f} cost {cost(ci, k):.9f}")

print(f"\nready-to-paste wrapper constants:")
print(f"_M45_BAND_DROP = (  # tier-3 UNIVERSAL (mid cases sum-bound at any cores)")
print(f"    (60, 100, frozenset({sorted(coarse_mid)})),")
print(f")")
print(f"_M45_LOWCORE_DROP = (  # tier-4, only when detected cores <= _M45_CORES_MAX")
print(f"    (40, 60, frozenset({sorted(BAND_R.get((40, 60), []))})),")
print(f"    (100, 110, frozenset({sorted(BAND_R.get((100, 110), []))})),")
print(f"    (110, 10**9, frozenset({sorted(BAND_R.get((110, 10**9), []))})),")
print(f")")

# ── M46: wall-setter speedup sensitivity — uniform & FREE-targeted alpha ──────
# After M45 the pool-prune axis is exhausted ((100,inf] candidates=0); the only
# RF continuation is making the max-setter RUNS faster (C++ hot path). A pure
# speedup keeps positions bit-identical -> Q and selection unchanged -> per-case
# t' <= t -> cost' <= cost for EVERY median and EVERY cores count (stronger than
# the M45 strict gate). This section quantifies the payoff: scale kept-profile
# times by alpha (uniform, or FREE-stack-only) and project the real total.
print("\n" + "=" * 64)
print("M46: max-setter speedup sensitivity (uniform / FREE-stack alpha)")
print("=" * 64)

# shipped-constant drift checks (mirrors the M42 assert)
assert coarse_mid == set(oc._M45_BAND_DROP[0][2]), "tier-3 drift vs shipped"
for lo, hi, d in oc._M45_LOWCORE_DROP:
    assert set(d) == set(BAND_R.get((lo, hi), [])), f"tier-4 drift ({lo},{hi}]"
print("shipped-chain check: M45 tier-3/tier-4 constants match model OK")


def m46_pool(ci, cores):
    """The ACTUAL shipped pool at a given cores count (tier-4 auto-gates)."""
    n = CASES[ci]["n"]
    pool = shipped_pool(ci)
    for lo, hi, D in oc._M45_BAND_DROP:
        if lo < n <= hi:
            pool = [k for k in pool if k not in D]
    if cores <= oc._M45_CORES_MAX:
        for lo, hi, D in oc._M45_LOWCORE_DROP:
            if lo < n <= hi:
                pool = [k for k in pool if k not in D]
    return pool


FREE_SET = {k for k in live if "ICCAD_FREE_ASPECT" in PROFILES[k]}


def rf46(M, cores, af):
    """Real total with per-profile time scaling af(k) (selection/Q unchanged —
    exact for a pure bit-identical speedup)."""
    s = 0.0
    for c_ in CASES:
        ci = c_["idx"]
        pool = m46_pool(ci, cores)
        q = cost(ci, select(ci, pool))
        ts = [data[(ci, k)][1] * af(k) for k in pool]
        w = max(max(ts), sum(ts) / cores)
        s += c_["w"] * q * max(FLOOR, (w / M) ** GAMMA)
    return s / totW


ONE = lambda k: 1.0
# sanity: alpha=1 must reproduce the M45 projections at matching cores
for cores_, ref in ((12, V_uni), (4, V_low)):
    a, b = rf46(11, cores_, ONE), rf_total(ref, 11, cores_)
    assert abs(a - b) < 1e-9, f"alpha=1 sanity failed @cores={cores_}: {a} vs {b}"
print("sanity: alpha=1.0 reproduces shipped projections @12c/@4c OK")

# max-setter record (regen after any exe rebuild)
from collections import Counter
tal = Counter()
for c_ in CASES:
    if c_["n"] > 100:
        ci = c_["idx"]
        pool = m46_pool(ci, 12)
        tal[max(pool, key=lambda k: data[(ci, k)][1])] += 1
print("\nmax-setter tally n>100 @12c: "
      + "  ".join(f"#{k}({pname(PROFILES[k])})x{v}" for k, v in tal.most_common()))

ALPHAS_U = (0.95, 0.9, 0.85, 0.8, 0.7, 0.5)
print("\nprojected REAL delta vs shipped (uniform alpha on ALL kept profiles):")
print(f"{'cores':>6} {'M':>3} " + "".join(f"{'a='+str(a):>9}" for a in ALPHAS_U))
for cores_ in (4, 8, 12, 16, 48):
    for M in (6, 8, 11, 14, 20):
        base = rf46(M, cores_, ONE)
        row = f"{cores_:>6} {M:>3} "
        for a in ALPHAS_U:
            v = rf46(M, cores_, lambda k, a=a: a)
            row += f"{100*(v-base)/base:>+8.2f}%"
        print(row)

ALPHAS_T = (0.9, 0.8, 0.7)
print(f"\nprojected REAL delta vs shipped (alpha on FREE-stack profiles ONLY, "
      f"{len(FREE_SET)} of {N_LIVE}):")
print(f"{'cores':>6} {'M':>3} " + "".join(f"{'a='+str(a):>9}" for a in ALPHAS_T))
for cores_ in (4, 8, 12, 16, 48):
    for M in (8, 11, 14):
        base = rf46(M, cores_, ONE)
        row = f"{cores_:>6} {M:>3} "
        for a in ALPHAS_T:
            v = rf46(M, cores_, lambda k, a=a: a if k in FREE_SET else 1.0)
            row += f"{100*(v-base)/base:>+8.2f}%"
        print(row)

print("\nRF MODEL DONE")
