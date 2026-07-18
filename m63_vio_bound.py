"""M63 (OFFLINE, never shipped): boundary-regret beam pre-audit — violation
numerator upper bounds on the shipped M51 baseline.

External codex hypothesis A3: a beam over the constrained/anchored placement
stage (future edge-capacity lexicographic key) could lower V_rel. Per the
M57-M61 lesson (line-level facts being true != score value), this script
computes the weighted score upper bound BEFORE any engineering: it re-derives
every case's soft-violation numerator from results_shipped_m51.json positions
under exact official-evaluator semantics, classifies every bit, and zeroes
the movable-attributable ones (HPWL/Area/RF untouched, RF=1).

Tiers (cost' = cost * exp(2*(V'-V)/n_soft)):
  T1 strict-single : zero vBd movable-single bits only
  T2 strict (GATE) : zero all non-preplaced vBd bits (single + cluster-member)
  T3 loose         : T2 + all vCl + vMb (everything except frozen vBd)

Verdict rule: T2 weighted delta < 0.3% -> A3 beam RED without implementation.

Modes:
  (none)  main audit: gate0 + decomposition + tier bounds + M60 cross-check
  pool    pool-wide movable-vBd oracle from audit_cache.pkl (42 profiles x
          100 cases, positions cached by profile_audit.py): does ANY existing
          profile reach fewer movable vBd bits than the shipped layout, and
          at what official cost? Direct pack-time-reachability evidence for
          the beam decision. Caveats: n>100 cache rows are the K=12 REFINE
          counterfactual (not the live K=4 layout); profile #41 is the OM16
          standby (cached but outside the live pool).

Run:  & "C:\\Users\\Nordra\\.conda\\envs\\iccadv\\python.exe" m63_vio_bound.py [pool]
"""
import json
import math
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import (ContestEvaluator, evaluate_solution,  # noqa: E402
                                compute_total_score)
from shapely.geometry import box as _sbox  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

EPS = 1e-6
BAR = 0.003            # 0.3% kill bar on the T2 weighted delta
M60_CASES = (89, 97, 61, 79, 66)   # ledger: zero movable violators (L1 hosts)

# ── dataset ──────────────────────────────────────────────────────────────────
print("[load] dataset ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
CASES = {}
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _inp, _lab = _s["input"], _s["label"]
    _at, _b2b, _p2b, _pins, _cons = _inp
    _n = int((_at != -1).sum().item())
    _base, _tp = _ev._extract_baseline(_idx, _lab, _b2b, _p2b, _pins, _n)
    CASES[_idx] = dict(
        idx=_idx, n=_n, base=_base, tp=_tp, at=_at, b2b=_b2b, p2b=_p2b,
        pins=_pins, cons=_cons,
        cn=[[int(v) for v in _cons[i].tolist()] for i in range(_n)],
    )
print("[load] 100 cases ready", flush=True)

_ship = json.load(open(_DIR / "results_shipped_m51.json"))
SHIP = {t["test_id"]: t for t in _ship["test_results"]}
SHIP_TOTAL = _ship["total_score"]
print(f"[load] shipped json total = {SHIP_TOTAL!r}", flush=True)


def cost_eval(ci, ps):
    """Official strict scoring, RF=1 (runtime=median=1.0)."""
    c = CASES[ci]
    return evaluate_solution(
        {"positions": ps, "runtime": 1.0}, c["base"], c["cons"][: c["n"]],
        c["b2b"], c["p2b"], c["pins"], c["at"][: c["n"]],
        target_positions=c["tp"][: c["n"]], median_runtime=1.0)


def decompose(ci, ps):
    """Mirror evaluator soft-violation semantics bit-by-bit with class labels.

    Returns dict with per-class vBd counts, per-group vCl/vMb, n_soft, and
    the raw violating-bit records for the json dump.
    """
    c = CASES[ci]
    n = c["n"]
    cn = c["cn"]
    # constraint columns: [fixed, preplaced, mib_id, cluster_id, boundary_code]
    pre = [cn[i][1] != 0 for i in range(n)]
    fix = [cn[i][0] != 0 for i in range(n)]
    mib = [cn[i][2] for i in range(n)]
    clu = [cn[i][3] for i in range(n)]
    bnd = [cn[i][4] for i in range(n)]

    n_mib_groups = max(mib) if mib else 0
    n_clu_groups = max(clu) if clu else 0

    # n_soft, mirroring iccad2026_evaluate.py:459-471
    n_soft = sum(1 for b in bnd if b != 0)
    for g in range(1, n_mib_groups + 1):
        n_soft += max(0, sum(1 for m in mib if m == g) - 1)
    for g in range(1, n_clu_groups + 1):
        n_soft += max(0, sum(1 for m in clu if m == g) - 1)

    clu_has_pre = {g: any(pre[i] for i in range(n) if clu[i] == g)
                   for g in range(1, n_clu_groups + 1)}

    # vBd, mirroring :519-541 (bbox from solution, eps 1e-6, bitmask)
    x_min = min(p[0] for p in ps)
    y_min = min(p[1] for p in ps)
    x_max = max(p[0] + p[2] for p in ps)
    y_max = max(p[1] + p[3] for p in ps)
    bd_frozen = bd_cluster = bd_single = bd_single_fixed = 0
    bd_cluster_mixed = 0     # cluster-member violators whose group has a preplaced member
    bd_records = []
    for i in range(n):
        code = bnd[i]
        if code == 0:
            continue
        bx, by, bw, bh = ps[i]
        touches = {
            1: abs(bx - x_min) < EPS,
            2: abs(bx + bw - x_max) < EPS,
            4: abs(by + bh - y_max) < EPS,
            8: abs(by - y_min) < EPS,
        }
        if all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
            continue
        if pre[i]:
            cls = "frozen"
            bd_frozen += 1
        elif clu[i] > 0:
            cls = "cluster"
            bd_cluster += 1
            if clu_has_pre[clu[i]]:
                bd_cluster_mixed += 1
        else:
            cls = "single"
            bd_single += 1
            if fix[i]:
                bd_single_fixed += 1
        bd_records.append(dict(block=i, code=code, cls=cls,
                               fixed=bool(fix[i]), cluster_id=clu[i],
                               cluster_has_pre=bool(clu[i] and clu_has_pre[clu[i]])))

    # vCl, mirroring :501-506 (shapely unary_union per group, zero tolerance)
    v_cl = v_cl_mixed = 0
    cl_records = []
    for g in range(1, n_clu_groups + 1):
        members = [i for i in range(n) if clu[i] == g]
        polys = [_sbox(ps[i][0], ps[i][1], ps[i][0] + ps[i][2],
                       ps[i][1] + ps[i][3]) for i in members]
        u = unary_union(polys)
        frags = len(u.geoms) - 1 if u.geom_type == "MultiPolygon" else 0
        if frags:
            v_cl += frags
            if clu_has_pre[g]:
                v_cl_mixed += frags
            cl_records.append(dict(group=g, fragments=frags,
                                   has_pre=bool(clu_has_pre[g]),
                                   size=len(members)))

    # vMb, mirroring :511-517 (distinct rounded (w,h) per group)
    v_mb = 0
    for g in range(1, n_mib_groups + 1):
        shapes = {(round(ps[i][2], 4), round(ps[i][3], 4))
                  for i in range(n) if mib[i] == g}
        if shapes:
            v_mb += len(shapes) - 1

    return dict(
        n_soft=n_soft,
        bd_frozen=bd_frozen, bd_cluster=bd_cluster, bd_single=bd_single,
        bd_single_fixed=bd_single_fixed, bd_cluster_mixed=bd_cluster_mixed,
        v_cl=v_cl, v_cl_mixed=v_cl_mixed, v_mb=v_mb,
        bd_records=bd_records, cl_records=cl_records,
    )


# ── pool mode: movable-vBd oracle over audit_cache.pkl ───────────────────────
def bd_classes(ci, ps):
    """Lean vBd-only classification -> (frozen, cluster, single) counts."""
    c = CASES[ci]
    n = c["n"]
    cn = c["cn"]
    x0 = min(p[0] for p in ps)
    y0 = min(p[1] for p in ps)
    x1 = max(p[0] + p[2] for p in ps)
    y1 = max(p[1] + p[3] for p in ps)
    frz = clu = sgl = 0
    for i in range(n):
        code = cn[i][4]
        if code == 0:
            continue
        bx, by, bw, bh = ps[i]
        t = {1: abs(bx - x0) < EPS, 2: abs(bx + bw - x1) < EPS,
             4: abs(by + bh - y1) < EPS, 8: abs(by - y0) < EPS}
        if all(t[b] for b in (1, 2, 4, 8) if code & b):
            continue
        if cn[i][1]:
            frz += 1
        elif cn[i][3] > 0:
            clu += 1
        else:
            sgl += 1
    return frz, clu, sgl


if len(sys.argv) > 1 and sys.argv[1] == "pool":
    cache = pickle.load(open(_DIR / "audit_cache.pkl", "rb"))
    data = cache["data"]
    nprof = max(k for (_, k) in data) + 1
    print(f"[pool] audit cache: {len(data)} combos, {nprof} profiles "
          f"(41 live + OM16 standby)")
    print("[pool] CAVEAT: cache rows have no _band_env overlay -> n>60 rows are "
          "full-REFINE K=12 counterfactuals, not the live K=8/K=4 layouts")
    wsum = sum(math.exp(SHIP[i]["block_count"] / 12.0) for i in range(100))
    better, cleared, win_gain = [], 0, 0.0
    for ci in range(100):
        ps_ship = [tuple(p) for p in SHIP[ci]["positions"]]
        f0, c0, s0 = bd_classes(ci, ps_ship)
        mv0 = c0 + s0
        per_k = {}
        for k in range(nprof):
            ent = data.get((ci, k))
            if not ent or not ent[0]:
                continue
            f, cl, sg = bd_classes(ci, [tuple(p) for p in ent[0]])
            per_k[k] = cl + sg
        mn = min(per_k.values())
        ks = sorted(k for k, v in per_k.items() if v == mn)
        if mn == 0:
            cleared += 1
        if mn < mv0:
            best = None
            for k in ks[:8]:      # official cost only for min-mv candidates
                m = cost_eval(ci, [tuple(p) for p in data[(ci, k)][0]])
                cst = m.cost if m.is_feasible else 10.0
                if best is None or cst < best[1]:
                    best = (k, cst)
            better.append((ci, SHIP[ci]["block_count"], mv0, mn, len(ks), best))
            if best[1] < SHIP[ci]["cost"]:
                win_gain += (SHIP[ci]["cost"] - best[1]) * \
                    math.exp(SHIP[ci]["block_count"] / 12.0) / wsum
    print(f"\n=== POOL ORACLE ===  cases where some profile has FEWER movable "
          f"vBd bits than shipped: {len(better)}/100 (pool clears ALL movable "
          f"bits in {cleared}/100 cases)")
    print(f"{'case':>4} {'n':>4} {'shipMv':>6} {'minMv':>5} {'#prof':>5} "
          f"{'bestProf':>8} {'bestCost':>9} {'shipCost':>9} {'d%':>7}")
    for ci, n, mv0, mn, nk, best in better:
        d = (best[1] - SHIP[ci]["cost"]) / SHIP[ci]["cost"] * 100
        print(f"{ci:>4} {n:>4} {mv0:>6} {mn:>5} {nk:>5} "
              f"{best[0]:>8} {best[1]:>9.6f} {SHIP[ci]['cost']:>9.6f} {d:>+7.3f}")
    print(f"\nweighted realizable gain from lower-mv profiles that also win "
          f"official cost: {win_gain / SHIP_TOTAL * 100:.4f}% "
          f"(counterfactual-K caveat applies)")
    sys.exit(0)


# ── main pass: gate0 + decomposition per case ────────────────────────────────
rows = []
max_cost_diff = 0.0
gate_fail = []
for ci in range(100):
    c = CASES[ci]
    ps = [tuple(p) for p in SHIP[ci]["positions"]]
    m = cost_eval(ci, ps)

    # gate 0.1: recomputed cost vs shipped json cost
    dcost = abs(m.cost - SHIP[ci]["cost"])
    max_cost_diff = max(max_cost_diff, dcost)
    if dcost > 1e-9:
        gate_fail.append(f"case {ci}: cost recompute {m.cost!r} vs json {SHIP[ci]['cost']!r}")
    if not m.is_feasible:
        gate_fail.append(f"case {ci}: recompute infeasible")
    if abs(m.violations_relative - SHIP[ci]["violations_relative"]) > 1e-12:
        gate_fail.append(f"case {ci}: vrel recompute {m.violations_relative!r} "
                         f"vs json {SHIP[ci]['violations_relative']!r}")

    d = decompose(ci, ps)
    # cross-check: decomposition sums == evaluator aggregates
    bd_sum = d["bd_frozen"] + d["bd_cluster"] + d["bd_single"]
    assert bd_sum == m.boundary_violations, \
        f"case {ci}: vBd decomp {bd_sum} != evaluator {m.boundary_violations}"
    assert d["v_cl"] == m.grouping_violations, \
        f"case {ci}: vCl decomp {d['v_cl']} != evaluator {m.grouping_violations}"
    assert d["v_mb"] == m.mib_violations, \
        f"case {ci}: vMb decomp {d['v_mb']} != evaluator {m.mib_violations}"
    assert d["n_soft"] == m.max_possible_violations, \
        f"case {ci}: n_soft decomp {d['n_soft']} != evaluator {m.max_possible_violations}"

    V = m.total_soft_violations
    ns = max(d["n_soft"], 1)
    cost = SHIP[ci]["cost"]
    # tier deltas (bits removed from the numerator)
    cut1 = d["bd_single"]
    cut2 = d["bd_single"] + d["bd_cluster"]
    cut3 = cut2 + d["v_cl"] + d["v_mb"]
    tier_costs = [cost * math.exp(2.0 * (-cut) / ns) for cut in (cut1, cut2, cut3)]
    rows.append(dict(
        idx=ci, n=c["n"], cost=cost, n_soft=d["n_soft"], V=V,
        vrel=m.violations_relative,
        bd_frozen=d["bd_frozen"], bd_cluster=d["bd_cluster"],
        bd_single=d["bd_single"], bd_single_fixed=d["bd_single_fixed"],
        bd_cluster_mixed=d["bd_cluster_mixed"],
        v_cl=d["v_cl"], v_cl_mixed=d["v_cl_mixed"], v_mb=d["v_mb"],
        t1=tier_costs[0], t2=tier_costs[1], t3=tier_costs[2],
        bd_records=d["bd_records"], cl_records=d["cl_records"],
    ))
    print(f"[case {ci:>2}] n={c['n']:>3} V={V:>3} ns={d['n_soft']:>3} "
          f"bd(frz/clu/sgl)={d['bd_frozen']}/{d['bd_cluster']}/{d['bd_single']} "
          f"vCl={d['v_cl']} vMb={d['v_mb']} cost={cost:.6f}", flush=True)

# gate 0.3: weighted total from json costs
tot_check = compute_total_score([SHIP[i]["cost"] for i in range(100)],
                                [SHIP[i]["block_count"] for i in range(100)])
if abs(tot_check - SHIP_TOTAL) > 1e-12:
    gate_fail.append(f"total recompute {tot_check!r} vs json {SHIP_TOTAL!r}")

print(f"\n=== GATE 0 ===  max per-case |cost diff| = {max_cost_diff:.3e}; "
      f"total recompute = {tot_check!r} (json {SHIP_TOTAL!r})")
if gate_fail:
    print("GATE 0 FAIL:")
    for g in gate_fail:
        print("  " + g)
    sys.exit(1)
print("GATE 0 PASS (per-case cost, vrel, n_soft, weighted total)")

# ── classification totals ────────────────────────────────────────────────────
tots = {k: sum(r[k] for r in rows) for k in
        ("bd_frozen", "bd_cluster", "bd_single", "bd_single_fixed",
         "bd_cluster_mixed", "v_cl", "v_cl_mixed", "v_mb", "V")}
print(f"\n=== CLASSIFICATION (shipped M51, 100 cases) ===")
print(f"vBd bits total      = {tots['bd_frozen'] + tots['bd_cluster'] + tots['bd_single']}")
print(f"  frozen (preplaced)= {tots['bd_frozen']}")
print(f"  cluster-member    = {tots['bd_cluster']}  (in mixed cluster: {tots['bd_cluster_mixed']})")
print(f"  movable-single    = {tots['bd_single']}  (fixed-shape flagged: {tots['bd_single_fixed']})")
print(f"vCl fragments       = {tots['v_cl']}  (in mixed cluster: {tots['v_cl_mixed']})")
print(f"vMb                 = {tots['v_mb']}")
print(f"V numerator total   = {tots['V']}")

# ── tier totals ──────────────────────────────────────────────────────────────
ncounts = [r["n"] for r in rows]
base_total = compute_total_score([r["cost"] for r in rows], ncounts)
tier_totals = {t: compute_total_score([r[t] for r in rows], ncounts)
               for t in ("t1", "t2", "t3")}
print(f"\n=== TIER UPPER BOUNDS (HPWL/Area/RF untouched, RF=1) ===")
print(f"baseline total = {base_total:.10f}")
for t, label in (("t1", "T1 strict-single (zero vBd movable-single)"),
                 ("t2", "T2 strict GATE  (zero all non-preplaced vBd)"),
                 ("t3", "T3 loose        (T2 + all vCl + vMb)")):
    tt = tier_totals[t]
    dl = (base_total - tt) / base_total
    print(f"{label}: {tt:.10f}  delta = -{dl * 100:.4f}%")

t2_delta = (base_total - tier_totals["t2"]) / base_total
verdict = "RED" if t2_delta < BAR else "ABOVE BAR"
print(f"\n=== VERDICT ===  T2 weighted delta = {t2_delta * 100:.4f}%  "
      f"(bar {BAR * 100:.1f}%)  ->  {verdict}")

# ── per-case contribution table (movable-attributable cases only) ───────────
wsum = sum(math.exp(n / 12.0) for n in ncounts)
movers = [r for r in rows if r["bd_single"] + r["bd_cluster"] > 0]
movers.sort(key=lambda r: (r["cost"] - r["t2"]) * math.exp(r["n"] / 12.0),
            reverse=True)
print(f"\n=== PER-CASE T2 CONTRIBUTIONS ({len(movers)} cases with movable vBd bits) ===")
print(f"{'case':>4} {'n':>4} {'cost':>9} {'ns':>4} "
      f"{'frz':>3} {'clu':>3} {'sgl':>3} {'vCl':>3} "
      f"{'t2cost':>9} {'dW%':>8}")
for r in movers:
    dw = (r["cost"] - r["t2"]) * math.exp(r["n"] / 12.0) / wsum / base_total
    print(f"{r['idx']:>4} {r['n']:>4} {r['cost']:>9.6f} {r['n_soft']:>4} "
          f"{r['bd_frozen']:>3} {r['bd_cluster']:>3} {r['bd_single']:>3} {r['v_cl']:>3} "
          f"{r['t2']:>9.6f} {dw * 100:>8.4f}")

# ── M60 cross-validation ────────────────────────────────────────────────────
print(f"\n=== M60 CROSS-VALIDATION (cases {M60_CASES}: ledger says zero movable "
      f"violators on L1 winner hosts) ===")
m60_ok = True
for ci in M60_CASES:
    r = rows[ci]
    mv = r["bd_single"] + r["bd_cluster"]
    status = "PASS" if mv == 0 else "MISMATCH"
    if mv:
        m60_ok = False
    print(f"case {ci:>2}: bd_single={r['bd_single']} bd_cluster={r['bd_cluster']} "
          f"bd_frozen={r['bd_frozen']} -> {status}")
if not m60_ok:
    print("NOTE: M60 was measured on L1 quality-pool winner hosts; this audit uses "
          "the shipped M51 pool (different winners possible). Also M60's wall "
          "semantics = frame edges at pack time vs official bbox edges (1e-6) "
          "here; pack-time-satisfied bits can violate post-hoc.")

# ── dump ─────────────────────────────────────────────────────────────────────
out = dict(
    baseline="results_shipped_m51.json",
    baseline_total=base_total,
    tier_totals={t: tier_totals[t] for t in ("t1", "t2", "t3")},
    tier_deltas_pct={t: (base_total - tier_totals[t]) / base_total * 100
                     for t in ("t1", "t2", "t3")},
    verdict=verdict,
    bar_pct=BAR * 100,
    classification_totals=tots,
    per_case=[{k: r[k] for k in
               ("idx", "n", "cost", "n_soft", "V", "vrel",
                "bd_frozen", "bd_cluster", "bd_single", "bd_single_fixed",
                "bd_cluster_mixed", "v_cl", "v_cl_mixed", "v_mb",
                "t1", "t2", "t3", "bd_records", "cl_records")}
              for r in rows],
)
with open(_DIR / "results_M63_vio_bound.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\n[dump] results_M63_vio_bound.json written")
