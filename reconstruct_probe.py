"""OFFLINE ONLY — reconstruction ceiling probe (never shipped).

The reconstruction analog of oracle_perm_probe.py: measure the ORACLE ceiling
BEFORE investing in any reconstruction ML/algorithm.

For each of the 100 validation cases it outputs the ground-truth fp_sol layout
verbatim (each block = bbox of its ground-truth polygon) and scores it with the
EXACT contest evaluator (evaluate_solution). Then it:

  1. reports the weighted oracle Total Score (expect ~1.1079 per CLAUDE.md),
  2. decomposes WHY the oracle is not 1.0 — feasibility, soft V_rel by type,
     residual area/hpwl gap, and the max polygon vertex count
     (4-5 => rectangular blocks => bbox exact; >5 => rectilinear => bbox overlaps),
  3. locates the reconstructable headroom vs our current portfolio
     (optimizer_constructive_results.json, 1.3843), splitting OUR gap into a
     quality factor (hpwl+area) vs a violation factor (V_rel) component.

Self-check: on feasible cases, outputting fp_sol must reproduce baseline metrics
(hpwl_gap ~ 0, area_gap ~ 0). Non-zero => loader/scoring wiring is wrong.

Usage:  python reconstruct_probe.py
"""
import sys, math, json
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution

ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
ev._load_dataset()
PORT = json.load(open(_DIR / "iccad2026contest" / "optimizer_constructive_results.json"))
ours = {t["test_id"]: t for t in PORT["test_results"]}


def run(idx):
    s = ev.dataset[idx]
    inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    ps = [tuple(float(v) for v in tp[i]) for i in range(n)]
    m = evaluate_solution({"positions": ps, "runtime": 1.0}, base, cons[:n],
                          b2b, p2b, pins, at[:n], target_positions=tp[:n],
                          median_runtime=1.0)
    # polygon vertex stats (rect vs rectilinear): max distinct vertices per block
    polygons = lab[0]
    maxv = 0
    for i in range(n):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        maxv = max(maxv, int(len(valid)))
    return n, m, maxv


rows = []
totW = 0.0
for idx in range(100):
    n, m, maxv = run(idx)
    w = math.exp(n / 12.0)
    totW += w
    o = ours[idx]
    rows.append(dict(
        idx=idx, n=n, w=w, maxv=maxv,
        o_cost=float(m.cost), o_feas=bool(m.is_feasible),
        o_hgap=float(m.hpwl_gap), o_agap=float(m.area_gap),
        o_vrel=float(m.violations_relative),
        o_bnd=int(m.boundary_violations), o_grp=int(m.grouping_violations),
        o_mib=int(m.mib_violations),
        our_cost=float(o["cost"]), our_feas=bool(o["is_feasible"]),
        our_hgap=float(o["hpwl_gap"]), our_agap=float(o["area_gap"]),
        our_vrel=float(o["violations_relative"]),
    ))

json.dump(rows, open(_DIR / "reconstruct_probe_cache.json", "w"), indent=0)


def wmean(key):
    return sum(r["w"] * r[key] for r in rows) / totW


oracle_total = wmean("o_cost")
infeas = [r for r in rows if not r["o_feas"]]
maxv_all = max(r["maxv"] for r in rows)

print("=" * 72)
print(f"ORACLE reconstruction Total Score = {oracle_total:.4f}"
      f"   (our portfolio = {PORT['total_score']:.4f})")
print(f"reconstructable headroom (weighted) = {PORT['total_score'] - oracle_total:+.4f}")
print("=" * 72)

# --- self-check: feasible cases must reproduce baseline (gap ~ 0) ---
feas = [r for r in rows if r["o_feas"]]
max_hgap = max((abs(r["o_hgap"]) for r in feas), default=0.0)
max_agap = max((abs(r["o_agap"]) for r in feas), default=0.0)
print(f"\n[self-check] feasible oracle gaps: max|hgap|={max_hgap:.2e} "
      f"max|agap|={max_agap:.2e}  (should be ~0)")

# --- residual decomposition (why oracle != 1.0) ---
print(f"\n[feasibility] {len(feas)}/100 feasible; "
      f"infeasible cases = {[r['idx'] for r in infeas]}")
print(f"[shape] max polygon vertices over all blocks = {maxv_all}  "
      f"(4-5 => rectangular, bbox exact; >5 => rectilinear, bbox overlaps)")
nz_bnd = sum(1 for r in rows if r["o_bnd"])
nz_grp = sum(1 for r in rows if r["o_grp"])
nz_mib = sum(1 for r in rows if r["o_mib"])
print(f"[oracle soft-violations] cases with boundary>0:{nz_bnd}  "
      f"grouping>0:{nz_grp}  mib>0:{nz_mib}")
print(f"[oracle weighted factors] quality(1+0.5(h+a))={1 + 0.5 * (wmean('o_hgap') + wmean('o_agap')):.4f}  "
      f"viol exp(2*Vrel)~={math.exp(2 * wmean('o_vrel')):.4f}  "
      f"meanVrel={wmean('o_vrel'):.4f}")

# --- headroom split: ours vs oracle (quality side vs violation side) ---
print(f"\n[OUR weighted factors]    quality(1+0.5(h+a))={1 + 0.5 * (wmean('our_hgap') + wmean('our_agap')):.4f}  "
      f"viol exp(2*Vrel)~={math.exp(2 * wmean('our_vrel')):.4f}  "
      f"meanVrel={wmean('our_vrel'):.4f}")
print(f"[weighted means] our_hgap={wmean('our_hgap'):.4f} oracle_hgap={wmean('o_hgap'):.4f} | "
      f"our_agap={wmean('our_agap'):.4f} oracle_agap={wmean('o_agap'):.4f} | "
      f"our_vrel={wmean('our_vrel'):.4f} oracle_vrel={wmean('o_vrel'):.4f}")

# --- concentration: top cases by weighted headroom contribution ---
for r in rows:
    r["contrib"] = r["w"] * (r["our_cost"] - r["o_cost"]) / totW
top = sorted(rows, key=lambda r: r["contrib"], reverse=True)[:15]
print("\n[headroom concentration] top-15 cases by weighted (our_cost - oracle_cost):")
print(f"  {'idx':>3} {'n':>3} {'wContr%':>7} {'our':>7} {'oracle':>7} | "
      f"{'o_feas':>6} {'o_vrel':>6} {'o_hgap':>6} {'o_agap':>6} {'maxv':>4}")
cum = 0.0
for r in top:
    cum += r["contrib"]
    print(f"  {r['idx']:>3d} {r['n']:>3d} {r['contrib'] / (PORT['total_score'] - oracle_total) * 100:>6.1f}% "
          f"{r['our_cost']:>7.4f} {r['o_cost']:>7.4f} | {str(r['o_feas']):>6} "
          f"{r['o_vrel']:>6.3f} {r['o_hgap']:>6.3f} {r['o_agap']:>6.3f} {r['maxv']:>4d}")
print(f"  top-15 cover {cum / (PORT['total_score'] - oracle_total) * 100:.1f}% of total headroom")
