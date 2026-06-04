#!/usr/bin/env python3
"""
Offline proxy-formula search. Reads proxy_raw.json (produced by proxy_analysis.py)
and scores many candidate selectors against the TRUE per-profile contest cost.

LEGIT constraint: a selector may only use per-profile computable fields
(hpwl, area, vrel_pf) and sum_block_area. The ground-truth baseline (H, A) and
true_cost are used ONLY to (a) score the resulting selection and (b) derive
prior constants from aggregate statistics (a learned prior, not a test label).

Run: python proxy_search.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "proxy_raw.json"
records = json.load(open(RAW))
print(f"loaded {len(records)} cases from {RAW.name}\n")


def total_score(costs, ns):
    mx = max(ns)
    w = [math.exp((n - mx) / 12) for n in ns]
    return sum(c * wi for c, wi in zip(costs, w)) / sum(w)


def score(selector):
    costs, ns = [], []
    for r in records:
        p = selector(r)
        costs.append(r["profiles"][p]["true_cost"])
        ns.append(r["n"])
    return total_score(costs, ns)


def best_h(r):
    return min(d["hpwl"] for d in r["profiles"].values()) or 1.0


# ---------- reference selectors ----------
def oracle(r):
    return min(r["profiles"], key=lambda p: r["profiles"][p]["true_cost"])

def current(r):
    return r["proxy_winner"]

def min_vrel(r):
    return min(r["profiles"], key=lambda p: r["profiles"][p]["vrel_pf"])


# ---------- parametrized proxy ----------
def proxy_sel(alpha=0.5, beta=2.0, area_c=1.035, hpwl_c=1.0, clamp_hpwl=True,
              tie_margin=0.0):
    def f(r):
        profs = r["profiles"]
        sa = r["sum_block_area"]
        est_area = area_c * sa if sa > 0 else 1.0
        bh = best_h(r) * hpwl_c   # estimated baseline HPWL (hpwl_c<1: true baseline below pool)
        vals = {}
        for p, d in profs.items():
            ag = max(0.0, (d["area"] - est_area) / est_area) if est_area > 0 else 0.0
            hr = (d["hpwl"] - bh) / bh if bh > 0 else 0.0
            if clamp_hpwl:
                hr = max(0.0, hr)
            vals[p] = (1.0 + alpha * (ag + hr)) * math.exp(beta * d["vrel_pf"])
        if tie_margin <= 0:
            return min(vals, key=vals.get)
        lo = min(vals.values())
        near = [p for p, v in vals.items() if v <= lo * (1 + tie_margin)]
        return min(near, key=lambda p: profs[p]["vrel_pf"])
    return f


orc = score(oracle)
cur = score(current)
print(f"ORACLE (ceiling)          : {orc:.4f}")
print(f"current proxy (recomputed): {score(proxy_sel()):.4f}")
print(f"  (live proxy_winner field: {cur:.4f})")
print(f"min-vrel only             : {score(min_vrel):.4f}")

# ---------- feasibility leak check ----------
inf_pick = inf_avoidable = 0
for r in records:
    p = current(r)
    if not r["profiles"][p]["feasible"]:
        inf_pick += 1
        if any(d["feasible"] for d in r["profiles"].values()):
            inf_avoidable += 1
print(f"\nproxy picks an INFEASIBLE profile: {inf_pick} cases "
      f"({inf_avoidable} avoidable)")

# ---------- baseline-error diagnostics ----------
a_ratio = [r["A"] / r["sum_block_area"] for r in records if r["sum_block_area"] > 0]
h_ratio = [r["H"] / best_h(r) for r in records]
print(f"\nA/sum_block_area : median={statistics.median(a_ratio):.3f} "
      f"mean={statistics.mean(a_ratio):.3f} "
      f"[{min(a_ratio):.2f},{max(a_ratio):.2f}]  (proxy assumes 1.035)")
print(f"H/best_h_in_pool : median={statistics.median(h_ratio):.3f} "
      f"mean={statistics.mean(h_ratio):.3f} "
      f"[{min(h_ratio):.2f},{max(h_ratio):.2f}]  (proxy assumes 1.0=pool)")

# ---------- sweeps ----------
print("\n-- area_c sweep (alpha=.5 beta=2 clamp) --")
for ac in [1.035, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4]:
    print(f"  area_c={ac:.3f}: {score(proxy_sel(area_c=ac)):.4f}")

print("\n-- beta sweep (alpha=.5 area_c=1.035 clamp) --")
for b in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    print(f"  beta={b:.1f}: {score(proxy_sel(beta=b)):.4f}")

print("\n-- alpha sweep (beta=2 area_c=1.035 clamp) --")
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"  alpha={a:.2f}: {score(proxy_sel(alpha=a)):.4f}")

print("\n-- tie-break margin (alpha=.5 beta=2 area_c=1.035) --")
for tm in [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]:
    print(f"  margin={tm:.2f}: {score(proxy_sel(tie_margin=tm)):.4f}")

print("\n-- hpwl_c sweep (alpha=.5 beta=2 area_c=1.035 clamp) --")
for hc in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
    print(f"  hpwl_c={hc:.2f}: {score(proxy_sel(hpwl_c=hc)):.4f}")

print("\n-- no-clamp hpwl_rel --")
print(f"  clamp=False: {score(proxy_sel(clamp_hpwl=False)):.4f}")

# ---------- small joint grid ----------
print("\n-- joint grid (best few) --")
results = []
for a in [0.25, 0.5, 0.75]:
    for b in [1.5, 2.0, 2.5, 3.0]:
        for hc in [1.0, 0.85, 0.8, 0.75]:
            for tm in [0.0, 0.02, 0.05]:
                s = score(proxy_sel(alpha=a, beta=b, hpwl_c=hc, tie_margin=tm))
                results.append((s, a, b, hc, tm))
results.sort()
for s, a, b, hc, tm in results[:10]:
    print(f"  {s:.4f}  alpha={a} beta={b} hpwl_c={hc} margin={tm}")
