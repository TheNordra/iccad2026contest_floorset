#!/usr/bin/env python3
"""
Proxy-selector analysis (OFFLINE ONLY — uses ground-truth baseline, must NOT be
used in the live selector).

Goal: before tuning the portfolio's contest-shape proxy, measure the ceiling.
Runs all default profiles per case, computes each profile's TRUE contest cost
via the harness's own evaluate_solution (real ground-truth baseline), and reports:

  - current-proxy selection score  (what the live portfolio gets)
  - ORACLE selection score          (always pick the true-best profile = ceiling)
  - each-profile-alone score        (lower-bound reference)

It also dumps raw per-case per-profile data to proxy_raw.json so alternative
proxy formulas can be evaluated offline (proxy_search.py) without re-running SA.

Usage (run from repo root; data_path resolves to ./ which is what
iccad2026contest/../ points to):
  python proxy_analysis.py            # all 100 cases (~15 min)
  python proxy_analysis.py --limit 5  # smoke test on first 5 cases
"""
import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import (
    ContestEvaluator, evaluate_solution, compute_cost, compute_total_score,
)
# NOTE: build_opt_target_pos (below) is the only live export — it is imported by
# analyze_constructive.py / dbg_constructive.py / dbg_boundary.py. The offline
# proxy-ceiling analysis in run_all_profiles()/main() depended on the now-deleted
# optimizer_portfolio module, so that import is lazy and those paths are dead.


def build_opt_target_pos(target_pos, constraints, block_count):
    """Replicate ContestEvaluator.evaluate()'s opt_target_pos construction:
    preplaced -> (x,y,w,h); fixed-shape -> (.,.,w,h); else -1."""
    opt = torch.full((block_count, 4), -1.0)
    if target_pos is not None and constraints is not None:
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(block_count):
            is_fixed = nc > 0 and constraints[i, 0] != 0
            is_preplaced = nc > 1 and constraints[i, 1] != 0
            if is_preplaced:
                tx, ty, tw, th = target_pos[i]
                opt[i] = torch.tensor([tx, ty, tw, th])
            elif is_fixed:
                _, _, tw, th = target_pos[i]
                opt[i, 2] = tw
                opt[i, 3] = th
    return opt


def run_all_profiles(profiles, n, area_t, b2b, p2b, pins, constraints, opt_tp):
    """Run every profile's C++ subprocess in parallel; return {profile: positions}."""
    import optimizer_portfolio as pf  # dead: module removed in cleanup
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(profiles)) as ex:
        futs = {
            ex.submit(pf._run_one_profile, prof, n, area_t, b2b, p2b, pins,
                      constraints, opt_tp): prof
            for prof in profiles
        }
        for fut in concurrent.futures.as_completed(futs):
            prof, positions, _dt = fut.result()
            if positions is not None:
                out[prof] = positions
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N cases")
    ap.add_argument("--out", default=str(_DIR / "proxy_raw.json"))
    args = ap.parse_args()

    import optimizer_portfolio as pf  # dead: module removed in cleanup
    pf._ensure_compiled()
    profiles = list(pf._DEFAULT_PROFILES)
    if "gnn" in profiles and not pf._gnn_available():
        profiles = [p for p in profiles if p != "gnn"]
    print(f"profiles ({len(profiles)}): {profiles}", file=sys.stderr)

    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    ncases = len(ev.dataset)
    ids = list(range(ncases if args.limit <= 0 else min(args.limit, ncases)))

    records = []
    t_start = time.time()
    for k, idx in enumerate(ids):
        sample = ev.dataset[idx]
        inputs, labels = sample["input"], sample["label"]
        area_t, b2b, p2b, pins, constraints = inputs
        n = int((area_t != -1).sum().item())
        baseline, target_pos = ev._extract_baseline(idx, labels, b2b, p2b, pins, n)
        opt_tp = build_opt_target_pos(target_pos, constraints, n)

        positions_by_prof = run_all_profiles(
            profiles, n, area_t, b2b, p2b, pins, constraints, opt_tp)
        if not positions_by_prof:
            print(f"[warn] case {idx}: all profiles failed", file=sys.stderr)
            continue

        # Live proxy values (exactly what the portfolio computes to select).
        results_for_pick = {p: (pos, 0.0) for p, pos in positions_by_prof.items()}
        proxy_winner, _, proxy_dbg = pf._pick_best(
            results_for_pick, n, area_t, b2b, p2b, pins, constraints)

        prof_rec = {}
        for prof, positions in positions_by_prof.items():
            m = evaluate_solution(
                {"positions": positions, "runtime": 1.0}, baseline, constraints,
                b2b, p2b, pins, area_t, target_pos, median_runtime=1.0)
            true_cost = compute_cost(
                m.hpwl_gap, m.area_gap, m.violations_relative, 1.0, m.is_feasible)
            proxy, ph, pa, pv = proxy_dbg[prof]
            prof_rec[prof] = {
                "true_cost": true_cost,
                "feasible": bool(m.is_feasible),
                "hpwl_gap": m.hpwl_gap, "area_gap": m.area_gap,
                "vrel_eval": m.violations_relative,
                "proxy": proxy, "hpwl": ph, "area": pa, "vrel_pf": pv,
            }

        records.append({
            "idx": idx, "n": n,
            "H": baseline["hpwl_baseline"], "A": baseline["area_baseline"],
            "sum_block_area": float(sum(float(area_t[i]) for i in range(n))),
            "proxy_winner": proxy_winner,
            "profiles": prof_rec,
        })
        if (k + 1) % 10 == 0 or k + 1 == len(ids):
            print(f"  {k+1}/{len(ids)} cases  ({time.time()-t_start:.0f}s)",
                  file=sys.stderr)

    json.dump(records, open(args.out, "w"))
    print(f"\nsaved {len(records)} cases -> {args.out}", file=sys.stderr)

    # ---- Summaries ----
    def score(selector):
        costs, ns = [], []
        for r in records:
            prof = selector(r)
            costs.append(r["profiles"][prof]["true_cost"])
            ns.append(r["n"])
        return compute_total_score(costs, ns)

    def oracle_sel(r):
        return min(r["profiles"], key=lambda p: r["profiles"][p]["true_cost"])

    cur = score(lambda r: r["proxy_winner"])
    orc = score(oracle_sel)
    print(f"\n=== SELECTION CEILING ({len(records)} cases) ===")
    print(f"current proxy : {cur:.4f}")
    print(f"ORACLE        : {orc:.4f}   (gap to proxy: {cur-orc:+.4f}, "
          f"{(cur-orc)/cur*100:.1f}%)")
    print("\nper-profile-alone:")
    for prof in profiles:
        if all(prof in r["profiles"] for r in records):
            s = score(lambda r, p=prof: p)
            print(f"  {prof:14s} {s:.4f}")

    # proxy pick accuracy
    hits = sum(1 for r in records if r["proxy_winner"] == oracle_sel(r))
    print(f"\nproxy picks true-best: {hits}/{len(records)} "
          f"({hits/len(records)*100:.0f}%)")


if __name__ == "__main__":
    main()
