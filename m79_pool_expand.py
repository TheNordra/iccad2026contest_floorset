"""M79 pool expansion screen -- Route C step 2/3 (OFFLINE, never shipped).

HANDOFF_2026-08-02 section 3-C: at 48 cores a case's wall = the SLOWEST profile
in the pool, and the free-restore budget says the pool can grow from ~37 to
75-80 entries before it flips to sum-bound.  So any NEW profile that is not
slower than the incumbent max-setter is free.  The constraint is that it must
be a NEW strategy -- putting pruned ones back is already exhausted (tier-5).

This screen answers, from cache alone (no C++ run, no wall-clock produced):
  (a) does a candidate clear the >0.05% oracle-min bar?
  (b) would the DEPLOYED _RH=1.4 proxy actually pick it (realizable <= oracle)?
  (c) is it free at 48c, i.e. dt <= the incumbent max-setter of that case?
  (d) the handoff section 5 hygiene checks: per-case better/worse/same, and
      max single-case share of the gain < 40%.

The candidates screened by default are the 8 profiles already appended to
_PROFILES at indices >= 41 and already covered by audit_cache.pkl:
  41..44  _M55_EXTRA   -- the six M54/M55 cluster/anchored knobs that ARE
                          implemented in the shipped constructive.cpp but that
                          NO shipped profile uses (dormant C++ capability)
  45..48  _M73_ESCAPE  -- copies of #2/#22/#23/#25 (teammate M76 judged the
                          escape-hatch MECHANISM red; re-screened here as plain
                          pool additions, which is a different question)

BASIS CAVEAT (read before quoting any number): audit_cache.pkl holds the
REFINE K=12 counterfactual -- it has no _band_env overlay, while the shipped
config runs K=4 (n>100) / K=8 (60<n<=100).  M67-F correction B showed the
tier-3 constants drifted for exactly this reason.  So the anchor here is the
full-pool 1.3269 (rf_score_model's documented sanity value), NOT the shipped
1.326473, and every survivor MUST be re-verified under the shipped K overlay
before it means anything.  This screen is a cheap FILTER, not a verdict.

Modes: gate0 | screen | report
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]
sys.path.insert(0, str(_DIR))

import m67e_rf48 as e                             # noqa: E402 (loads dataset)

SHIPPED_N = 41                                    # indices 0..40 == shipped pool
BASE = list(range(SHIPPED_N))
CAND = list(range(SHIPPED_N, 49))
LABEL = {**{k: "M55-dormant-knobs" for k in range(41, 45)},
         **{k: "M73-escape-copy" for k in range(45, 49)}}
BAR = 0.05                                        # oracle-min gain, %
# full 41-profile pool, K=12, RF=1.0.  CLAUDE.md "ICCAD_ADAPTIVE_POOL=0 還原
# full 41-prof = quality-best 1.3248".  (rf_score_model's docstring still says
# 1.3269 -- that is the PRE-M51 40-profile value and is stale since M51 added
# #40 fa22_fc_pin_tight_wire.)
SANITY = 1.3248
OUT_JSON = _DIR / "results_M79_pool_expand.json"

CIS = sorted({k[0] for k in e.DATA})
W = {ci: math.exp(e.CASES[ci]["n"] / 12.0) for ci in CIS}
TOTW = sum(W.values())


def total(sel):
    return sum(W[ci] * sel[ci] for ci in CIS) / TOTW


def base_costs():
    """Per case: the cost the DEPLOYED proxy picks out of the shipped pool."""
    return {ci: e.cost(ci, e.select(ci, BASE)) for ci in CIS}


def oracle_costs(pool):
    return {ci: min(e.cost(ci, k) for k in pool) for ci in CIS}


def mode_gate0():
    n_case = len(CIS)
    n_prof = len({k[1] for k in e.DATA})
    assert n_case == 100 and n_prof >= 49, f"cache {n_case}x{n_prof}"
    print(f"G1 audit cache covers {n_case} cases x {n_prof} profiles   [PASS]")

    sel = base_costs()
    t_sel = total(sel)
    assert abs(t_sel - SANITY) < 5e-4, \
        f"G2 proxy over shipped pool = {t_sel:.6f}, expected ~{SANITY}"
    print(f"G2 proxy(_RH=1.4) over shipped 41 = {t_sel:.6f} "
          f"~= {SANITY} (rf_score_model full-pool anchor)   [PASS]")

    orc = oracle_costs(BASE)
    t_orc = total(orc)
    leak = 100 * (t_sel - t_orc) / t_orc
    print(f"G3 oracle-min over shipped 41 = {t_orc:.6f}; proxy leakage "
          f"{leak:+.4f}%  (M31: 'oracle-min == proxy, zero leakage')   "
          f"[{'PASS' if leak < 0.02 else 'WARN'}]")
    print("\ngate0: ALL PASS")
    return t_sel, t_orc


def band_of(n):
    for lo, hi in e.BANDS:
        if lo < n <= hi:
            return (lo, hi)
    return e.BANDS[0]


def mode_screen():
    sel = base_costs()
    t_sel = total(sel)
    print(f"basis: shipped-41 proxy total {t_sel:.6f} (K=12 counterfactual)\n")

    # incumbent max-setter dt per case, over the pool actually deployed at 48c
    maxdt = {ci: max(e.DATA[(ci, k)][1] for k in e.pool_shipped(ci, 48))
             for ci in CIS}

    print(f"{'k':>3} {'family':<18} {'oracle%':>8} {'real%':>8} {'wins':>5} "
          f"{'top1share':>10} {'dt<=max':>8} {'worst dt/max':>13}")
    rows = {}
    for k in CAND:
        gains, real, free, ratios = {}, {}, 0, []
        for ci in CIS:
            ck = e.cost(ci, k)
            g = max(0.0, sel[ci] - ck)
            gains[ci] = g
            # realizable: does the deployed proxy actually pick it?
            pick = e.select(ci, BASE + [k])
            real[ci] = max(0.0, sel[ci] - e.cost(ci, pick))
            r = e.DATA[(ci, k)][1] / maxdt[ci]
            ratios.append(r)
            free += (r <= 1.0)
        go = 100 * sum(W[ci] * gains[ci] for ci in CIS) / TOTW / t_sel
        gr = 100 * sum(W[ci] * real[ci] for ci in CIS) / TOTW / t_sel
        nwin = sum(1 for ci in CIS if gains[ci] > 1e-9)
        top = (max(W[ci] * gains[ci] for ci in CIS)
               / max(1e-15, sum(W[ci] * gains[ci] for ci in CIS)))
        print(f"{k:>3} {LABEL[k]:<18} {go:>7.3f}% {gr:>7.3f}% {nwin:>5} "
              f"{100 * top:>9.0f}% {free:>6}/100 {max(ratios):>12.2f}x")
        rows[k] = {"family": LABEL[k], "oracle_pct": go, "real_pct": gr,
                   "wins": nwin, "top1_share": top, "dt_free_cases": free,
                   "worst_dt_ratio": max(ratios)}

    # all candidates together
    allp = BASE + CAND
    o_all = oracle_costs(allp)
    s_all = {ci: e.cost(ci, e.select(ci, allp)) for ci in CIS}
    print(f"\nALL 8 added : oracle {100 * (t_sel - total(o_all)) / t_sel:+.3f}%"
          f"   realizable(proxy) {100 * (t_sel - total(s_all)) / t_sel:+.3f}%"
          f"   -> total {total(s_all):.6f}")
    rows["all8"] = {"oracle_pct": 100 * (t_sel - total(o_all)) / t_sel,
                    "real_pct": 100 * (t_sel - total(s_all)) / t_sel,
                    "total": total(s_all)}

    print(f"\nbar = {BAR}% oracle-min (HANDOFF §5 / profile_vs_portfolio.py); "
          f"hygiene: top-1 case share must be < 40%")
    passed = [k for k in CAND if rows[k]["oracle_pct"] > BAR]
    print(f"candidates over the bar: {passed if passed else 'NONE'}")
    return rows


def mode_report():
    g = mode_gate0()
    print()
    rows = mode_screen()
    json.dump({"basis_total": g[0], "rows": rows}, open(OUT_JSON, "w"), indent=1)
    print(f"\n-> {OUT_JSON.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate0", "screen", "report"])
    a = ap.parse_args()
    t0 = time.perf_counter()
    {"gate0": mode_gate0, "screen": mode_screen, "report": mode_report}[a.mode]()
    e.csave()                                    # persist the lazy cost cache
    print(f"\n[{a.mode} {time.perf_counter() - t0:.1f}s]")
