"""L149 verdict — the REFINE cap re-asked, and LP depth k=2 priced on the deployed path.

Base is L137 + L147 (the config we are proposing to ship). Two questions:

  1. L137's hint caps REFINE_ITERS globally at 4 (constructive.cpp:2142), which
     silently subsumed the M49/M50 band-cut AND cut the mid/light bands from 6/12
     to 4. The teammate's cap sweep predates the tangent cut, so the optimum may
     have moved. Arms: ICCAD_HINT_REFINE 6 and 12, with ADAPTIVE_REFINE=0 so the
     cap is the only limiter.

  2. LP depth k=2 on top of the tangent cut measured +0.5967% in set. The added
     work is LP, i.e. single-threaded Python, so it transfers to the grader in
     SECONDS, not as a ratio (L147 §3.5). Priced with the measured per-case dt.
"""
import json
import statistics as st
from pathlib import Path

import l146_rf_price as L

_DIR = Path(__file__).parent
BASE = "results_L147_on_L137.json"
BAR = 0.30          # OOS ship bar, %


def tot(f):
    d = json.load(open(_DIR / f))
    R = d["test_results"]
    return (d["total_score"], sum(1 for r in R if r["is_feasible"]),
            {r["block_count"]: r["runtime_seconds"] for r in R},
            {r["test_id"]: r.get("positions") for r in R})


def main():
    b, bf, _bt, bp = tot(BASE)
    print(f"\nbase (L137+L147)  {b:.12f}  feasible {bf}/100\n")

    print("=== Q1: the REFINE cap, re-asked on top of the tangent cut ===")
    for tag, lab in (("hint6", "ICCAD_HINT_REFINE=6"),
                     ("hint12", "ICCAD_HINT_REFINE=12")):
        f = f"results_L149_{tag}.json"
        if not (_DIR / f).exists():
            print(f"  {lab:<24} MISSING")
            continue
        t, fe, _q, p = tot(f)
        moved = sum(1 for i in bp if p[i] != bp[i])
        print(f"  {lab:<24} {t:.12f}  {100 * (b - t) / b:+8.4f}%  "
              f"feasible {fe}/100  moved {moved}/100"
              + ("   <-- SILENT NO-OP" if moved == 0 else ""))

    print("\n=== Q2: LP depth k=2, min-of-3, priced in SECONDS ===")
    reps = (1, 2, 3)
    have = all((_DIR / f"results_L149_t{r}_{a}.json").exists()
               for r in reps for a in ("base", "lp2"))
    if not have:
        print("  timing reps missing -- chain not finished")
        return 1

    def mins(arm):
        acc = {}
        for r in reps:
            for k, v in tot(f"results_L149_t{r}_{arm}.json")[2].items():
                acc.setdefault(k, []).append(v)
        return {k: min(v) for k, v in acc.items()}

    c, a = mins("base"), mins("lp2")
    dt = {n: a[n] - c[n] for n in c}
    sp = sorted(dt.values())
    q = 100 * (b - tot("results_L148_lp2.json")[0]) / b
    print(f"  added time   min {sp[0]:+.3f}  p50 {st.median(sp):+.3f}  "
          f"p90 {sp[int(0.9 * len(sp))]:+.3f}  max {sp[-1]:+.3f}  sum {sum(sp):+.2f}s")
    print(f"  wall ratio   {sum(a.values()) / sum(c.values()):.4f}x")
    print(f"  quality      {q:+.4f}%  (in-set, L148)")
    r = L.price_seconds(lambda n: dt.get(n, 0.0), quality_delta_pct=q, perm=300)
    print(f"  RF cost      {r['rf_cost']:+.4f}%   "
          f"permuted p50 {r['perm_p50']:+.4f}% / p05 {r['perm_p05']:+.4f}%")
    print(f"  NET          {r['net']:+.4f}%   bar {BAR}%  ->  "
          f"{'PASS' if r['net'] >= BAR else 'FAIL'}")
    print("\n  median sensitivity (only RF moves):")
    rows0 = L.load()
    for s in (1.00, 0.90, 0.85, 0.80, 0.75):
        rows = [dict(x, med=x["med"] * s) for x in rows0]
        rr = L.price_seconds(lambda n: dt.get(n, 0.0), quality_delta_pct=q,
                             rows=rows)
        print(f"    medians x{s:.2f}   RF {rr['rf_cost']:+.4f}%   "
              f"NET {rr['net']:+.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
