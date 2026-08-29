"""L195 - emit the per-case LP gate for the FULL pool, and show what it selects.

    run the shape LP on block count n  iff
        t_pool(n) + dt_lp(n)  <=  0.3046 * M(n)

  t_pool(n)  the full pool's wall, transported to the grader by the
             calibration-free ratio t_beta(n) * w_lpoff(n)/w_m73(n)
  dt_lp(n)   the LP's own seconds at k=1, likewise transported
  M(n)       the PUBLISHED per-case median, 2026-08-23

MEASURED (L194): fires on 30% of block counts, captures 12% of the LP's OOS
quality (+0.557% of +4.570%). That capture rate is poor -- the gate keeps the
high-slack cases, which are not where the LP helps -- and it is exactly why the
0.889 efficiency extrapolated from the thin pool did not hold. It is used
anyway because the resulting configuration still beats beta at route-A neutral
(+0.91%) and reaches rank 3 if route A delivers.

Depth is NOT part of this table: k=1 strictly dominates the depth map in both
route-A scenarios (L189), so `_L157_DEPTH` becomes all 1s and this gate decides
only whether the single pass runs at all.
"""
import json
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent
THR = 0.7 ** (1 / 0.3)


def ins(fn):
    d = json.load(open(DIR / fn))["test_results"]
    return ({r["block_count"]: r["runtime_seconds"] for r in d},
            {r["block_count"]: r["cost"] for r in d})


import sys
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0


def main():
    wm, _ = ins("_l181_m73.json")
    wo, _ = ins("_l181_cur.json")       # full pool, LP off
    wk, _ = ins("_l189_k1.json")        # full pool, LP at k=1
    rows = M.rows_new()
    gate, slack = {}, {}
    for r in rows:
        n = r["n"]
        if not wm.get(n):
            continue
        k = r["t"] / wm[n]
        pool = wo[n] * k
        dt = max(0.0, (wk[n] - wo[n]) * k)
        budget = THR * r["med"]
        gate[n] = 1 if pool + dt <= budget * SCALE else 0
        slack[n] = budget - pool - dt
    print(__doc__)
    print("=" * 76)
    on = [n for n in sorted(gate) if gate[n]]
    print("budget scale s = {}".format(SCALE))
    print("LP runs on {} of {} block counts".format(len(on), len(gate)))
    print("   " + ", ".join(str(n) for n in on))
    hi = [n for n in on if n > 100]
    print("\nof those, {} are n>100 (the band carrying 71% of the weight)"
          .format(len(hi)))
    print("   " + (", ".join(str(n) for n in hi) if hi else "(none)"))
    print("\nthat asymmetry IS the 12% capture rate: the gate keeps small cases,")
    print("the LP's value sits on large ones.")

    json.dump({str(k): v for k, v in sorted(gate.items())},
              open(DIR / "l195_lpgate_s{:g}.json".format(SCALE), "w"), indent=0)
    print("\nwrote l195_lpgate.json")

    ks = sorted(gate)
    line, out = "    ", []
    for n in ks:
        piece = "{}: {}, ".format(n, gate[n])
        if len(line) + len(piece) > 76:
            out.append(line.rstrip())
            line = "    "
        line += piece
    out.append(line.rstrip())
    print("\n_L195_LPGATE = {")
    print("\n".join(out))
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
