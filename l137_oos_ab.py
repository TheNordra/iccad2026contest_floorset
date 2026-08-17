"""L137 — OOS A/B of the GORDIAN hint, on the SHIPPED optimizer.

WHY NOT m77_oos_probe. That tool answers "what is an EXTERNAL candidate worth
when added to the pool". L137 is not an external candidate: it changes what the
pool's own profiles produce. So the honest instrument is simply to run the
shipped `MyOptimizer.solve()` over the OOS sample twice -- hint off, hint on --
and score both officially.

In-set said +0.0437% (1.2284738 -> 1.2279371, 49/100 cases changed), which is
too small for the in-set gate to settle on its own: L132 measured that gate's
noise floor at roughly its own bar. 240 OOS cases is the resolution this needs.

Cases and loading are m77_oos_probe's, so this is the same sample every historical
OOS number in this project used (M67-D/F, M72, M75, M76, L133).

  <python> -u l137_oos_ab.py --sample s1            # reads ICCAD_HINT_MODE
  <python> -u l137_oos_ab.py --sample s1 --limit 20
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

# 🚨 CAPTURE THE KNOB BEFORE IMPORTING THE PROBE. m77_oos_probe deletes every
# ICCAD_* from os.environ at import time (its line ~78), deliberately, so that a
# measurement always runs on shipped defaults rather than on whatever the shell
# is carrying. That discipline is right for its own job and silently destroys
# this one: the first run of this file reported HINT_MODE=0 for BOTH arms and
# produced two byte-identical 240-case results, i.e. a clean, plausible,
# completely empty A/B. Read it first, restore it after the imports.
_KNOBS = {k: os.environ[k] for k in ("ICCAD_HINT_MODE", "ICCAD_HINT_REFINE")
          if k in os.environ}
_HINT_MODE = _KNOBS.get("ICCAD_HINT_MODE", "0")

import torch

import m77_oos_probe as M
import m67_oos_probe as m67
from proxy_analysis import build_opt_target_pos
from iccad2026_evaluate import evaluate_solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # restore the knob the probe's import stripped, so BOTH the python hint and
    # the C++ subprocesses that inherit this environment see it
    mode = _HINT_MODE
    os.environ.update(_KNOBS)
    import optimizer_constructive as oc
    print(f"[l137] restored {_KNOBS or '{}'}  "
          f"(HINT_MODE now {os.environ.get('ICCAD_HINT_MODE', '0')})  "
          f"binary={oc._BIN.name if oc._BIN else '?'}", flush=True)
    opt = oc.MyOptimizer(verbose=False)

    specs = M._specs(a.sample)
    if a.limit:
        specs = specs[:a.limit]
    by_file = defaultdict(list)
    for ck, fk, lay_id, n in specs:
        by_file[fk].append((ck, lay_id, n))

    rows = []
    t0 = time.time()
    for fk, items in by_file.items():
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in items:
            lay = m67._load_case(d, lay_id)
            lay["base"], _dev = m67._baseline_official(lay)
            tp = lay["tp"]
            tt = torch.tensor([[float(v) for v in q] for q in tp])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            t1 = time.perf_counter()
            P = opt.solve(n, lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                          lay["cons"], otp)
            dt = time.perf_counter() - t1
            m = evaluate_solution({"positions": [list(p) for p in P],
                                   "runtime": 1.0},
                                  lay["base"], lay["cons"], lay["b2b"],
                                  lay["p2b"], lay["pins"], lay["at"],
                                  target_positions=tt[:n], median_runtime=1.0)
            rows.append(dict(key=ck, n=n, cost=float(m.cost),
                             feasible=bool(m.is_feasible),
                             hpwl_gap=float(m.hpwl_gap),
                             area_gap=float(m.area_gap),
                             vrel=float(m.violations_relative),
                             runtime=dt))

    ws = sum(math.exp(r["n"] / 12.0) for r in rows)
    def wavg(k):
        return sum(math.exp(r["n"] / 12.0) * r[k] for r in rows) / ws

    print(f"\n=== L137 OOS ({a.sample}), HINT_MODE={mode}, {len(rows)} cases ===")
    print(f"  feasible        {sum(1 for r in rows if r['feasible'])}/{len(rows)}")
    for k in ("cost", "hpwl_gap", "area_gap", "vrel", "runtime"):
        print(f"  weighted {k:<12} {wavg(k):.6f}")
    print(f"  wall            {time.time() - t0:.0f}s")
    if a.out:
        json.dump(dict(hint_mode=mode, sample=a.sample, rows=rows),
                  open(a.out, "w"))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
