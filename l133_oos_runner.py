"""L133 — run the L129 placer on the M77 OOS sample and emit a results json.

L132 concluded that the in-set gate is a one-case instrument for this candidate
(DENSITY 0.30 and 0.40 cover the IDENTICAL 77 cases and gate 2.8x apart, entirely
on case 67), so DENSITY=0.40 cannot be validated in set. This produces the input
`m77_oos_probe.py score` needs.

Cases come from `m77_oos_probe._specs(sample)` and are loaded with
`m67_oos_probe._load_case`, i.e. the SAME loader the portfolio's own audit uses,
so the candidate sees exactly what the incumbents saw. `tp` is used only for
`build_opt_target_pos` (the preplaced/fixed placements the evaluator hands every
optimizer, which the C++ also reads on stdin) -- the placer itself stays
label-free.

  <python> -u l133_oos_runner.py --sample s1 --out l129_oos_s1_d040.json
  <python> -u l133_oos_runner.py --sample s1 --limit 5 --verbose
"""
import argparse
import json
import time
from collections import defaultdict

import torch

import m77_oos_probe as M
import m67_oos_probe as m67
from proxy_analysis import build_opt_target_pos
import l129_global_placer as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    specs = M._specs(a.sample)
    if a.limit:
        specs = specs[:a.limit]
    print(f"[l133] {len(specs)} cases, sample {a.sample}, "
          f"GORDIAN={int(L.GORDIAN)} DENSITY={L.DENSITY} "
          f"ABUT={int(L.EXACT_ABUT)}", flush=True)

    # one .th load per file, not per case
    by_file = defaultdict(list)
    order = {}
    for i, (case_key, fk, lay_id, n) in enumerate(specs):
        by_file[fk].append((case_key, lay_id, n))
        order[case_key] = i

    rows = []
    cov = 0
    t0 = time.time()
    for fk, items in by_file.items():
        d = torch.load(m67._path_of(fk))
        for case_key, lay_id, n in items:
            lay = m67._load_case(d, lay_id)
            assert lay["n"] == n, f"n mismatch {lay['n']} != {n} on {case_key}"
            otp = build_opt_target_pos(lay["tp"], lay["cons"], lay["n"])
            c = dict(n=lay["n"], at=lay["at"], cons=lay["cons"],
                     b2b=lay["b2b"], p2b=lay["p2b"], pins=lay["pins"], otp=otp)
            t1 = time.perf_counter()
            try:
                P = L.place(c)
                if P is not None and L.LP_POLISH:
                    P = L.lp_polish(c, P)
            except Exception as e:                       # noqa: BLE001
                if a.verbose:
                    print(f"  {case_key:<34} n={n:>3}  EXC {type(e).__name__}: {e}")
                P = None
            dt = time.perf_counter() - t1
            if P is None:
                if a.verbose:
                    print(f"  {case_key:<34} n={n:>3}  no candidate  {dt:.2f}s")
                continue
            cov += 1
            rows.append(dict(oos_id=order[case_key], key=case_key,
                             positions=[list(map(float, p)) for p in P],
                             runtime_seconds=dt))
            if a.verbose:
                print(f"  {case_key:<34} n={n:>3}  ok  {dt:.2f}s")

    rows.sort(key=lambda r: r["oos_id"])
    print(f"\n[l133] covered {cov}/{len(specs)}  "
          f"({100 * cov / max(len(specs), 1):.0f}%)   wall {time.time() - t0:.0f}s")
    if a.out:
        json.dump(dict(submission_name="L129", sample=a.sample,
                       test_results=rows), open(a.out, "w"))
        print(f"[l133] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
