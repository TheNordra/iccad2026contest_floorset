"""L250 - partition the quality deficit: SELECTION loss vs GENERATION loss.

§4b established the gap to rank 1 is 2.2% of quality, and that rank 1's lead
over us is 17.9% -- a placer-level gap, not a schedule-level one. Before looking
for a better placer, the deficit has to be split, because the two halves have
completely different fixes and one of them is free:

    label cost            what the ground truth scores (its own vrel is not 0)
      |  GENERATION loss  the pool does not CONTAIN anything better
    oracle-over-pool      the best of the 51 candidates, by TRUE cost
      |  SELECTION loss   the pool contains it and we do not pick it
    proxy pick            what the shipped selector actually returns

CLAUDE.md:488 asserts "proxy is oracle-perfect on heterogeneous candidates
(M76/M77)". That is the assumption this file tests, and it is exactly the shape
of claim that has been overturned three times this session. It is also
internally contradicted by the M80 entry in the same file:

  "hmin coupling is real -- the proxy's hmin is the pool-wide min HPWL, so a new
   candidate that lowers it scales every candidate's hpwl term without touching
   the area term, and the existing ordering can flip"

i.e. the proxy's two baselines are ESTIMATES:

    proxy = (area/A_hat + RH*hpwl/hmin) * exp(2*vrel)   A_hat = 1.035*sumA
    true  = (1 + 0.5*(hpwl_gap + area_gap)) * exp(2*vrel)   gaps vs the LABEL

The exchange rate between the area and hpwl terms has never been calibrated
against the labels. If it is wrong the selector is systematically off, and
fixing it costs nothing at runtime.

Uses labels for OFFLINE DIAGNOSIS only -- permitted; the 2026-08-05 ruling bans
label-supervised ML, not oracle probes.

  <python> l250_selection.py --sample s1 --nmin 101 --limit 40
"""
import argparse
import math
import os
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--nmin", type=int, default=101)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    sys.argv = ["x"]
    # 🚨 ORDER MATTERS. m67_oos_probe.py:61-63 deletes EVERY ICCAD_* at import
    # time ("shipped defaults only"), so setting the cores before importing it
    # is silently undone: _effective_cores_hi() then reports this box's 32, all
    # four >=40-core tiers switch off, and the pool is 13 instead of 51. The
    # first run of this file measured that 13-profile pool and printed a
    # perfectly plausible table. Set the env AFTER the strip, and assert it.
    import torch
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    import l124_r3_scale as R
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    import optimizer_constructive as oc

    RH = oc._RH
    _np = len(list(oc._pool_indices(120)))
    print("[l250] pool at n=120: {} profiles".format(_np))
    if _np != 51:
        print("!! the >=40-core tiers are OFF -- this is not the shipped pool."
              " Refusing to measure a configuration we do not ship.")
        return 1
    specs = [(ck, fk, L, n) for ck, fk, L, n in m77._specs(a.sample)
             if n >= a.nmin]
    specs.sort(key=lambda t: -t[3])
    if a.limit:
        specs = specs[:a.limit]
    print("[l250] sample {}  {} cases n>={}  @{}c  _RH={}"
          .format(a.sample, len(specs), a.nmin, a.cores, RH))

    byf = {}
    for ck, fk, L, n in specs:
        byf.setdefault(fk, []).append((ck, L, n))

    rows = []
    done = 0
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, L, n in byf[fk]:
            lay = m67._load_case(d, L)
            lay["base"], _ = m67._baseline_official(lay)
            cap = R._capture(oc, lay, "0")          # FLAG="0" -> shipped config
            if len(cap) < 2:
                continue
            idxs = sorted(cap)
            met = [cap[i][1] for i in idxs]
            sumA = sum(max(0.0, float(lay["at"][i])) for i in range(n))
            A_hat = 1.035 * max(sumA, 1e-9)
            hmin = min(m["hpwl"] for m in met) or 1.0
            prox = [(m["area"] / A_hat + RH * m["hpwl"] / hmin)
                    * math.exp(2.0 * m["vrel"]) for m in met]
            true = []
            for i in idxs:
                try:
                    true.append(float(m67._cost(cap[i][0], lay).cost))
                except Exception:
                    true.append(float("inf"))
            k_prox = min(range(len(idxs)), key=lambda k: prox[k])
            k_true = min(range(len(idxs)), key=lambda k: true[k])
            lab = float(m67._cost(lay["tp"], lay).cost)
            rows.append(dict(ck=ck, n=n, npool=len(idxs),
                             picked=true[k_prox], oracle=true[k_true],
                             label=lab, same=(k_prox == k_true),
                             rank_of_pick=sorted(true).index(true[k_prox])))
            done += 1
            if done % 10 == 0:
                print("   {}/{} captured".format(done, len(specs)))

    if not rows:
        print("no cases captured")
        return 1
    W = lambda r: math.exp(r["n"] / 12.0)                        # noqa: E731
    SW = sum(W(r) for r in rows)
    wp = sum(W(r) * r["picked"] for r in rows) / SW
    wo = sum(W(r) * r["oracle"] for r in rows) / SW
    wl = sum(W(r) * r["label"] for r in rows) / SW
    print()
    print("=" * 68)
    print("weighted true cost over {} cases (n>={}, sample {})"
          .format(len(rows), a.nmin, a.sample))
    print("=" * 68)
    print("  proxy pick (what we ship)   {:.6f}".format(wp))
    print("  oracle over the same pool   {:.6f}   SELECTION loss {:+.4f}%"
          .format(wo, 100 * (wp - wo) / wp))
    print("  the LABEL itself            {:.6f}   GENERATION loss {:+.4f}%"
          .format(wl, 100 * (wo - wl) / wo))
    print("  total deficit vs label                       {:+.4f}%"
          .format(100 * (wp - wl) / wp))
    print()
    same = sum(1 for r in rows if r["same"])
    rk = [r["rank_of_pick"] for r in rows]
    print("  proxy picked the true best on {}/{} cases ({:.0f}%)"
          .format(same, len(rows), 100 * same / len(rows)))
    print("  when it did not, its pick ranked (0 = best): median {:.0f}, "
          "p90 {:.0f}, worst {} of ~{} candidates"
          .format(st.median(rk), sorted(rk)[-max(1, len(rk) // 10)], max(rk),
                  int(st.median(r["npool"] for r in rows))))
    print()
    print("  If SELECTION loss is large the fix is free -- a better proxy, no")
    print("  new search. If it is ~0 then CLAUDE.md:488 holds and the whole")
    print("  deficit is GENERATION: the pool does not contain the layouts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
