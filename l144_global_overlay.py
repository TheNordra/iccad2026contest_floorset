"""L144 - price the OTHER deployment form of ICCAD_BND_ABUT: global overlay.

The twin screen (l124_r3_scale, cache l144_twin_cache.pkl) APPENDS ON copies of
K profiles and lets the proxy arbitrate per case. Its cross-sample transfer came
out at 27% (s1-picked K=8 is worth +0.3729% on s1 and +0.0796% on s2), and the
winner tally is almost all 1s -- L127's noise-fitting signature.

But the same cache holds the ON capture of EVERY profile, so the M71-style form
is free to price: turn the flag on for the whole pool. Pool size is unchanged, so
there is no RF cost and no M67-E free-restore budget question -- the two things
that make the twin form expensive.

Three numbers per sample, all with `_solve_impl`'s exact selection (same A_hat,
same pool-wide hmin, same _RH, same proxy):

  OFF        today's pool                                  (baseline)
  GLOBAL ON  every profile carries the flag                (deployable, RF-free)
  ORACLE     per case, the better of the two portfolios     (ceiling of any gate)

L123 -> L124 is the precedent for why all three are needed: global coverage was
+0.6486% on one sample and -0.3730% on the other (sign flip) while the per-case
oracle was positive on both. Sign agreement across s1 and s2 is the gate.

READ-ONLY. Reads the L144 twin cache only; writes nothing.

  <python> -u l144_global_overlay.py --sample s1
"""
import argparse
import collections
import math
import os
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

import torch                                                        # noqa: E402

import m67_oos_probe as m67                                         # noqa: E402
import l124_r3_scale as R3                                          # noqa: E402


def _pick(cap, at, n, oc):
    """_solve_impl's selection over ONE capture dict {pool_index: (pos, metrics)}."""
    if not cap:
        return None
    sumA = sum(max(0.0, float(at[i])) for i in range(n))
    A_hat = 1.035 * max(sumA, 1e-9)
    hmin = min(m["hpwl"] for _p, m in cap.values()) or 1.0
    best, bp = None, float("inf")
    for pos, m in cap.values():
        px = (m["area"] / A_hat + oc._RH * m["hpwl"] / hmin) * math.exp(2.0 * m["vrel"])
        if px < bp:
            bp, best = px, pos
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--cache", default="l144_twin_cache.pkl")
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    import optimizer_constructive as oc

    C = pickle.load(open(_DIR / a.cache, "rb"))
    rows = [(k[1], v) for k, v in C.items() if k[0] == a.sample]
    if not rows:
        raise SystemExit(f"no {a.sample} records in {a.cache}")

    byf = collections.defaultdict(list)
    for ck, v in rows:
        byf[v["fk"]].append((ck, v))
    lays = {}
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, v in byf[fk]:
            lay = m67._load_case(d, v["L"])
            lay["base"], _ = m67._baseline_official(lay)
            lays[ck] = lay

    per = {}
    for ck, v in rows:
        lay = lays[ck]
        c_off = float(m67._cost(_pick(v["cap"]["0"], lay["at"], lay["n"], oc),
                                lay).cost)
        c_on = float(m67._cost(_pick(v["cap"]["1"], lay["at"], lay["n"], oc),
                               lay).cost)
        per[ck] = (v["n"], c_off, c_on)

    def total(sel):
        byn = collections.defaultdict(list)
        for ck, (n, o, x) in per.items():
            byn[n].append(sel(o, x))
        num = den = 0.0
        for n, vv in byn.items():
            w = math.exp(n / 12.0)
            num += w * (sum(vv) / len(vv))
            den += w
        return num / den

    base = total(lambda o, x: o)
    glob = total(lambda o, x: x)
    orac = total(lambda o, x: min(o, x))
    better = sum(1 for _n, o, x in per.values() if x < o - 1e-12)
    worse = sum(1 for _n, o, x in per.values() if x > o + 1e-12)
    same = len(per) - better - worse

    print(f"\n=== L144 global overlay, ICCAD_BND_ABUT, {a.sample}, "
          f"{len(per)} cases @{a.cores}c ===\n")
    print(f"  OFF   (today)      {base:.6f}")
    print(f"  GLOBAL ON          {glob:.6f}   {100 * (1 - glob / base):+.4f}%"
          f"   pool unchanged -> dRF = 0")
    print(f"  per-case ORACLE    {orac:.6f}   {100 * (1 - orac / base):+.4f}%"
          f"   (ceiling of any gate)")
    print(f"  cases better/worse/same   {better} / {worse} / {same}")
    heavy = {k: v for k, v in per.items() if v[0] > 110}
    if heavy:
        hb = sum(math.exp(n / 12) * o for n, o, _x in heavy.values())
        hn = sum(math.exp(n / 12) * x for n, _o, x in heavy.values())
        print(f"  n>110 subset ({len(heavy)} cases)  "
              f"{100 * (1 - hn / hb):+.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
