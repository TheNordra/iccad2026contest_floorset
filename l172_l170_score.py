"""L172j - score the L170 hb-baseline OOS arms the other agent produced,
and ADJUDICATE which depth map each of them actually ran.

Why the adjudication is needed: `l170c_oos.sh` ran s1 from 10:04:21 to
10:23:18 and started s2 at 10:23:29. `optimizer_constructive.py` was edited at
10:24:26 (this session, replacing `_L157_DEPTH`). s1 is unambiguously the old
map. s2 started 57 seconds before the edit, so it is only safe if the driver
imported the module before then -- likely, since the import precedes the
900-file index scan, but "likely" is not a measurement.

It is measurable. The two maps differ by ~0.43% of weighted cost on s2
(old map +0.8799% vs the k=1 anchor, x0.90 map +0.4452%), which is far larger
than the hb predictor's own in-set effect (+0.0772%). So the arm's weighted
cost identifies the map it ran.

    <python> l172_l170_score.py
"""
import json
import math
from pathlib import Path

import l172_depthmap as M

DIR = Path(__file__).parent


def wcost(path):
    j = json.load(open(path))
    rs = j["test_results"]
    num = sum(math.exp(r["n"] / 12.0) * r["cost"] for r in rs)
    den = sum(math.exp(r["n"] / 12.0) for r in rs)
    return num / den, len(rs), sum(1 for r in rs if r.get("feasible"))


def mixed(dmap, sample):
    """weighted cost of the arm-mixed control under `dmap`."""
    A = {1: "l147_oos_{}_r15g.json".format(sample),
         2: "l157_oos_{}_k2.json".format(sample),
         3: "l165_oos_{}_k3.json".format(sample)}
    Q = {k: {r["test_id"]: r for r in json.load(open(DIR / fn))["test_results"]}
         for k, fn in A.items()}
    ids = sorted(set(Q[1]) & set(Q[2]) & set(Q[3]))
    w = lambda i: math.exp(Q[1][i]["n"] / 12.0)                    # noqa: E731
    num = sum(w(i) * Q[dmap.get(Q[1][i]["n"], 1)][i]["cost"] for i in ids)
    return num / sum(w(i) for i in ids)


def main():
    x090 = {int(k): v for k, v in
            json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    print(__doc__)
    print("=" * 74)
    for s in ("s1", "s2"):
        f = DIR / "l170_oos_{}_hb.json".format(s)
        if not f.exists():
            print("{}: not finished yet".format(s))
            continue
        c, n, fe = wcost(f)
        c_old = mixed(M.SHIPPED, s)
        c_new = mixed(x090, s)
        c_k1 = mixed({k: 1 for k in x090}, s)
        which = "OLD map" if abs(c - c_old) < abs(c - c_new) else "x0.90 map"
        log = DIR / "l170_{}_hb.log".format(s)
        sa = log.read_text(errors="ignore").count("SA fallback") if log.exists() else -1
        print("\n{}  arm cost {:.6f}   {} cases, {} feasible, {} SA fallbacks"
              .format(s, c, n, fe, sa))
        print("   arm-mixed controls:  k=1 {:.6f}   OLD map {:.6f}   "
              "x0.90 map {:.6f}".format(c_k1, c_old, c_new))
        print("   nearest control -> the arm ran the {}".format(which))
        ctl = c_old if which == "OLD map" else c_new
        print("   hb predictor delta vs that control: {:+.4f}%"
              .format(100 * (ctl - c) / ctl))
        print("   (a clean read needs |delta| well under the {:.4f}% that "
              "separates the two maps)".format(100 * abs(c_old - c_new) / c_old))
    print("\nNOTE: the arm-mixed control is exact for a GATED run of that map;")
    print("the hb arm also carries ICCAD_LP_HB_PRED, so the residual after")
    print("subtracting the control is the predictor's own effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
