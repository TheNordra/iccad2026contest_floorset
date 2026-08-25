"""L238 -- score the in-set gates for the L234 + L235 package.

Written fresh rather than extending l199_verdict.py because two of that file's
gates encode a package that no longer exists (G11 asserts the mid band is "6";
G2's anchors predate the gate table AND the REFINE bands, which L216/L229
already had to split twice). Re-pointing them a third time would leave a file
where most of the assertions are about history. The gates that still bite are
carried over verbatim.

  <python> l238_verdict.py
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).parent
ARMS = ("det1", "det2", "gateoff", "k1", "l147off", "hboff", "lpoff",
        "refine4", "refinemid6", "pooldropon")


def load(tag):
    f = DIR / "results_L238_{}.json".format(tag)
    if not f.exists():
        return None
    return {r["block_count"]: r for r in json.load(open(f))["test_results"]}


def loadf(name):
    f = DIR / name
    if not f.exists():
        return None
    return {r["block_count"]: r for r in json.load(open(f))["test_results"]}


def same(a, b, key="cost"):
    return sum(1 for n in a if n in b and a[n][key] == b[n][key])


def movers(a, b):
    return sorted(n for n in a if n in b and a[n]["cost"] != b[n]["cost"])


def gate_table():
    import re
    s = (DIR / "optimizer_constructive.py").read_text(encoding="utf-8")
    return eval(re.search(r"^_L196_LPGATE = \{.*?^\}", s, re.S | re.M)
                .group(0).split("=", 1)[1])


def stats_set(tag):
    f = DIR / "l238_{}_stats.txt".format(tag)
    if not f.exists():
        return set()
    return {int(l.split()[0]) for l in f.read_text().splitlines() if l.split()}


def main():
    A = {t: load(t) for t in ARMS}
    missing = [t for t in ARMS if A[t] is None]
    if missing:
        print("missing arms:", missing)
        return 1
    W = lambda n: math.exp(n / 12.0)                             # noqa: E731
    SW = sum(W(n) for n in A["det1"])
    wq = lambda d: sum(W(n) * d[n]["cost"] for n in d) / SW      # noqa: E731
    ok = True
    print("=" * 72)
    print("L238 -- in-set gates, L234 (mid REFINE 2, gate 71 on) + L235 (LP)")
    print("=" * 72)

    c = same(A["det1"], A["det2"])
    p = same(A["det1"], A["det2"], "positions")
    print("G1  determinism      cost {}/100  positions {}/100   {}"
          .format(c, p, "PASS" if c == p == 100 else "FAIL"))
    ok &= (c == 100 and p == 100)

    post = loadf("results_L237_post.json")
    if post:
        c = same(A["det1"], post)
        p = same(A["det1"], post, "positions")
        print("G0  chain of custody vs results_L237_post.json  cost {}/100  "
              "positions {}/100   {}".format(c, p,
                                             "PASS" if c == p == 100 else "FAIL"))
        ok &= (c == 100 and p == 100)
    base = loadf("results_L237_base.json")
    if base and post:
        c = same(base, post)
        p = same(base, post, "positions")
        print("G13 L235 rewrite INVISIBLE (pre vs post, same tree otherwise)  "
              "cost {}/100  positions {}/100   {}"
              .format(c, p, "PASS" if c == p == 100 else "FAIL"))
        ok &= (c == 100 and p == 100)

    g = gate_table()
    want = {n for n, v in g.items() if v}
    got, allg, none = stats_set("det1"), stats_set("gateoff"), stats_set("lpoff")
    good = got == want and allg == set(g) and not none
    print("G3  gate fired       default {} == table 1-set {}   LP_GATE=0 {} == "
          "all {}   SHAPE_LP=0 {} == 0 {}   {}"
          .format(len(got), got == want, len(allg), allg == set(g), len(none),
                  not none, "PASS" if good else "FAIL"))
    if not good:
        print("     ran-but-must-not {}   must-but-did-not {}"
              .format(sorted(got - want)[:12], sorted(want - got)[:12]))
    ok &= good
    print("     gate is on for {} block counts above n=100 of 20"
          .format(sum(1 for n in want if n > 100)))

    c = same(A["det1"], A["k1"])
    p = same(A["det1"], A["k1"], "positions")
    print("G4  depth map flat   k1 vs det1 cost {}/100 positions {}/100   {}"
          .format(c, p, "PASS" if c == p == 100 else "FAIL"))
    ok &= (c == 100 and p == 100)

    bad = [t for t in ARMS if sum(1 for n in A[t] if A[t][n]["is_feasible"]) != 100]
    print("G5  feasibility      100/100 in all {} arms   {}"
          .format(len(ARMS), "PASS" if not bad else "FAIL " + str(bad)))
    ok &= not bad

    q0, q1 = wq(A["lpoff"]), wq(A["det1"])
    mv = movers(A["lpoff"], A["det1"])
    worse = sum(1 for n in mv if A["det1"][n]["cost"] > A["lpoff"][n]["cost"])
    print("G7  LP value         {:+.4f}%   {} moved ({} better / {} worse)   {}"
          .format(100 * (q0 - q1) / q0, len(mv), len(mv) - worse, worse,
                  "PASS" if q1 < q0 else "FAIL"))
    ok &= q1 < q0

    mv = movers(A["det1"], A["refine4"])
    outside = [n for n in mv if n <= 100]
    print("G11 REFINE heavy     kill switch moves {} case(s), all above n=100: "
          "{}   {}".format(len(mv), not outside,
                           "PASS" if mv and not outside else "FAIL"))
    if outside:
        print("     leaked into the mid/light band at {}".format(outside[:12]))
    ok &= bool(mv) and not outside

    mv = movers(A["det1"], A["refinemid6"])
    outside = [n for n in mv if not (60 < n <= 100)]
    print("G12 REFINE mid       kill switch moves {} case(s), all in 60<n<=100:"
          " {}   {}".format(len(mv), not outside,
                            "PASS" if mv and not outside else "FAIL"))
    if outside:
        print("     leaked outside the mid band at {}".format(outside[:12]))
    ok &= bool(mv) and not outside

    mv = movers(A["det1"], A["pooldropon"])
    print("G10 pool drop        OFF by default; ICCAD_L211_POOLDROP=1 still "
          "moves {} case(s)   {}".format(len(mv), "PASS" if mv else "FAIL"))
    ok &= bool(mv)

    print("-" * 72)
    print("in-set weighted cost: {:.9f}   (LP off {:.9f})".format(q1, q0))
    print("VERDICT: {}".format("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
