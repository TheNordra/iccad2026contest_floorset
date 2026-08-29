"""L172e - the 18 cases that are already OFF the RF floor on the new table.

Our cost-weighted RF is 0.70161 against a hard floor of 0.70000. All of that
0.161% sits on a handful of cases whose own runtime already exceeds
0.3046 * median BEFORE any shape LP is added -- the beta package had no LP at
all. They cannot be fixed by spending less LP; only by making the pool itself
faster on those cases, or by accepting the cost.

This lists them so the next runtime decision is aimed rather than uniform.
"""
import math

import l146_rf_price as L
import l172_depthmap as M


def main():
    rows = M.rows_new()
    W = sum(r["w"] for r in rows)
    base = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
               for r in rows) / W
    floor = sum(r["w"] * r["q"] * 0.7 for r in rows) / W
    print(__doc__)
    print("=" * 74)
    print("graded {:.7f}   if every case were at the floor {:.7f}   "
          "gap {:+.4f}%".format(base, floor, 100 * (base - floor) / base))

    over = sorted((r for r in rows if r["slack"] < 1.0),
                  key=lambda r: -r["w"] * r["q"]
                  * (max(0.7, (r["t"] / r["med"]) ** 0.3) - 0.7))
    print("\n{:>5}{:>5}{:>9}{:>9}{:>8}{:>10}{:>12}{:>11}"
          .format("case", "n", "our t", "median", "slack", "RF now",
                  "excess s", "score cost"))
    tot = 0.0
    for r in over:
        rf = max(0.7, (r["t"] / r["med"]) ** 0.3)
        cost = 100 * r["w"] * r["q"] * (rf - 0.7) / (base * W)
        tot += cost
        print("{:>5}{:>5}{:>8.3f}s{:>8.3f}s{:>8.2f}{:>10.4f}{:>11.3f}s"
              "{:>+10.4f}%".format(r["i"], r["n"], r["t"], r["med"],
                                   r["slack"], rf,
                                   r["t"] - L.THR * r["med"], cost))
    print("{:>54}{:>11.3f}s{:>+10.4f}%"
          .format("TOTAL", sum(r["t"] - L.THR * r["med"] for r in over), tot))

    print("\nweight concentration: the {} over-floor cases carry {:.1f}% of the"
          " corpus weight".format(len(over), 100 * sum(r["w"] for r in over) / W))
    heavy = [r for r in over if r["n"] > 100]
    print("of those, {} are n>100 and carry {:.1f}% of the corpus weight"
          .format(len(heavy), 100 * sum(r["w"] for r in heavy) / W))
    print("\nSo the whole obtainable runtime gain is {:+.4f}% and {:.0f}% of it"
          " sits on {} cases.".format(tot,
                                      100 * sum(c for c in [
                                          100 * r["w"] * r["q"]
                                          * (max(0.7, (r["t"] / r["med"]) ** 0.3) - 0.7)
                                          / (base * W) for r in over[:3]]) / tot,
                                      3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
