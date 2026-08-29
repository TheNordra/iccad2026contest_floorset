"""L172d - what runtime budget is actually left, on the 2026-08-23 medians?

Every "is this mechanism affordable" verdict in this ledger was priced against
the 2026-08-19 median table. That table has been replaced and all 100 entries
fell (p50 x0.742). This reprices the budget itself, so the next mechanism is
not shelved or shipped against a number that no longer exists.

Two things this makes explicit that the aggregate hides:
  * our cost-weighted RF is 0.7016 against a hard floor of 0.7000, so there is
    at most 0.23% of score obtainable by going FASTER. Runtime is not a source
    of gain; it is only a constraint.
  * the free seconds are not where the weight is. exp(n/12) puts 78% of the
    weight above n=100, and those are the cases with the least slack.
"""
import math
import statistics as st

import l146_rf_price as L
import l172_depthmap as M

THR = L.THR


def band_table(rows, label):
    print("\n--- {} ---".format(label))
    tot_w = sum(r["w"] for r in rows)
    print("{:>12}{:>8}{:>10}{:>9}{:>9}{:>11}{:>11}"
          .format("band", "cases", "% weight", "slack p50", "min",
                  "free s", "spent s"))
    for lo, hi, lab in ((0, 60, "n<=60"), (60, 90, "60<n<=90"),
                        (90, 105, "90<n<=105"), (105, 999, "n>105")):
        seg = [r for r in rows if lo < r["n"] <= hi]
        if not seg:
            continue
        w = sum(r["w"] for r in seg) / tot_w
        free = sum(max(0.0, THR * r["med"] - r["t"]) for r in seg)
        over = sum(max(0.0, r["t"] - THR * r["med"]) for r in seg)
        print("{:>12}{:>8}{:>9.1f}%{:>9.2f}x{:>9.2f}x{:>10.2f}s{:>10.2f}s"
              .format(lab, len(seg), 100 * w,
                      st.median([r["slack"] for r in seg]),
                      min(r["slack"] for r in seg), free, over))
    free = sum(max(0.0, THR * r["med"] - r["t"]) for r in rows)
    over = sum(max(0.0, r["t"] - THR * r["med"]) for r in rows)
    print("{:>12}{:>8}{:>9.1f}%{:>9.2f}x{:>9.2f}x{:>10.2f}s{:>10.2f}s"
          .format("TOTAL", len(rows), 100.0,
                  st.median([r["slack"] for r in rows]),
                  min(r["slack"] for r in rows), free, over))


def main():
    old, new = L.load(), M.rows_new()
    print(__doc__)
    print("=" * 74)

    num = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3) for r in new)
    raw = sum(r["w"] * r["q"] for r in new)
    print("beta package on the NEW table: raw {:.7f}  graded {:.7f}  "
          "cost-weighted RF {:.5f}"
          .format(raw / sum(r["w"] for r in new),
                  num / sum(r["w"] for r in new),
                  num / raw))
    print("the floor is 0.70000, so being faster is worth at most {:+.3f}% "
          "of score.".format(100 * (num / raw - 0.7) / (num / raw)))

    band_table(old, "OLD medians (2026-08-19) -- what every prior verdict used")
    band_table(new, "NEW medians (2026-08-23) -- what is actually true")

    print("\n--- uniform slowdown, priced on the NEW table ---")
    base = L._total(new, lambda r: 1.0)
    print("{:>9}{:>13}{:>11}{:>12}".format("uniform", "total", "RF cost",
                                           "off floor"))
    for s in (1.02, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.0):
        t = L._total(new, lambda r, s=s: s)
        off = sum(1 for r in new if (s * r["t"] / r["med"]) ** 0.3 > 0.7)
        print("{:>8.2f}x{:>13.6f}{:>+10.4f}%{:>9}/100"
              .format(s, t, 100 * (base - t) / base, off))
    print("\nfor comparison, the OLD table read 1.25x -0.09%, 1.5x -0.70%.")

    print("\n--- what a flat +dt per case costs, NEW table ---")
    print("{:>10}{:>12}{:>12}".format("+dt/case", "RF cost", "cumulative s"))
    for dt in (0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
        c = L.price_seconds(lambda n, dt=dt: dt, 0.0, rows=new)["rf_cost"]
        print("{:>9.2f}s{:>+11.4f}%{:>11.1f}s".format(dt, c, dt * 100))
    print("\nA mechanism must clear its own RF cost from this table, not from")
    print("the 19.79s 'free budget' the 2026-08-19 medians implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
