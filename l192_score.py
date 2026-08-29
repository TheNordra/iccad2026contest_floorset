"""L192 verdict - does thinning the pool survive out of sample?

L189 put `thin pool + LP k=1` as the only configuration that beats beta without
betting on route A, and it gets there by giving up 1.390% of IN-SET quality to
save 7.4 grader-seconds. This measures that 1.390% on two disjoint 240-case
held-out samples.

WHY THE IN-SET NUMBER IS EXPECTED TO BE OPTIMISTIC. The thin arm turns off three
mechanisms at once, and one of them -- the L124 twins -- is KNOWN to be worth
0.0000% in set and +0.6664% out of sample (L186, and it moved 0 of 100 in-set
cases while moving 106 of 240 held out). So the in-set deficit understates the
real cost of thinning by at least that much.

The runtime side is already measured and does not change here (L188/L189, one
box, route A off, transported by t_beta(n) * w(n)/w_m73(n)):

    full pool + LP k=1    81.2 grader-s     RF -9.209%   vs beta
    thin pool + LP k=1    73.8 grader-s     RF -4.793%
                          ------------- thinning buys +4.416pp of RF
"""
import json
import math
from pathlib import Path

DIR = Path(__file__).parent
RF_FULL, RF_THIN = -9.209, -4.793          # L189, route A neutral
RF_FULL_RA, RF_THIN_RA = -1.551, -0.753    # L189, route A 0.68x
BETA = 0.9265861161320369
RANKS = [(1, 0.8586322662042342), (2, 0.888187391), (3, 0.8993286931994098),
         (4, 0.9265861161320369), (5, 0.9507093062865333),
         (6, 0.9552271810705998), (7, 0.9638548902636931)]


def rank_of(t):
    return sum(1 for _, x in RANKS if x < t - 1e-9) + 1


def wcost(fn):
    d = json.load(open(DIR / fn))["test_results"]
    w = lambda r: math.exp(r["n"] / 12.0)                          # noqa: E731
    return (sum(w(r) * r["cost"] for r in d) / sum(w(r) for r in d),
            len(d), sum(1 for r in d if r.get("feasible")))


def main():
    print(__doc__)
    print("=" * 76)
    deltas = []
    for s in ("s1", "s2"):
        try:
            qf, n, ff = wcost("l192_{}_full.json".format(s))
            qt, _, ft = wcost("l192_{}_thin.json".format(s))
        except FileNotFoundError:
            print("{}: not finished".format(s))
            continue
        # qt/qf are COSTS: higher is worse. Report the signed quality
        # delta of thin vs full, so negative = thin is worse.
        d = -100 * (qt - qf) / qf
        deltas.append(d)
        print("{}  full {:.6f} ({}/{} feasible)   thin {:.6f} ({}/{})"
              .format(s, qf, ff, n, qt, ft, n))
        print("     thin vs full: {:+.4f}% of quality   (in set: -1.390%)"
              .format(d))
    if len(deltas) < 2:
        return 1
    mean = sum(deltas) / len(deltas)
    print("\nmean OOS quality cost of thinning: {:+.4f}%".format(mean))
    print("                    in-set said:    -1.390%")

    print("\n{:>34}{:>11}{:>11}{:>8}".format("", "NET vs beta", "graded", "rank"))
    for tag, rf_f, rf_t in (("route A neutral", RF_FULL, RF_THIN),
                            ("route A 0.68x", RF_FULL_RA, RF_THIN_RA)):
        for lbl, rf, q in (("full pool + LP k=1", rf_f, 6.450),
                           ("thin pool + LP k=1", rf_t, 6.450 + mean)):
            net = q + rf
            g = BETA * (1 - net / 100.0)
            print("{:>18} {:<16}{:>+10.3f}%{:>11.5f}{:>8}"
                  .format(tag, lbl, net, g, rank_of(g)))
    print("\nquality for the thin arm is the full arm's +6.450% plus the")
    print("MEASURED OOS delta above, not the in-set one. RF is unchanged --")
    print("it was measured separately and does not depend on the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
