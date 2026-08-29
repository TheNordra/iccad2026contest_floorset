"""L353 -- which HALF of hpwl is our deficit in: block-to-block, or block-to-pin?

WHY NOBODY HAS ASKED. The contest's wirelength is `calculate_hpwl_b2b + calculate_hpwl_p2b`
and every analysis in this project has treated it as one number. L349 decomposed the score
into hpwl / area / violations and found hpwl is the only axis with room; it never split
hpwl into its two channels.

WHY IT MIGHT MATTER. Three measured facts line up:
  * L327: the pin channel carries a FAR sharper positional signal than b2b -- a
    pin-connected block sits at u_pin = 0.0514 of the maximum pin distance against a
    0.4341 baseline, 8.4x sharper than the b2b law.
  * Q&A A23/A24: the hidden set uses the SAME terminal-connectivity generation, with noise
    on shapes and placements only. So the pin law transfers.
  * The packer's step score sums b2b and p2b into ONE `wire` term weighted only by each
    edge's own weight, and `ICCAD_WIRE_MULT` scales the SUM. Checked against
    m79_knob_cloud_probe: the 512-vector cloud samples WIRE_MULT and BFS_PIN, but
    **nothing that re-balances p2b against b2b.** So this axis was swept by neither M80's
    coefficients nor L351's forms.

WHAT THIS MEASURES. Per case, our hpwl and the label's, split by channel. If our excess is
disproportionately in one channel, the deficit is located; if the split matches the label's,
there is nothing here and the axis closes cheaply.

Offline oracle probe: reads labels for diagnosis only. Nothing shipped.
"""
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
sys.path.insert(0, str(DIR / "iccad2026contest"))
from iccad2026_evaluate import calculate_hpwl_b2b, calculate_hpwl_p2b  # noqa: E402
from l342_strictcost import case  # noqa: E402

OURS = DIR / "l313_win48_rfsafe.json"


def main():
    pos_by_n = {}
    for t in json.load(open(OURS))["test_results"]:
        if t.get("positions"):
            pos_by_n[t["block_count"]] = [tuple(map(float, r)) for r in t["positions"]]

    rows = []
    for n in range(21, 121):
        try:
            c = case(n)
        except Exception:
            continue
        ours = pos_by_n.get(c["n"])
        if not ours:
            continue
        ob = calculate_hpwl_b2b(ours, c["b2b"])
        op = calculate_hpwl_p2b(ours, c["p2b"], c["pins"])
        lb = calculate_hpwl_b2b(c["tp"], c["b2b"])
        lp = calculate_hpwl_p2b(c["tp"], c["p2b"], c["pins"])
        rows.append(dict(n=c["n"], w=math.exp(c["n"] / 12.0),
                         ob=ob, op=op, lb=lb, lp=lp))

    W = sum(r["w"] for r in rows)

    def wm(f):
        return sum(r["w"] * f(r) for r in rows) / W

    OB, OP, LB, LP = wm(lambda r: r["ob"]), wm(lambda r: r["op"]), \
        wm(lambda r: r["lb"]), wm(lambda r: r["lp"])
    print("== L353: hpwl deficit by channel, %d validation cases ==" % len(rows))
    print()
    print("   %-10s %14s %14s %14s %10s"
          % ("", "b2b", "p2b", "total", "p2b share"))
    print("   %-10s %14.1f %14.1f %14.1f %9.1f %%"
          % ("label", LB, LP, LB + LP, 100 * LP / (LB + LP)))
    print("   %-10s %14.1f %14.1f %14.1f %9.1f %%"
          % ("ours", OB, OP, OB + OP, 100 * OP / (OB + OP)))
    print()
    print("   excess over the label:")
    print("     b2b    %+10.1f   = %+7.2f %% of the label's b2b" % (OB - LB, 100 * (OB / LB - 1)))
    print("     p2b    %+10.1f   = %+7.2f %% of the label's p2b" % (OP - LP, 100 * (OP / LP - 1)))
    print("     total  %+10.1f   = %+7.2f %%"
          % ((OB + OP) - (LB + LP), 100 * ((OB + OP) / (LB + LP) - 1)))
    tot_ex = (OB - LB) + (OP - LP)
    if abs(tot_ex) > 1e-9:
        print()
        print("   *** SHARE OF OUR TOTAL hpwl EXCESS:  b2b %.1f %%   p2b %.1f %% ***"
              % (100 * (OB - LB) / tot_ex, 100 * (OP - LP) / tot_ex))
        print("   (against a p2b share of the label's own wirelength of %.1f %%)"
              % (100 * LP / (LB + LP)))
    print()

    # per-band, because the score weight is all in the heavy band
    print("   by band (weighted within band):")
    print("   %10s %6s %11s %11s %12s"
          % ("band", "cases", "b2b excess", "p2b excess", "p2b share of excess"))
    for lo, hi in ((21, 50), (51, 80), (81, 100), (101, 120)):
        sel = [r for r in rows if lo <= r["n"] <= hi]
        if not sel:
            continue
        sw = sum(r["w"] for r in sel)

        def q(f):
            return sum(r["w"] * f(r) for r in sel) / sw
        eb = q(lambda r: r["ob"]) - q(lambda r: r["lb"])
        ep = q(lambda r: r["op"]) - q(lambda r: r["lp"])
        tt = eb + ep
        print("   %10s %6d %+11.1f %+11.1f %11.1f %%"
              % ("%d-%d" % (lo, hi), len(sel), eb, ep,
                 100 * ep / tt if abs(tt) > 1e-9 else float("nan")))
    print()
    nb = sum(1 for r in rows if r["op"] > r["lp"])
    print("   cases where our p2b is worse than the label's: %d/%d" % (nb, len(rows)))
    nb2 = sum(1 for r in rows if r["ob"] > r["lb"])
    print("   cases where our b2b is worse than the label's: %d/%d" % (nb2, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
