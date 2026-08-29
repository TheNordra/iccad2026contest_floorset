"""L277 -- what is the violation axis WORTH on the graded shape, as an upper bound?

L275 moved the target to the corpus the score is computed on. There, vrel is
0.0141 and driving it to zero is worth **-2.81 %** -- 15 % of the total headroom,
and the one axis the L250-L274 arc never touched. L140 audited violations on OOS
(the wrong corpus, and it reported boundary as "306/307 placeable but not
placed"). This runs the SAME audit (`l135_soft_audit.audit_case`) on the in-set
and prices what is actually removable.

Two filters decide removability, and both matter:

  GEOMETRY   the slide path / gap corridor has to be clear, or the block cannot
             reach the edge it must touch without displacing something else;
  CONSTRAINT a `preplaced` block's position is a HARD constraint. Its boundary
             violation is unsatisfiable from outside the placer -- moving it
             would break feasibility, not fix a violation. Counting those as
             recoverable is the obvious way to manufacture a large fake prize.

So the prize is over `CLEAR AND soft` only. Everything else is reported too, so
the gap between the headline and the honest number stays visible.

🚨 This is an UPPER bound and should be read as one: it assumes every removable
violation is removed at zero cost to hpwl and area, which is exactly the
assumption L267_L269 §2.3 showed to be false for area (the exchange rate). A
mechanism that removes violations by moving blocks will pay somewhere.

  <python> l277_vio_prize.py results_L274_base_48c.json
"""
import json
import math
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import ContestEvaluator          # noqa: E402
from l135_soft_audit import audit_case                   # noqa: E402


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results_L274_base_48c.json"
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    res = {r["test_id"]: r for r in json.load(open(src))["test_results"]}

    rows = []
    tb = tg = 0
    cb = cg = 0          # CLEAR and soft
    pre_cnt = blocked_cnt = 0
    for idx in sorted(res):
        r = res[idx]
        P = r.get("positions")
        if not P:
            continue
        s = ev.dataset[idx]
        at, _b2b, _p2b, _pins, cons = s["input"]
        n = int((at != -1).sum().item())
        g, b, m, tgc, tbc, tmc, _fm = audit_case(idx, P, cons, at, n)
        tb += tbc
        tg += tgc
        # boundary: (idx, i, side, dist, ratio, "blocked"|"CLEAR", kind)
        nb = sum(1 for x in b if x[5] == "CLEAR" and x[6] == "soft")
        pre_cnt += sum(1 for x in b if x[6] != "soft")
        blocked_cnt += sum(1 for x in b if x[5] != "CLEAR")
        # grouping: (idx, grp, size, gap, ratio, "clear"|"blocked"|"?")
        ng = sum(1 for x in g if str(x[5]).lower().startswith("clear"))
        cb += nb
        cg += ng
        V = tgc + tbc + tmc
        vrel = float(r.get("violations_relative", 0.0))
        nsoft = (V / vrel) if vrel > 1e-12 else 0.0
        rows.append(dict(idx=idx, n=int(r["block_count"]), cost=float(r["cost"]),
                         V=V, vrel=vrel, nsoft=nsoft, rem=nb + ng))

    W = lambda n: math.exp(n / 12.0)
    sw = sum(W(r["n"]) for r in rows)
    base = sum(W(r["n"]) * r["cost"] for r in rows) / sw

    def after(pick):
        t = 0.0
        for r in rows:
            k = pick(r)
            if k <= 0 or r["nsoft"] <= 0:
                t += W(r["n"]) * r["cost"]
                continue
            dv = -min(k, r["V"]) / r["nsoft"]
            t += W(r["n"]) * r["cost"] * math.exp(2.0 * dv)
        return t / sw

    print("in-set 100, current shipped code -- soft-violation inventory")
    print("  boundary {}   grouping {}   (MIB 0)".format(tb, tg))
    print("    of the {} boundary: {} are PREPLACED (hard constraint, unfixable"
          " from outside the placer), {} have a BLOCKED slide path".format(
              tb, pre_cnt, blocked_cnt))
    print("  removable = CLEAR and soft:  boundary {}   grouping {}   total {}"
          .format(cb, cg, cb + cg))
    print()
    print("  weighted base cost {:.6f}".format(base))
    for lab, f in (("remove ALL violations (the -2.81% figure)", lambda r: r["V"]),
                   ("remove every CLEAR+soft one (honest upper bound)", lambda r: r["rem"]),
                   ("remove ONE per case where possible", lambda r: min(1, r["rem"]))):
        t = after(f)
        print("    {:48s} {:.6f}   {:+.4f}%".format(lab, t, 100 * (t - base) / base))
    print()
    hot = sorted([r for r in rows if r["rem"] > 0],
                 key=lambda r: -W(r["n"]) * r["cost"] * (1 - math.exp(-2.0 * r["rem"] / max(r["nsoft"], 1e-9))))
    print("  cases carrying the removable prize (top 8):")
    print("    {:>5s} {:>5s} {:>6s} {:>6s} {:>7s} {:>9s}".format("case", "n", "V", "rem", "nsoft", "worth"))
    for r in hot[:8]:
        w = 100 * W(r["n"]) * r["cost"] * (1 - math.exp(-2.0 * r["rem"] / max(r["nsoft"], 1e-9))) / (sw * base)
        print("    {:5d} {:5d} {:6d} {:6d} {:7.1f} {:+8.4f}%".format(
            r["idx"], r["n"], r["V"], r["rem"], r["nsoft"], -w))
    return 0


if __name__ == "__main__":
    sys.exit(main())
