"""L213 score - does the pool drop survive out of sample?

In set, k=8 costs -0.1242% of quality for -5.50% of case wall. The ledger's
objection is specific and this file is the only thing that can answer it: the
drop table is keyed on BLOCK COUNT and was fitted on the one case per block
count that the in-set corpus contains, so its in-set quality cost is measured on
the same floorplans it was built from. Each OOS sample has 2-4 DIFFERENT
floorplans per block count.

Scored two ways, because they can disagree and the disagreement is the finding:

  per-case   every one of the 240 cases weighted exp(n/12) -- L196's scorer,
             comparable with every OOS number in this ledger.
  per-n      cases averaged within a block count first, then weighted by the
             graded corpus's own weights. This is the corpus the RUNTIME side
             is defined on (100 cases, one per block count), so it is the
             scorer that matches how the wall saving was priced.

The wall saving does not need re-measuring OOS: it is a property of the table
and the durations, both already measured, and it is reported here only so the
NET column means something.

  <python> l213_score.py
"""
import json
import math
from pathlib import Path

DIR = Path(__file__).parent
WALL_CUT_K8 = -5.50          # % of case wall, from l212_curve (profile x0.71)


def load(fn):
    f = DIR / fn
    if not f.exists():
        return None
    return {r["test_id"]: r for r in json.load(open(f))["test_results"]}


def per_case(a, b, ids):
    w = lambda i: math.exp(a[i]["n"] / 12.0)                   # noqa: E731
    sw = sum(w(i) for i in ids)
    qa = sum(w(i) * a[i]["cost"] for i in ids) / sw
    qb = sum(w(i) * b[i]["cost"] for i in ids) / sw
    return 100 * (qa - qb) / qa


def per_n(a, b, ids, rows):
    sa, sb, cnt = {}, {}, {}
    for i in ids:
        n = a[i]["n"]
        sa[n] = sa.get(n, 0.0) + a[i]["cost"]
        sb[n] = sb.get(n, 0.0) + b[i]["cost"]
        cnt[n] = cnt.get(n, 0) + 1
    ns = [n for n in cnt if n in rows]
    sw = sum(rows[n] for n in ns)
    qa = sum(rows[n] * sa[n] / cnt[n] for n in ns) / sw
    qb = sum(rows[n] * sb[n] / cnt[n] for n in ns) / sw
    return 100 * (qa - qb) / qa


def main():
    import l172_depthmap as M
    rows = {r["n"]: r["w"] for r in M.rows_new()}
    print("=" * 78)
    print("L213 -- pool drop k=8, OUT OF SAMPLE")
    print("=" * 78)
    print("{:>8}{:>7}{:>14}{:>14}{:>10}{:>10}"
          .format("sample", "cases", "quality/case", "quality/n", "moved",
                  "worse"))
    print("-" * 78)
    qs = []
    for s in ("s1", "s2"):
        a, b = load("l213_{}_base.json".format(s)), load("l213_{}_k8.json".format(s))
        if not (a and b):
            print("{:>8}   not run".format(s))
            continue
        ids = sorted(set(a) & set(b))
        qc, qn = per_case(a, b, ids), per_n(a, b, ids, rows)
        mv = sum(1 for i in ids if a[i]["cost"] != b[i]["cost"])
        ws = sum(1 for i in ids if b[i]["cost"] > a[i]["cost"] + 1e-12)
        nf = sum(1 for i in ids if not b[i].get("feasible", True))
        qs.append(qn)
        print("{:>8}{:>7}{:>+13.4f}%{:>+13.4f}%{:>10}{:>10}{}"
              .format(s, len(ids), qc, qn, mv, ws,
                      "  !! {} INFEASIBLE".format(nf) if nf else ""))
    print("-" * 78)
    if len(qs) == 2:
        mean = sum(qs) / 2
        print("mean OOS quality (per-n): {:+.4f}%".format(mean))
        print("in-set was              : -0.1242%")
        ratio = mean / -0.1242 if mean else 0.0
        print("OOS / in-set            : {:.2f}x   ({})"
              .format(ratio,
                      "amplified, as the thin pool did (1.9x)" if ratio > 1.3
                      else "held" if ratio > 0 else "REVERSED SIGN"))
        print()
        print("The wall it buys is unchanged by the corpus: {:+.2f}% of case "
              "wall.".format(WALL_CUT_K8))
        print("l212 scored that at NET +2.305% with the in-set quality cost.")
        print("Substituting the OOS cost: NET {:+.3f}%"
              .format(2.305 + (mean - (-0.1242))))
        print()
        print("SHIP TEST: both samples same sign, and the NET above still ahead")
        print("of today's +1.260%. Either sample negative enough to sink it is")
        print("a RED -- this mechanism has reversed out of sample twice before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
