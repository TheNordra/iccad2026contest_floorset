"""L217 score - the targeted drop shape, in set and out of sample.

The shipped table drops 8 profiles at EVERY block count. That is the wrong
shape, and the reason is exact rather than empirical: RF is
`max(0.7, (t/med)^0.3)`, so its derivative on the floor is **zero**. 54 of 100
block counts sit on the floor in the shipped configuration, and dropping a
profile there buys literally nothing while still costing quality. Uniform k
spends most of its quality budget where it cannot be repaid.

The shape measured here maximises, per block count,

    exact RF gain(n, k)  -  lam * k * w_n / W

with `lam` calibrated globally so a uniform k=8 reproduces the measured k=8
quality cost, then doubled as the conservative end. Only the RF side is per-n;
the quality side is a smooth global price, because choosing k from a block
count's own measured cost is exactly the overfitting the 2.41x OOS
amplification exists to price.

Reported against two baselines that fail differently:

  vs L209  (no drop at all)      -- what the whole mechanism is worth
  vs L214  (uniform k=8, shipped) -- whether re-shaping was worth the change

  <python> l217_score.py
"""
import json
import math
from pathlib import Path

DIR = Path(__file__).parent


def load(fn, key="test_id"):
    f = DIR / fn
    if not f.exists():
        return None
    return {r[key]: r for r in json.load(open(f))["test_results"]}


def wq(d, ids, nkey):
    w = lambda i: math.exp(d[i][nkey] / 12.0)                  # noqa: E731
    return sum(w(i) * d[i]["cost"] for i in ids) / sum(w(i) for i in ids)


def per_n(a, b, ids, rows, nkey):
    sa, sb, cnt = {}, {}, {}
    for i in ids:
        n = a[i][nkey]
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
    tab = json.loads((DIR / "l217_drop_targeted.json").read_text())
    nz = sum(1 for v in tab.values() if v)
    print("=" * 78)
    print("L217 targeted shape: {} of {} block counts get a drop, {} total "
          "(shipped: 100 / 800)".format(nz, len(tab),
                                        sum(len(v) for v in tab.values())))
    print("=" * 78)

    # ---- in set --------------------------------------------------------------
    b0, b8 = load("results_L209_det1.json"), load("results_L214_det1.json")
    tg = load("results_L217_inset.json")
    if b0 and tg:
        for lbl, ref in (("vs no drop (L209)", b0), ("vs uniform k=8 (L214)", b8)):
            if not ref:
                continue
            ids = sorted(set(ref) & set(tg))
            q = 100 * (wq(ref, ids, "block_count") - wq(tg, ids, "block_count")) \
                / wq(ref, ids, "block_count")
            mv = sum(1 for i in ids if ref[i]["cost"] != tg[i]["cost"])
            ws = sum(1 for i in ids if tg[i]["cost"] > ref[i]["cost"] + 1e-12)
            nf = sum(1 for i in ids if not tg[i].get("is_feasible", True))
            print("IN SET  {:<24} quality {:+.4f}%   {} moved / {} worse{}"
                  .format(lbl, q, mv, ws,
                          "   !! {} INFEASIBLE".format(nf) if nf else ""))
    else:
        print("IN SET  not run")

    # ---- out of sample -------------------------------------------------------
    print("-" * 78)
    qs = []
    for s in ("s1", "s2"):
        a = load("l213_{}_base.json".format(s), "test_id")
        t = load("l217_{}_tgt.json".format(s), "test_id")
        k8 = load("l213_{}_k8.json".format(s), "test_id")
        if not (a and t):
            print("OOS {:<5} not run".format(s))
            continue
        ids = sorted(set(a) & set(t))
        qn = per_n(a, t, ids, rows, "n")
        mv = sum(1 for i in ids if a[i]["cost"] != t[i]["cost"])
        nf = sum(1 for i in ids if not t[i].get("feasible", True))
        extra = ""
        if k8:
            ids2 = sorted(set(k8) & set(t))
            extra = "   vs uniform k=8: {:+.4f}%".format(
                per_n(k8, t, ids2, rows, "n"))
        qs.append(qn)
        print("OOS {:<5} quality/n {:+.4f}% vs no drop   {} moved{}{}"
              .format(s, qn, mv, extra,
                      "   !! {} INFEASIBLE".format(nf) if nf else ""))
    print("-" * 78)
    if len(qs) == 2:
        mean = sum(qs) / 2
        print("mean OOS quality cost: {:+.4f}%".format(mean))
        print("uniform k=8 was      : -0.2989%   (for a smaller RF gain)")
        print()
        print("MODELLED NET: targeted +2.190% vs shipped +1.692% vs no drop "
              "+1.260%.")
        print("The RF half of that is a model; this file measures the quality")
        print("half. Substituting the measured cost for the modelled -0.2993%:")
        print("   NET {:+.3f}%".format(2.190 + (mean - (-0.2993))))
        print()
        print("SHIP TEST: both samples same sign, 0 infeasible, and the NET")
        print("above still ahead of the shipped +1.692%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
