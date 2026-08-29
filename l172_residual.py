"""L172f - how many grader-seconds are left AFTER the tangent and the x0.90 map?

This is the number the next mechanism has to fit inside. It replaces the
"19.79s of budget" the handoff quotes, which came from the 2026-08-19 table.
"""
import l146_rf_price as L
import l172_depthmap as M

THR = L.THR


def spend(rows, dmap, dtan, dpass, near, f=M.F):
    return {r["i"]: (dtan.get(near(r["n"]), 0.0)
                     + (dmap.get(r["n"], 1) - 1) * dpass.get(near(r["n"]), 0.0)) / f
            for r in rows}


def main():
    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    import json
    from pathlib import Path
    x090 = {int(k): v for k, v in
            json.load(open(Path(__file__).parent / "l172_depthmap_x090.json")).items()}
    print(__doc__)
    print("=" * 74)
    for lbl, dmap in (("beta package (no LP at all)", None),
                      ("+ L147 tangent, k=1", {n: 1 for n in x090}),
                      ("+ the SHIPPED old map", M.SHIPPED),
                      ("+ the x0.90 map (now shipped)", x090)):
        if dmap is None:
            sp = {r["i"]: 0.0 for r in rows}
        else:
            sp = spend(rows, dmap, dtan, dpass, near)
        free = sum(max(0.0, THR * r["med"] - r["t"] - sp[r["i"]]) for r in rows)
        over = sum(max(0.0, r["t"] + sp[r["i"]] - THR * r["med"]) for r in rows)
        noff = sum(1 for r in rows if r["t"] + sp[r["i"]] > THR * r["med"])
        num = sum(r["w"] * r["q"]
                  * max(0.7, ((r["t"] + sp[r["i"]]) / r["med"]) ** 0.3)
                  for r in rows)
        W = sum(r["w"] for r in rows)
        print("{:<32} spends {:>6.2f}s   free left {:>6.2f}s   "
              "overspent {:>5.2f}s   off floor {:>2}/100   graded {:.6f}"
              .format(lbl, sum(sp.values()), free, over, noff, num / W))

    print("\nSo the next mechanism has ~{:.1f} grader-seconds of genuinely free"
          " room,".format(
              sum(max(0.0, THR * r["med"] - r["t"]
                      - spend(rows, x090, dtan, dpass, near)[r["i"]])
                  for r in rows)))
    print("but it is spread across the LIGHT cases: exp(n/12) puts 71% of the")
    print("weight on n>105, and those cases have slack p50 1.41x, not 2.02x.")
    print("A mechanism that costs a flat 0.10s/case costs -0.32% and is NOT")
    print("covered by that free pool, because the pool is in the wrong cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
