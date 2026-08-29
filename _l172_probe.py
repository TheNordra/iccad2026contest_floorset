"""Find the exact formula that produced the shipped _L157_DEPTH map."""
import l172_depthmap as M
import l146_rf_price as L

rows = L.load()
dtan, dpass, near = M.costs()


def build(thr, f, use_near, tan_div):
    out = {}
    for r in rows:
        n = r["n"]
        g = near if use_near else (lambda x: x)
        dt = dtan.get(g(n), 0.0)
        dp = dpass.get(g(n), 0.0)
        budget = thr * r["med"] - r["t"] - (dt / f if tan_div else dt)
        k = 1
        for kk in (2, 3):
            if (kk - 1) * dp / f <= budget:
                k = kk
        out[n] = k
    return out


best = None
for thr in (L.THR, 0.3046):
    for f in (3.17, 2.71, 1.0):
        for un in (True, False):
            for td in (True, False):
                m = build(thr, f, un, td)
                d = sum(1 for n in M.SHIPPED if m.get(n) != M.SHIPPED[n])
                tag = (round(thr, 6), f, "near" if un else "exactn",
                       "tan/f" if td else "tanraw")
                if best is None or d < best[0]:
                    best = (d, tag, m)
                if d == 0:
                    print("EXACT MATCH:", tag)
print("best mismatch count", best[0], best[1])
m = best[2]
print("mismatches:",
      [(n, M.SHIPPED[n], m[n]) for n in sorted(M.SHIPPED) if m.get(n) != M.SHIPPED[n]])
print()
print("dtan/dpass coverage: n in dpass?",
      sorted(set(M.SHIPPED) - set(dpass))[:20])
print("n values with a beta case:", len({r["n"] for r in rows}))
