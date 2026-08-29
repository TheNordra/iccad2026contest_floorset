"""L172c - emit the x0.90 depth map and everything needed to justify it."""
import json
import math
from collections import Counter
from pathlib import Path

import l172_depthmap as M
import l172_grid as G
import l146_rf_price as L

DIR = Path(__file__).parent
SCALE = 0.90

rows = M.rows_new()
dtan, dpass, near = M.costs()
new = M.build(rows, dtan, dpass, near, scale=SCALE)
k1 = {n: 1 for n in M.SHIPPED}

print("depth histogram   shipped {}   x0.90 {}"
      .format(dict(sorted(Counter(M.SHIPPED.values()).items())),
              dict(sorted(Counter(new.values()).items()))))
deeper = [n for n in sorted(new) if new[n] > M.SHIPPED.get(n, 1)]
shallower = [n for n in sorted(new) if new[n] < M.SHIPPED.get(n, 1)]
print("deeper than shipped:    {}  {}".format(len(deeper), deeper))
print("shallower than shipped: {}".format(len(shallower)))
print("n covered: {}  min {}  max {}".format(len(new), min(new), max(new)))
print("every n in 21..120 present:",
      sorted(new) == list(range(21, 121)))

for lbl, m in (("k=1", k1), ("SHIPPED", M.SHIPPED), ("x0.90", new)):
    q1 = M.quality(m, "s1")
    q2 = M.quality(m, "s2")
    print("{:>9}  OOS s1 {:+.4f}% ({} moved / {} worse)   "
          "s2 {:+.4f}% ({} moved / {} worse)"
          .format(lbl, q1[0], q1[1], q1[2], q2[0], q2[1], q2[2]))

print("\nRF on the new table, vs the k=1 anchor:")
for lbl, m in (("SHIPPED", M.SHIPPED), ("x0.90", new)):
    for s in (1.15, 1.00, 0.90, 0.80):
        b = G.rf_on(rows, k1, dtan, dpass, near, s)
        r = G.rf_on(rows, m, dtan, dpass, near, s)
        print("  {:>8}  s_true x{:.2f}   RF {:+.4f}%".format(
            lbl, s, 100 * (b - r) / b))

json.dump({str(k): v for k, v in sorted(new.items())},
          open(DIR / "l172_depthmap_x090.json", "w"), indent=0)
print("\nwrote l172_depthmap_x090.json")

# the literal, formatted the way _L157_DEPTH already is
ks = sorted(new)
line, out = "    ", []
for n in ks:
    piece = "{}: {}, ".format(n, new[n])
    if len(line) + len(piece) > 76:
        out.append(line.rstrip())
        line = "    "
    line += piece
out.append(line.rstrip().rstrip(","))
print("\n_L157_DEPTH = {")
print("\n".join(out))
print("}")
