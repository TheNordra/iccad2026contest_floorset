"""L180c - diff the two cProfile tables. The gap is ~1.0s/case of Python."""
import os
import pstats
from pathlib import Path

DIR = Path(__file__).resolve().parent


def table(tag):
    st = pstats.Stats(str(DIR / "_l180_{}.prof".format(tag)))
    out = {}
    for k, v in st.stats.items():
        # match on FUNCTION NAME only: the two wrappers are different
        # files with different line numbers, so a file:line key never
        # matches and every function looks "new".
        name = "{} [{}]".format(k[2], os.path.basename(k[0]))
        if os.path.basename(k[0]) == "optimizer_constructive.py":
            name = k[2] + " [wrapper]"
        pv = out.get(name, (0.0, 0.0, 0))
        out[name] = (pv[0] + v[2], pv[1] + v[3], pv[2] + v[0])       # tottime, cumtime, ncalls
    return st.total_tt, out


tc, C = table("cur")
tm, M = table("m73")
print("total profiled time   current {:.3f}s   M73 {:.3f}s   delta {:+.3f}s"
      .format(tc, tm, tc - tm))

print("\n=== biggest tottime INCREASES, current vs M73 ===")
print("{:>10}{:>10}{:>10}{:>9}  {}".format("d tottime", "cur", "m73",
                                           "ncalls", "function"))
rows = []
for name, (tt, cum, nc) in C.items():
    tt0 = M.get(name, (0.0, 0.0, 0))[0]
    rows.append((tt - tt0, tt, tt0, nc, name))
rows.sort(reverse=True)
for d, tt, tt0, nc, name in rows[:16]:
    if d < 0.005:
        break
    print("{:>+9.3f}s{:>9.3f}s{:>9.3f}s{:>9}  {}".format(d, tt, tt0, nc, name))

print("\n=== present in current, ABSENT in M73 (new per-case work) ===")
new = [(v[0], v[2], k) for k, v in C.items() if k not in M and v[0] > 0.004]
new.sort(reverse=True)
for tt, nc, name in new[:14]:
    print("{:>9.3f}s{:>9}  {}".format(tt, nc, name))
