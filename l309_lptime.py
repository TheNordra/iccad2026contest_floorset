"""L309 -- the shape LP's cost, measured directly instead of by differencing walls.

`ICCAD_LP_TIMING=1` makes `_shape_lp_maybe` print, per case,
    [lptime] n=<blocks> cpu=<process seconds> wall=<seconds> passes=...
so the LP's cost stops being an 8 %-noisy difference of two 130-second walls and
becomes a direct measurement.  `cpu` vs `wall` also settles whether the LP is
single-threaded -- which is the premise that makes L308's single-thread f the
right constant for it.

  <python> l309_lptime.py parse ship.log gate0k2.log
"""
import re, sys, statistics
from pathlib import Path

DIR = Path(__file__).parent
RX = re.compile(r"\[lptime\] n=(\d+) cpu=([0-9.]+) wall=([0-9.]+)")


def load(p):
    d = {}
    for m in RX.finditer(Path(p).read_text(errors="replace")):
        n, c, w = int(m.group(1)), float(m.group(2)), float(m.group(3))
        d.setdefault(n, []).append((c, w))
    return {n: (sum(c for c, _ in v), sum(w for _, w in v)) for n, v in d.items()}


if __name__ == "__main__":
    args = sys.argv[1:]
    logs = {Path(a).stem: load(a) for a in args}
    for lbl, d in logs.items():
        cpu = sum(c for c, _ in d.values()); wall = sum(w for _, w in d.values())
        print("  %-20s cases with an LP %3d   cpu %7.2f s   wall %7.2f s   cpu/wall %.3f"
              % (lbl, len(d), cpu, wall, cpu / wall if wall else 0))
    if len(logs) == 2:
        a, b = list(logs.values())
        ns = sorted(set(a) | set(b))
        dt = {n: b.get(n, (0, 0))[1] - a.get(n, (0, 0))[1] for n in ns}
        print("\n  dt (LP wall, arm - base), by block count:")
        v = sorted(dt.values())
        print("     sum %+.2f s   p50 %+.3f   p90 %+.3f   max %+.3f"
              % (sum(v), statistics.median(v), v[int(.9 * len(v))], max(v)))
        import pickle
        pickle.dump(dt, open(DIR / "l309_dt.pkl", "wb"))
        print("     -> l309_dt.pkl")
