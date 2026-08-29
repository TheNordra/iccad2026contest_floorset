"""L173c - fit wall(C) = a + b/C and read off what a 48-core grader actually pays.

`a` is the core-independent part: the M47 serial `_proxy_metrics` tail (~71 ms
per profile on the main thread, GIL-bound -- M47 records that running the
proxies concurrently was 4x WORSE), plus the wrapper and the harness. It
transfers to the grader in full, whatever the core count.

`b/C` is the profile pool. A 48-core grader buys back part of it, and only that
part.

This is the number that decides whether the 6.2x wall regression against M73
is a measurement artefact of a 32-core dev box or a real cost we will be
graded on.
"""
import glob
import json
import re
from pathlib import Path

DIR = Path(__file__).parent
F = 3.17          # dev second -> grader second, single-threaded (L161)


def load():
    out = {}
    for p in glob.glob(str(DIR / "_l173c_*_*.json")):
        m = re.search(r"_l173c_(\d+)_(\d+)\.json$", p)
        if not m:
            continue
        C, case = int(m.group(1)), int(m.group(2))
        try:
            r = json.load(open(p))["test_results"][0]
        except Exception:
            continue
        out.setdefault(case, {})[C] = (r["runtime_seconds"], r["block_count"])
    return out


def fit(points):
    """least squares on wall = a + b*x where x = 1/C."""
    xs = [1.0 / c for c in points]
    ys = [points[c] for c in points]
    n = len(xs)
    if n < 2:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def main():
    D = load()
    print(__doc__)
    print("=" * 74)
    if not D:
        print("no _l173c_*.json yet -- run l173_cores.sh first")
        return 1
    for case in sorted(D):
        pts = {c: v[0] for c, v in sorted(D[case].items())}
        n = list(D[case].values())[0][1]
        a, b = fit(pts)
        print("\ncase {}  n={}".format(case, n))
        print("   measured: " + "   ".join(
            "{}c {:.3f}s".format(c, t) for c, t in sorted(pts.items())))
        if a is None:
            continue
        print("   fit wall(C) = {:.3f} + {:.1f}/C".format(a, b))
        w48 = a + b / 48
        w32 = a + b / 32
        print("   serial part a = {:.3f}s  ({:.0f}% of the 32-core wall)"
              .format(a, 100 * a / w32))
        print("   predicted at 48 real cores: {:.3f}s dev  ->  {:.3f}s grader"
              " (f={})".format(w48, w48 / F, F))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
