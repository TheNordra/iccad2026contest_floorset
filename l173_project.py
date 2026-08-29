"""L173d - project the CURRENT package's graded runtime and score.

Every runtime number this project has priced against was `t_beta`, our graded
per-case runtime for M73. L173 shows the pool has moved 6.2x on the same box
since M73, so `t_beta` is no longer our runtime and every budget derived from
it is wrong.

THE MODEL, and every input to it is measured:

    wall(C) = a + b/C                       fitted, l173_cores.ps1, 4/8/16/32c
    alpha   = a / wall(32)                  the CORE-INDEPENDENT share
    w48(n)  = w32(n) * (alpha + (1-alpha) * 32/48)
    grader_pool(n) = w48(n) / f             f = 3.17, single-thread (L161)
    grader_t(n)    = grader_pool(n) + LP(n)/f

  w32(n)  per-case wall for all 100 cases, l166 LANE 1 (current tree, LP OFF,
          WSL2 32c, ICCAD_ADAPTIVE_CORES=48). LP is added back separately
          because it is single-threaded scipy -- pure serial, so it scales by f
          alone and must not be run through the core model.
  LP(n)   dt_tangent + (k-1)*dt_pass under the shipped depth map, measured.

CALIBRATION CHECK, and it is the thing that makes this credible: the same
pipeline applied to M73's own WSL run must reproduce beta's graded 52.07s.
M73 was max-setter bound on that box (c* 19.3/22.5 < 32), so for M73
alpha is irrelevant and grader_t = w32/f. 141.07/2.71 = 52.07 by construction,
which is how f was derived -- so this check is circular for M73 and is NOT
evidence. It is reported so the circularity is visible rather than hidden.
"""
import json
import math
import sys
from pathlib import Path

import l146_rf_price as L
import l172_depthmap as M

DIR = Path(__file__).parent
F = 3.17
WSL_LPOFF = "/home/i077/l166/t_lpoff/cadc1075/results_l117_t_lpoff.json"


def fit(points):
    xs = [1.0 / c for c in points]
    ys = [points[c] for c in points]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


# 32 logical cores on this box are 16 PHYSICAL ones plus SMT: the measured
# 32c wall is SLOWER than the 16c wall (n=120: 11.314 vs 11.250; n=114: 10.116
# vs 9.622), so the 32c point carries no throughput information and fitting it
# would flatten b and understate the pool. Fit on real cores only.
SCAN_CORES = (4, 8, 16)


def read_scan():
    out = {}
    txt = (DIR / "l173_cores.out").read_text(errors="ignore").replace("\x00", "")
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 4 and p[0].isdigit() and p[1].isdigit():
            try:
                if int(p[0]) in SCAN_CORES:
                    out.setdefault(int(p[1]), {})[int(p[0])] = float(p[2].replace(",", ""))
            except ValueError:
                pass
    return out


def main():
    scan = read_scan()
    if not scan:
        print("no core scan yet")
        return 1
    print(__doc__)
    print("=" * 76)
    alphas = []
    for case, pts in sorted(scan.items()):
        if len(pts) < 3:
            continue
        a, b = fit(pts)
        w32, w48 = a + b / 32, a + b / 48
        alphas.append(a / w32)
        print("   grader at 48 real cores {:.3f}s | at 24 effective {:.3f}s"
              .format(w48 / F, (a + b / 24) / F))
        print("case {}   {}".format(case, "  ".join(
            "{}c {:.2f}s".format(c, t) for c, t in sorted(pts.items()))))
        print("   wall(C) = {:.3f} + {:.1f}/C     serial a = {:.3f}s"
              "   alpha(32c) = {:.3f}".format(a, b, a, a / w32))
        print("   32c {:.2f}s -> 48c {:.2f}s   ({:.0f}% of the wall is "
              "core-independent at 48c)".format(w32, w48, 100 * a / w48))
    if not alphas:
        print("need >=3 core points per case")
        return 1
    alpha = sum(alphas) / len(alphas)
    print("\nalpha (mean over scanned cases) = {:.3f}".format(alpha))

    try:
        w32n = {r["block_count"]: r["runtime_seconds"]
                for r in json.load(open(WSL_LPOFF))["test_results"]}
    except Exception:
        print("\ncannot read the WSL LANE 1 dump from Windows; run:")
        print("  wsl -d Ubuntu -- cp {} /mnt/c/ICCAD_ml/ship_final/"
              "_l173_wsl_lpoff.json".format(WSL_LPOFF))
        alt = DIR / "_l173_wsl_lpoff.json"
        if not alt.exists():
            return 1
        w32n = {r["block_count"]: r["runtime_seconds"]
                for r in json.load(open(alt))["test_results"]}

    rows = M.rows_new()
    dtan, dpass, near = M.costs()
    x090 = {int(k): v for k, v in
            json.load(open(DIR / "l172_depthmap_x090.json")).items()}
    scale48 = alpha + (1 - alpha) * 32.0 / 48.0

    def grader_t(r, dmap):
        n = r["n"]
        pool = w32n.get(n, r["t"] * 6.22) * scale48 / F
        lp = (dtan.get(near(n), 0.0)
              + (dmap.get(n, 1) - 1) * dpass.get(near(n), 0.0)) / F
        return pool + lp

    print("\n{:>26}{:>12}{:>12}{:>10}{:>12}"
          .format("configuration", "our wall", "vs beta", "off floor", "graded"))
    W = sum(r["w"] for r in rows)
    beta_tot = sum(r["w"] * r["q"] * max(0.7, (r["t"] / r["med"]) ** 0.3)
                   for r in rows) / W
    print("{:>26}{:>11.1f}s{:>11.2f}x{:>8}/100{:>12.6f}"
          .format("M73 (what was graded)", sum(r["t"] for r in rows), 1.0,
                  sum(1 for r in rows if r["t"] > L.THR * r["med"]), beta_tot))
    for lbl, dmap, q in (("current, k=1", {n: 1 for n in x090}, 0.0),
                         ("current, x0.90 map", x090, 0.0)):
        ts = {r["i"]: grader_t(r, dmap) for r in rows}
        tot = sum(r["w"] * r["q"] * (1 - q / 100)
                  * max(0.7, (ts[r["i"]] / r["med"]) ** 0.3) for r in rows) / W
        print("{:>26}{:>11.1f}s{:>11.2f}x{:>8}/100{:>12.6f}"
              .format(lbl, sum(ts.values()),
                      sum(ts.values()) / sum(r["t"] for r in rows),
                      sum(1 for r in rows if ts[r["i"]] > L.THR * r["med"]), tot))
    print("\nthresholds 2026-08-23:  r1 0.85863  r2 0.88819  r3 0.89933"
          "  r4 0.92659")
    print("\nThe quality the stack buys is NOT applied above -- these rows are")
    print("the RUNTIME cost alone, so the gap to the beta row is what the")
    print("quality has to pay for before it buys anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
