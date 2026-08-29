"""L281 first-order prize, over the WHOLE graded corpus, with no LP at all.

For unit u, moving it rigidly to its weighted-L1-median with everything else
held fixed is the EXACT minimum of u's own wire term, so

    prize(u) = hw_scale * (wire(u, 0) - wire(u, d*))     hw_scale = 0.5/h_base

is the exact first-order reduction of the cost bracket 0.5*(hgap+agap) that
relocating u can contribute through its own edges.

What this bounds and what it does not:
  * it IS the exact best case for u's own edges, ignoring every constraint --
    no bbox, no overlap, no coherence.  Every real relocation gets less.
  * it is NOT a bound on the total, because relocating u also frees space and
    lets the LP compact others.  Measured deltas can exceed it; the LP arm is
    what settles that.  It is reported as a ceiling on the term relocation is
    supposed to buy, on all 100 graded cases, for the price of no LP.
"""
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import l281_reloc_probe as L                                       # noqa: E402

W, CASES = L.W, L.CASES

anchor = sys.argv[1] if len(sys.argv) > 1 else str(
    _DIR / "results_L274_base_48c.json")
aj = json.loads(open(anchor, "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
print(f"[anchor] {Path(anchor).name} total={aj['total_score']:.10f}")

rows = []
for ci in sorted(ANCH):
    e = ANCH[ci]
    P = [tuple(p) for p in e["positions"]]
    ranked, G = L.rank_units(ci, P)
    free = [r for r in ranked if not r["pinned"]]
    if not free:
        continue
    import math
    ev = math.exp(2.0 * e["violations_relative"])
    best1 = free[0]["prize"] * ev
    top5 = sum(r["prize"] for r in free[:5]) * ev
    rows.append((ci, CASES[ci]["n"], e["cost"], best1, top5,
                 len(free), len(ranked) - len(free)))
    print(f"case {ci:3d} n={CASES[ci]['n']:3d} cost {e['cost']:.6f}  "
          f"best-1 prize {best1:.6f} ({100 * best1 / e['cost']:6.4f} %)  "
          f"top-5 {top5:.6f} ({100 * top5 / e['cost']:6.4f} %)  "
          f"free units {len(free)}/{len(ranked)}", flush=True)

wsum = sum(W[r[0]] for r in rows)
tot = sum(W[r[0]] * r[2] for r in rows) / wsum
g1 = sum(W[r[0]] * r[3] for r in rows) / wsum
g5 = sum(W[r[0]] * r[4] for r in rows) / wsum
hv = [r for r in rows if r[1] >= 101]
wh = sum(W[r[0]] for r in hv)
print(f"\n== {len(rows)} cases, weighted exp(n/12), base {tot:.6f} ==")
print(f"  best single unit per case, relocated to its wire optimum, all else "
      f"fixed : {100 * g1 / tot:+.4f} %")
print(f"  best five units per case (not jointly realisable)              "
      f"      : {100 * g5 / tot:+.4f} %")
if hv:
    th = sum(W[r[0]] * r[2] for r in hv) / wh
    print(f"  heavy band n>=101 ({len(hv)} cases): best-1 "
          f"{100 * (sum(W[r[0]] * r[3] for r in hv) / wh) / th:+.4f} %  "
          f"top-5 {100 * (sum(W[r[0]] * r[4] for r in hv) / wh) / th:+.4f} %")
pr = sorted(100 * r[3] / r[2] for r in rows)
print(f"  per-case best-1 as %% of that case: p25 {pr[len(pr) // 4]:.4f}  "
      f"p50 {pr[len(pr) // 2]:.4f}  p75 {pr[3 * len(pr) // 4]:.4f}  "
      f"max {pr[-1]:.4f}")
npin = sum(r[6] for r in rows)
print(f"  units excluded as boundary/extreme-pinned: {npin} of "
      f"{sum(r[5] + r[6] for r in rows)} "
      f"({100.0 * npin / max(sum(r[5] + r[6] for r in rows), 1):.1f} %)")
