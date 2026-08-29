"""L144 verify1 -- is the 3-profile 'AUGMENT +0.884%' anything but best-of-N inflation?

The shipped optimizer runs the FULL adaptive pool per case and keeps the best.
The colleague's screen ran ONE profile solo. My portfolio re-analysis of their
logs showed min-over-3 improves when the flag-on runs are added as extra arms
(+0.884%) -- but min-over-N always falls as N grows, whatever the extra arms are.

The control: l140_oos_s1_c48.json holds the SHIPPED optimizer's per-case cost on
these exact cases at the grader's 48-core pool shape. If the shipped full-pool
cost already sits below the '6-arm augmented' value, the +0.884% is measured
against a crippled 3-arm portfolio and is worth nothing.

Read-only.
"""
import json
import math
import os
import re
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]
import m77_oos_probe as m77                                          # noqa: E402
import optimizer_constructive as oc                                  # noqa: E402

SPECS = m77._specs("s1")[:32]
NS = {ck: n for ck, fk, lid, n in SPECS}
ROW = re.compile(r"^(worker_\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)$")

DATA = {}
for tag in ("p0", "p2", "p5"):
    rows = {}
    for line in (_DIR / f"_l144v2_{tag}_32.log").read_text().splitlines():
        m = ROW.match(line.strip())
        if m:
            rows[m.group(1)] = (int(m.group(2)), int(m.group(3)),
                                float(m.group(4)), float(m.group(5)))
    DATA[tag] = rows

SHIP = {r["key"]: r for r in
        json.load(open(_DIR / "l140_oos_s1_c48.json"))["test_results"]}

CASES = sorted(DATA["p0"])
W = {c: math.exp(NS[c] / 12.0) for c in CASES}


def wmean(v):
    return sum(W[c] * v[c] for c in CASES) / sum(W[c] for c in CASES)


pmin0 = {c: min(DATA[t][c][2] for t in DATA) for c in CASES}
pmin1 = {c: min(DATA[t][c][3] for t in DATA) for c in CASES}
pboth = {c: min(pmin0[c], pmin1[c]) for c in CASES}
ship = {c: SHIP[c]["cost"] for c in CASES}

print("=== pool size actually used by the shipped optimizer on these cases ===")
sizes = sorted({len(oc._pool_indices(NS[c])) for c in CASES})
print(f"  block counts n in [{min(NS[c] for c in CASES)}..{max(NS[c] for c in CASES)}]"
      f"   pool sizes {sizes}  (the screen used 3 arms)")

print("\n=== weighted cost on the same 32 cases ===")
print(f"  3-arm portfolio, flag OFF      {wmean(pmin0):.6f}")
print(f"  3-arm portfolio, flag ON       {wmean(pmin1):.6f}")
print(f"  6-arm 'augment' (OFF+ON)       {wmean(pboth):.6f}")
print(f"  SHIPPED full pool (l140, 48c)  {wmean(ship):.6f}   <-- the real baseline")

b = sum(1 for c in CASES if pboth[c] < ship[c] - 1e-9)
print(f"\n  cases where the 6-arm augmented screen beats the shipped full pool: "
      f"{b}/32")
gap = 100 * (wmean(pboth) - wmean(ship)) / wmean(ship)
print(f"  the 6-arm augmented screen is {gap:+.2f}% vs shipped "
      f"(positive = WORSE than what already ships)")

print("\n=== per-case: does the flag-on arm ever beat the shipped pool? ===")
wins = [(c, pmin1[c], ship[c]) for c in CASES if pmin1[c] < ship[c] - 1e-9]
print(f"  flag-ON arms beat shipped on {len(wins)}/32 cases")
for c, a, s in sorted(wins, key=lambda t: t[1] - t[2])[:10]:
    print(f"    {c:<30} on={a:.5f}  shipped={s:.5f}  ({a-s:+.5f})")

print("\n=== boundary violations: shipped vs screen ===")
sb = sum(SHIP[c]["v_bnd"] for c in CASES)
print(f"  shipped full pool v_bnd total on these 32 cases: {sb}")
print(f"  screen p0/p2/p5 flag OFF totals: "
      f"{sum(DATA['p0'][c][0] for c in CASES)}/"
      f"{sum(DATA['p2'][c][0] for c in CASES)}/"
      f"{sum(DATA['p5'][c][0] for c in CASES)}")
print(f"  screen p0/p2/p5 flag ON  totals: "
      f"{sum(DATA['p0'][c][1] for c in CASES)}/"
      f"{sum(DATA['p2'][c][1] for c in CASES)}/"
      f"{sum(DATA['p5'][c][1] for c in CASES)}")
