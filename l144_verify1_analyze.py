"""L144 verify1 -- independent re-analysis of the L144v2 screen logs.

Read-only: parses the three 32-case A/B logs the colleague produced and
recomputes everything from scratch with my own weighting, plus two checks
their report never ran:

  (A) portfolio-min behaviour (the shipped optimizer takes the best profile
      per case, it does not run one profile solo),
  (B) a per-case sign test / leave-one-out on the boundary counts, to see
      whether any single case carries the reported deltas.
"""
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
    assert len(rows) == 32, (tag, len(rows))
    DATA[tag] = rows

CASES = sorted(DATA["p0"])
W = {c: math.exp(NS[c] / 12.0) for c in CASES}


def wmean(vals, cases):
    s = sum(W[c] for c in cases)
    return sum(W[c] * vals[c] for c in cases) / s


print("=== 1. per-profile reproduction (my own weighting) ===")
for tag in ("p0", "p2", "p5"):
    r = DATA[tag]
    c0 = wmean({c: r[c][2] for c in CASES}, CASES)
    c1 = wmean({c: r[c][3] for c in CASES}, CASES)
    b0 = sum(r[c][0] for c in CASES)
    b1 = sum(r[c][1] for c in CASES)
    mv = sum(1 for c in CASES if r[c][2] != r[c][3])
    print(f"  {tag}  cost {c0:.6f} -> {c1:.6f} ({100*(c0-c1)/c0:+.3f}%)   "
          f"bnd {b0} -> {b1}   cost-moved {mv}/32")

print("\n=== 2. weight concentration (is this really 32 cases?) ===")
tot = sum(W.values())
top = sorted(CASES, key=lambda c: -W[c])
run = 0.0
for i, c in enumerate(top[:6]):
    run += W[c]
    print(f"  {i+1:>2}. {c:<30} n={NS[c]:>3}  w={W[c]/tot:6.2%}  cum={run/tot:6.2%}")
print(f"  effective sample size (Kish) = "
      f"{tot**2 / sum(w*w for w in W.values()):.2f} of 32")

print("\n=== 3. leave-one-out on the weighted cost delta ===")
for tag in ("p0", "p2", "p5"):
    r = DATA[tag]
    base = 100 * (wmean({c: r[c][2] for c in CASES}, CASES)
                  - wmean({c: r[c][3] for c in CASES}, CASES)) \
        / wmean({c: r[c][2] for c in CASES}, CASES)
    deltas = []
    for drop in CASES:
        keep = [c for c in CASES if c != drop]
        a = wmean({c: r[c][2] for c in keep}, keep)
        b = wmean({c: r[c][3] for c in keep}, keep)
        deltas.append((100 * (a - b) / a, drop))
    deltas.sort()
    print(f"  {tag}  full={base:+.3f}%   LOO range [{deltas[0][0]:+.3f}% "
          f"(drop {deltas[0][1]}) .. {deltas[-1][0]:+.3f}% (drop {deltas[-1][1]})]")
    print(f"        sign flips to positive on "
          f"{sum(1 for d, _ in deltas if d > 0)}/32 leave-one-outs")

print("\n=== 4. PORTFOLIO behaviour over the 3 screened profiles ===")
print("    (the shipped optimizer takes the best profile per case;")
print("     a solo-profile A/B cannot see this.)")
pmin0 = {c: min(DATA[t][c][2] for t in DATA) for c in CASES}
pmin1 = {c: min(DATA[t][c][3] for t in DATA) for c in CASES}
pboth = {c: min(pmin0[c], pmin1[c]) for c in CASES}
a = wmean(pmin0, CASES)
b = wmean(pmin1, CASES)
d = wmean(pboth, CASES)
print(f"  REPLACE  min over 3 profiles @0 = {a:.6f}")
print(f"           min over 3 profiles @1 = {b:.6f}   ({100*(a-b)/a:+.3f}%)")
print(f"  AUGMENT  min over all 6 runs    = {d:.6f}   ({100*(a-d)/a:+.3f}%)"
      f"   [costs 2x runtime]")
bw = sum(1 for c in CASES if pmin1[c] < pmin0[c] - 1e-9)
ww = sum(1 for c in CASES if pmin1[c] > pmin0[c] + 1e-9)
print(f"  replace: better on {bw} cases, worse on {ww} cases")
aw = [c for c in CASES if pboth[c] < pmin0[c] - 1e-9]
print(f"  augment: the flag wins the portfolio on {len(aw)} cases {aw}")

print("\n=== 5. boundary count: does one case carry it? ===")
for tag in ("p0", "p2", "p5"):
    r = DATA[tag]
    moved = [(c, r[c][0], r[c][1]) for c in CASES if r[c][0] != r[c][1]]
    net = sum(b1 - b0 for _, b0, b1 in moved)
    print(f"  {tag}  net {net:+d} over {len(moved)} cases that moved: "
          + ", ".join(f"{c.split('/')[-1]} {b0}->{b1}" for c, b0, b1 in moved))
allb0 = sum(DATA[t][c][0] for t in DATA for c in CASES)
allb1 = sum(DATA[t][c][1] for t in DATA for c in CASES)
print(f"  pooled over 3x32=96 solves: {allb0} -> {allb1} ({allb1-allb0:+d})")
