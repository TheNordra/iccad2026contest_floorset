"""L144 verify1 -- pooled analysis over SIX profiles (their 0/2/5 + my 1/3/7).

Profile 7 is the only positive arm found (+0.349%). Three questions decide
whether it is real or a multiple-comparisons artefact:
  (a) how does +0.349% sit in the distribution of the six arms?
  (b) does one case carry it (leave-one-out)?
  (c) does the flag-ON arm ever beat the SHIPPED full pool, which is the only
      baseline that matters for a NET decision?
Read-only.
"""
import json
import math
import os
import re
import sys
from pathlib import Path

_DIR = Path(__file__).parent
_SP = Path(r"C:\Users\0150B8~1\AppData\Local\Temp\claude\C--ICCAD-ml"
           r"\574c551c-a4ae-4d5d-8ef2-d147eceabc4b\scratchpad")
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))
for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]
import m77_oos_probe as m77                                          # noqa: E402

SPECS = m77._specs("s1")[:32]
NS = {ck: n for ck, fk, lid, n in SPECS}
ROW = re.compile(r"^(worker_\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)$")

SRC = {"p0": _DIR / "_l144v2_p0_32.log", "p2": _DIR / "_l144v2_p2_32.log",
       "p5": _DIR / "_l144v2_p5_32.log", "p1": _SP / "v1_p1_32.log",
       "p3": _SP / "v1_p3_32.log", "p7": _SP / "v1_p7_32.log"}
DATA = {}
for tag, path in SRC.items():
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = ROW.match(line.strip())
        if m:
            rows[m.group(1)] = (int(m.group(2)), int(m.group(3)),
                                float(m.group(4)), float(m.group(5)))
    assert len(rows) == 32, (tag, len(rows))
    DATA[tag] = rows

SHIP = {r["key"]: r for r in
        json.load(open(_DIR / "l140_oos_s1_c48.json"))["test_results"]}
CASES = sorted(DATA["p0"])
W = {c: math.exp(NS[c] / 12.0) for c in CASES}


def wm(v, cs=None):
    cs = cs or CASES
    return sum(W[c] * v[c] for c in cs) / sum(W[c] for c in cs)


print("=== (a) all six arms, 32 cases each ===")
res = []
for t in ("p0", "p1", "p2", "p3", "p5", "p7"):
    r = DATA[t]
    c0, c1 = wm({c: r[c][2] for c in CASES}), wm({c: r[c][3] for c in CASES})
    d = 100 * (c0 - c1) / c0
    res.append(d)
    print(f"  {t}  cost {c0:.6f} -> {c1:.6f}  {d:+.3f}%   "
          f"bnd {sum(r[c][0] for c in CASES)} -> {sum(r[c][1] for c in CASES)}")
mu = sum(res) / len(res)
sd = (sum((x - mu) ** 2 for x in res) / (len(res) - 1)) ** 0.5
print(f"  mean {mu:+.3f}%   sd {sd:.3f}%   positive arms: "
      f"{sum(1 for x in res if x > 0)}/6   >= +0.30% bar: "
      f"{sum(1 for x in res if x >= 0.30)}/6")

print("\n=== (b) leave-one-out on profile 7, the only positive arm ===")
r = DATA["p7"]
loo = []
for drop in CASES:
    keep = [c for c in CASES if c != drop]
    a, b = wm({c: r[c][2] for c in keep}, keep), wm({c: r[c][3] for c in keep}, keep)
    loo.append((100 * (a - b) / a, drop))
loo.sort()
print(f"  full = +0.349%   LOO range [{loo[0][0]:+.3f}% (drop {loo[0][1]})"
      f" .. {loo[-1][0]:+.3f}% (drop {loo[-1][1]})]")
neg = [d for d, _ in loo if d < 0]
bar = [d for d, _ in loo if d < 0.30]
print(f"  sign flips NEGATIVE on {len(neg)}/32 single-case drops")
print(f"  falls below the +0.30% bar on {len(bar)}/32 single-case drops")
top = sorted(CASES, key=lambda c: -W[c] * (r[c][2] - r[c][3]))[:3]
for c in top:
    print(f"    top contributor {c:<30} {r[c][2]:.5f} -> {r[c][3]:.5f}"
          f"  w={W[c]/sum(W.values()):.2%}")

print("\n=== (c) vs the SHIPPED full pool (35 arms) -- the only real baseline ===")
ship = {c: SHIP[c]["cost"] for c in CASES}
pmin0 = {c: min(DATA[t][c][2] for t in DATA) for c in CASES}
pmin1 = {c: min(DATA[t][c][3] for t in DATA) for c in CASES}
print(f"  6-arm portfolio flag OFF   {wm(pmin0):.6f}")
print(f"  6-arm portfolio flag ON    {wm(pmin1):.6f}")
print(f"  SHIPPED full pool (l140)   {wm(ship):.6f}")
for t in ("p0", "p1", "p2", "p3", "p5", "p7"):
    w = sum(1 for c in CASES if DATA[t][c][3] < ship[c] - 1e-9)
    print(f"  flag-ON arm {t} beats shipped on {w}/32 cases")
anyw = [c for c in CASES if pmin1[c] < ship[c] - 1e-9]
print(f"  ANY flag-ON arm beats shipped on {len(anyw)}/32 cases -> "
      f"marginal value as an added pool arm = "
      f"{100*(wm(ship)-wm({c: min(ship[c], pmin1[c]) for c in CASES}))/wm(ship):+.4f}%")
print(f"  shipped v_bnd on these 32 cases = {sum(SHIP[c]['v_bnd'] for c in CASES)}"
      f"   (best single screened arm flag-ON = "
      f"{min(sum(DATA[t][c][1] for c in CASES) for t in DATA)})")
