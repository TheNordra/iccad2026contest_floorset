"""Re-weight the per-case columns already printed by l144v2_ab.py.

No solver runs: it only re-reads the logs and the case weights wt=exp(n/12), so
it can answer "is profile 5 just the one 10.0-capped case?" for free.
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
import m77_oos_probe as m77                                         # noqa: E402

NS = {ck: n for ck, fk, lid, n in m77._specs("s1")[:32]}
ROW = re.compile(r"^(worker_\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)$")

for tag in ("p0", "p2", "p5"):
    rows = []
    for line in (_DIR / f"_l144v2_{tag}_32.log").read_text().splitlines():
        m = ROW.match(line.strip())
        if m:
            rows.append((m.group(1), int(m.group(2)), int(m.group(3)),
                         float(m.group(4)), float(m.group(5))))
    assert len(rows) == 32, (tag, len(rows))
    drop = [c for c, b0, b1, c0, c1 in rows if max(c0, c1) >= 9.999]
    for label, keep in (("all 32", rows),
                        (f"minus capped {drop}",
                         [r for r in rows if r[0] not in drop])):
        if not drop and label != "all 32":
            continue
        W = sum(math.exp(NS[c] / 12.0) for c, *_ in keep)
        a = sum(math.exp(NS[c] / 12.0) * c0 for c, b0, b1, c0, c1 in keep) / W
        b = sum(math.exp(NS[c] / 12.0) * c1 for c, b0, b1, c0, c1 in keep) / W
        nb0 = sum(r[1] for r in keep)
        nb1 = sum(r[2] for r in keep)
        print(f"{tag} {label:<45} n={len(keep):>2}  "
              f"cost {a:.6f} -> {b:.6f} ({100*(a-b)/a:+.3f}%)   "
              f"bnd {nb0} -> {nb1}")
