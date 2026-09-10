"""M78 per-case mover diff between two m78_probe.py `score` dumps. OFFLINE."""
import json
import math
import sys
from pathlib import Path

_DIR = Path(__file__).parent


def load(p):
    d = json.load(open(_DIR / p))
    return {r["test_id"]: r for r in d["test_results"]}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    W = sum(r["weight"] for r in a.values())
    ta = sum(r["weight"] * r["cost"] for r in a.values()) / W
    tb = sum(r["weight"] * r["cost"] for r in b.values()) / W
    print(f"A {sys.argv[1]}  total {ta:.15f}")
    print(f"B {sys.argv[2]}  total {tb:.15f}")
    print(f"delta {100 * (tb - ta) / ta:+.4f}%   (negative = B better)\n")
    rows = []
    for i, ra in a.items():
        rb = b[i]
        if ra["cost"] != rb["cost"]:
            # weighted contribution to the total, i.e. what this case actually buys
            rows.append((i, ra["block_count"], ra["cost"], rb["cost"],
                         100 * (rb["cost"] - ra["cost"]) / ra["cost"],
                         100 * ra["weight"] * (rb["cost"] - ra["cost"]) / (W * ta)))
    rows.sort(key=lambda r: r[5])
    better = sum(1 for r in rows if r[4] < 0)
    print(f"movers {len(rows)}   better {better}   worse {len(rows) - better}")
    print(f"  {'case':>4} {'n':>4} {'A cost':>9} {'B cost':>9} {'d%':>8} {'wtd%':>8}")
    for i, n, ca, cb, d, wd in rows:
        print(f"  {i:>4} {n:>4} {ca:9.5f} {cb:9.5f} {d:+7.3f}% {wd:+7.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
