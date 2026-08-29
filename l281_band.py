"""L281 heavy-band aggregation: per-case oracle gain and what it cost.

Cases 85 and 88 were scanned exhaustively (every free unit); the rest of the
n>=101 band was scanned over the top-5 units by wire prize only, which is the
affordable shape (l281_deploy.py) rather than the oracle.  Both are labelled.
"""
import json
import pickle
import re
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import m53_l3_probe as m53                                        # noqa: E402
W = m53.W

aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
DB = pickle.load(open(_DIR / "l281_cache.pkl", "rb"))["db"]

secs = {}
for ln in open(_DIR / "l281_probe_heavy.log", encoding="utf-8", errors="ignore"):
    m = re.match(r"case\s+(\d+).*?(\d+) new LP solves.*?\((\d+)s\)", ln)
    if m:
        secs[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
for ln in open(_DIR / "l281_probe_c8588.log", encoding="utf-8", errors="ignore"):
    m = re.match(r"case\s+(\d+).*?(\d+) new LP solves.*?\((\d+)s\)", ln)
    if m:
        secs[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

full = {85, 88}
rows = []
for ci in sorted({k[1] for k in DB if k[0] == "rel2"}):
    ct = [DB[("ctrl", ci)]["cost"]] if DB.get(("ctrl", ci), {}).get("feas") else []
    ct += [v["cost"] for k, v in DB.items()
           if k[0] == "ctrlp" and k[1] == ci and v.get("feas")]
    cb = min(ct + [ANCH[ci]["cost"]])
    best = min([min(v["cost"], v.get("polished", float("inf")))
                for k, v in DB.items()
                if k[0] == "rel2" and k[1] == ci and v.get("feas")] + [cb])
    ns, sec = secs.get(ci, (0, 0))
    rows.append((ci, cb, best, ns, sec, ci in full))

wsum = sum(W[r[0]] for r in rows)
bt = sum(W[r[0]] * r[1] for r in rows) / wsum
g = sum(W[r[0]] * (r[1] - r[2]) for r in rows) / wsum
print(f"{'case':>5}{'scan':>7}{'solves':>8}{'sec':>7}{'base':>12}"
      f"{'best':>12}{'gain':>10}")
for ci, cb, best, ns, sec, isfull in sorted(rows, key=lambda r: -(r[1] - r[2])):
    print(f"{ci:>5}{'FULL' if isfull else 'top5':>7}{ns:>8}{sec:>7}"
          f"{cb:>12.6f}{best:>12.6f}{100.0 * (cb - best) / cb:>9.4f}%")
nz = sum(1 for r in rows if r[1] - r[2] > 1e-9)
print(f"\n== {len(rows)} cases (heavy band n>=101), weighted exp(n/12), "
      f"base {bt:.6f} ==")
print(f"   per-case oracle gain vs the polished control : {100.0 * g / bt:+.4f} %")
print(f"   cases with any gain at all                   : {nz}/{len(rows)}")
print(f"   LP solves                                    : "
      f"{sum(r[3] for r in rows)}")
print(f"   LP wall time                                 : "
      f"{sum(r[4] for r in rows)} s "
      f"= {sum(r[4] for r in rows) / max(len(rows), 1):.0f} s/case "
      f"(shipped budget ~1.5 s/case)")
