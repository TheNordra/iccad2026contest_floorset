"""L172 verdict -- score the in-set gates for the rebuilt LP depth map.

Four checks, and the third is the one that does not exist anywhere else in
this project's gate set:

  G1  DETERMINISM   det1 == det2 on cost AND positions, 100/100.
  G2  KILL SWITCH   ICCAD_SHAPE_LP_L147=0 reproduces the committed pre-L147
                    band bit-for-bit, so the escape hatch still works.
  G3  THE MAP ACTUALLY FIRED   no case may spend more LP passes than the new
                    map allows it, and the histogram must MOVE relative to the
                    old map's. A depth map that silently kept its old values
                    would pass every other check in this file while changing
                    nothing -- the exact shape of the ICCAD_* no-op the ledger
                    records twice.
  G4  QUALITY       vs the k=1 anchor measured in the SAME session.

  <python> l172_verdict.py
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

DIR = Path(__file__).parent
OLD_MAP = {}          # filled from l172_depthmap.SHIPPED
NEW_MAP = {}


PREFIX = "L172"


def load(tag):
    f = DIR / "results_{}_{}.json".format(PREFIX, tag)
    if not f.exists():
        return None
    return {r["test_id"]: r for r in json.load(open(f))["test_results"]}


def stats(tag):
    """[(n, kept, tier, passes_spent)] from ICCAD_SHAPE_LP_STATS."""
    f = DIR / "{}_{}_stats.txt".format(PREFIX.lower(), tag)
    if not f.exists():
        return []
    out = []
    for line in f.read_text().split("\n"):
        p = line.split()
        if len(p) >= 4:
            out.append(tuple(int(x) for x in p[:4]))
    return out


def wq(d):
    num = sum(math.exp(r["block_count"] / 12.0) * r["cost"] for r in d.values())
    den = sum(math.exp(r["block_count"] / 12.0) for r in d.values())
    return num / den


def main():
    global PREFIX
    if len(sys.argv) > 1:
        PREFIX = sys.argv[1]
    import l172_depthmap as M
    global OLD_MAP, NEW_MAP
    OLD_MAP = M.SHIPPED
    src = (DIR / "optimizer_constructive.py").read_text(encoding="utf-8")
    m = re.search(r"^_L157_DEPTH = \{.*?^\}", src, re.S | re.M)
    NEW_MAP = eval(m.group(0).split("=", 1)[1])

    d1, d2 = load("det1"), load("det2")
    k1, off = load("k1"), load("l147off")
    fails = []

    print("=" * 72)
    if d1 and d2:
        ids = sorted(set(d1) & set(d2))
        same_c = sum(1 for i in ids if d1[i]["cost"] == d2[i]["cost"])
        same_p = sum(1 for i in ids if d1[i].get("positions") == d2[i].get("positions"))
        ok = same_c == len(ids) and same_p == len(ids)
        fails += [] if ok else ["G1"]
        print("G1 determinism   cost {}/{}  positions {}/{}   {}"
              .format(same_c, len(ids), same_p, len(ids), "PASS" if ok else "FAIL"))
    else:
        fails.append("G1(not run)")
        print("G1 determinism   NOT RUN")

    ref = DIR / "results_L165_l147off.json"
    if off and ref.exists():
        R = {r["test_id"]: r for r in json.load(open(ref))["test_results"]}
        ids = sorted(set(off) & set(R))
        same = sum(1 for i in ids if off[i]["cost"] == R[i]["cost"])
        ok = same == len(ids)
        fails += [] if ok else ["G2"]
        print("G2 kill switch   {}/{} identical to results_L165_l147off.json   {}"
              .format(same, len(ids), "PASS" if ok else "FAIL"))
    else:
        fails.append("G2(not run)")
        print("G2 kill switch   NOT RUN")

    st = stats("det1")
    if st and d1:
        over = [(n, sp, NEW_MAP.get(n, 1)) for n, _k, _t, sp in st
                if sp > NEW_MAP.get(n, 1)]
        hist = dict(sorted(Counter(sp for _n, _k, _t, sp in st).items()))
        old_hist = {3: 66, 2: 24, 1: 10}
        moved = hist != dict(sorted(old_hist.items()))
        ok = not over and moved
        fails += [] if ok else ["G3"]
        print("G3 map fired     passes spent {}   (old map ran {})"
              .format(hist, dict(sorted(old_hist.items()))))
        print("                 cases spending more than the map allows: {}   {}"
              .format(len(over), "PASS" if ok else "FAIL"))
        if over:
            print("                 " + str(over[:10]))
        if not moved:
            print("                 !! histogram did NOT move -- the new map "
                  "may be a no-op")
    else:
        fails.append("G3(not run)")
        print("G3 map fired     NOT RUN (no stats file)")

    if d1 and k1:
        ids = sorted(set(d1) & set(k1))
        a = {i: k1[i] for i in ids}
        b = {i: d1[i] for i in ids}
        q = 100 * (wq(a) - wq(b)) / wq(a)
        mv = sum(1 for i in ids if b[i]["cost"] != a[i]["cost"])
        ws = sum(1 for i in ids if b[i]["cost"] > a[i]["cost"] + 1e-12)
        fe = sum(1 for i in ids if b[i].get("feasible", True))
        print("G4 quality       {:+.4f}% vs the k=1 anchor   {} moved / {} worse"
              "   feasible {}/{}".format(q, mv, ws, fe, len(ids)))
        print("                 (the OLD map read +0.6099%, 74 moved, 0 worse,")
        print("                  but on medians that no longer exist -- this")
        print("                  number is EXPECTED to be smaller and that is")
        print("                  the point: it is bought with less wall.)")
        if ws:
            print("                 !! {} cases got worse -- the old map had 0"
                  .format(ws))
    else:
        fails.append("G4(not run)")
        print("G4 quality       NOT RUN")

    print("=" * 72)
    # A gate that reports PASS because nothing ran is the same silent
    # no-op this whole line of work exists to prevent, so "not run"
    # counts as a failure, not as a skip.
    print("VERDICT: {}".format("ALL PASS" if not fails
                               else "FAIL " + ",".join(sorted(set(fails)))))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
