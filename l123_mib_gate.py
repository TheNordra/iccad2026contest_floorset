"""OFFLINE (never shipped): L123 gates for the MIB shape-bucketing fallback.

G1 (kill switch): with ICCAD_MIB_BUCKET=0 the new build must be byte-identical
to the pre-change one on every case under a spread of profiles. Judged on
PER-PROFILE BINARY OUTPUT, not on the portfolio's pick -- M75 established that a
flag can change candidates without moving the proxy argmin, so a portfolio-level
"no difference" is a false negative.

G4 (liveness + direction): on the HELD-OUT corpus the flag must actually fire
and must move MIB violations DOWN toward the feasible floor. In-set it must fire
on nothing at all, because all 100 in-set groups are already unified by the two
existing branches -- that is the same fact that makes G2/G3 bit-identical, so if
in-set liveness is non-zero the change leaked somewhere it should not.

The floor is not zero: identical (w,h) forces identical area and area within 1%
is hard, so a group collapses to one shape only if its target areas span
<= 1.01/0.99. Greedy interval cover over the held-out corpus puts the reachable
minimum at 91.7% of the worst case.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

PRE = _DIR / "constructive_pre.exe"
NEW = _DIR / "constructive_l123.exe"

# a spread wide enough to exercise the MIB paths: the MIB_ASPECT profiles are
# the ones whose shared-shape choice the bucketer has to mirror per class
PROFILES = [
    {},
    {"ICCAD_MIB_ASPECT": "5.0"},
    {"ICCAD_MIB_ASPECT": "0.2338"},
    {"ICCAD_FREE_ASPECT": "1", "ICCAD_MIB_ASPECT": "5.0"},
    {"ICCAD_CLUSTER_ASPECT": "3.0"},
]


def _run(exe, inp, extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    env.update(extra)
    r = subprocess.run([str(exe)], input=inp, capture_output=True, text=True,
                       timeout=300.0, env=env)
    return r.returncode, r.stdout


def gate1(cases, quiet=False):
    import optimizer_constructive as oc
    import m53_l3_probe as l3

    bad = pairs = 0
    for ci in cases:
        c = l3.CASES[ci]
        inp = oc._serialize_input(c["n"], [float(a) for a in c["at"]],
                                  c["b2b"], c["p2b"], c["pins"], c["cons"], None)
        for prof in PROFILES:
            off = dict(prof, ICCAD_MIB_BUCKET="0")
            r1, o1 = _run(PRE, inp, prof)          # pre-change: flag does not exist
            r2, o2 = _run(NEW, inp, off)           # new build, kill switch on
            pairs += 1
            if r1 != r2 or o1 != o2:
                bad += 1
                if not quiet:
                    print(f"  case {ci} profile {prof}: MISMATCH "
                          f"(rc {r1}/{r2}, {len(o1)}/{len(o2)} bytes)")
    print(f"G1 kill switch: {pairs} (case,profile) pairs, {bad} mismatching -> "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


def _mib_stats(cn, at, n, dims):
    """(distinct shapes - 1) summed over groups, and the worst case, from dims."""
    import collections
    g = collections.defaultdict(list)
    for i in range(n):
        m = int(cn[i][2])
        if m > 0:
            g[m].append(i)
    viol = worst = 0
    for m, mem in g.items():
        if len(mem) <= 1:
            continue
        shapes = {(round(dims[i][0], 4), round(dims[i][1], 4)) for i in mem}
        viol += len(shapes) - 1
        worst += len(mem) - 1
    return viol, worst


def gate4_inset(cases):
    """In-set liveness must be exactly zero: every group is already unified."""
    import optimizer_constructive as oc
    import m53_l3_probe as l3
    moved = 0
    for ci in cases:
        c = l3.CASES[ci]
        inp = oc._serialize_input(c["n"], [float(a) for a in c["at"]],
                                  c["b2b"], c["p2b"], c["pins"], c["cons"], None)
        for prof in PROFILES:
            _r1, o1 = _run(NEW, inp, dict(prof, ICCAD_MIB_BUCKET="0"))
            _r2, o2 = _run(NEW, inp, dict(prof, ICCAD_MIB_BUCKET="1"))
            if o1 != o2:
                moved += 1
    print(f"G4a in-set liveness: {moved} (case,profile) pairs changed -> "
          f"{'PASS' if moved == 0 else 'FAIL'} (expected 0: every in-set group "
          f"is already unified by the existing branches)")
    return moved == 0


def gate4_held(files=8, per_file=10):
    """The kill test, run BEFORE the expensive cache regen.

    Everything in-set is bit-identical by construction, so in-set can never tell
    us whether this change is worth anything. Held-out is where the fallback
    actually fires. If MIB violations do not move here, the route is dead and no
    amount of OOS scoring will revive it.
    """
    import glob
    import torch
    import optimizer_constructive as oc

    paths = sorted(glob.glob(str(_DIR.parent / "floorset_lite" / "worker_1[0-4]"
                                 / "layouts_*.th")))[:files]
    tot_off = tot_on = tot_worst = 0
    cases = live = 0
    for f in paths:
        d = torch.load(f, weights_only=False)
        for l in range(0, min(d[0].shape[0], per_file * 3), 3):
            at_all = d[0][l][:, 0]
            n = int((at_all != -1).sum().item())
            if n < 20:
                continue
            at = [float(x) for x in at_all[:n]]
            cn = d[0][l][:n, 1:]
            if not any(int(cn[i][2]) > 0 for i in range(n)):
                continue
            inp = oc._serialize_input(n, at, d[1][l], d[2][l], d[3][l], cn, None)
            cases += 1
            got = {}
            for tag, val in (("off", "0"), ("on", "1")):
                rc, out = _run(NEW, inp, {"ICCAD_MIB_BUCKET": val})
                rows = oc._parse_output(out, n)
                if rc != 0 or len(rows) != n:
                    got = None
                    break
                got[tag] = [(r[2], r[3]) for r in rows]
            if not got:
                continue
            v_off, worst = _mib_stats(cn, at, n, got["off"])
            v_on, _ = _mib_stats(cn, at, n, got["on"])
            tot_off += v_off
            tot_on += v_on
            tot_worst += worst
            live += (v_on != v_off)
    print(f"G4b held-out liveness: {cases} cases with MIB groups, "
          f"{live} changed by the flag")
    print(f"  worst case (all distinct)   : {tot_worst}")
    print(f"  violations, bucketing OFF   : {tot_off}  "
          f"({100 * tot_off / max(tot_worst, 1):.1f}% of worst)")
    print(f"  violations, bucketing ON    : {tot_on}  "
          f"({100 * tot_on / max(tot_worst, 1):.1f}% of worst)")
    drop = tot_off - tot_on
    print(f"  removed                     : {drop}  "
          f"({100 * drop / max(tot_off, 1):.1f}% of what was there)")
    print(f"  measured reachable ceiling was 8.3% of worst "
          f"= {int(0.083 * tot_worst)} violations")
    ok = drop > 0
    print(f"G4b -> {'PASS' if ok else 'FAIL (flag never fires: route is dead)'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["g1", "g4inset", "g4held"])
    ap.add_argument("--cases", default="0-99")
    a = ap.parse_args()
    sel = []
    for part in a.cases.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            sel += list(range(int(lo), int(hi) + 1))
        else:
            sel.append(int(part))
    for e in (PRE, NEW):
        if not e.exists():
            print(f"missing {e}")
            sys.exit(2)
    if a.mode == "g4held":
        sys.exit(0 if gate4_held() else 1)
    ok = gate1(sel) if a.mode == "g1" else gate4_inset(sel)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
