"""L144b - WHO blocks the compliant slot in the 21.7% that have none?

L144 established the shape of the boundary miss: over 745411 boundary-item
placements a bp==0 candidate existed 78.3% of the time and was taken
584001/584001 = 100% of those times, so "existed but not chosen" is ZERO and the
whole gap is the 21.7% where no compliant candidate exists at all. This driver
runs `constructive_l144b.exe`, whose ICCAD_BND_TRACE instrumentation now does a
post-mortem on exactly those placements:

  (e) no exact-edge candidate was generated/survived the bounds test at all
        econflict  two members of the same item demand different exact coords
        etoobig    the item does not fit inside the frame on that axis
        eclamp     the exact coord fell outside [0, frame-item] and was clamped
  (a) every exact-edge candidate overlapped a PREPLACED block
  (b) ... a placed block that is boundary-constrained on the SAME side
  (c) ... a boundary block of a different side
  (d) ... a plain non-boundary block

Attribution is on the LEAST-blocked compliant candidate, hardest blocker first
(preplaced > same-side > other-side > plain), so (d) means "nothing but ordinary
movable blocks was in the way" -- the only class a sequencing fix can address.

It also reports, per failing item, the item's along-edge extent against the
largest contiguous free run on the strip it would claim (L143's definition,
measured at pack time rather than on the final layout), and how many of the
failures are compound cluster items.

This file is NEW and READ-ONLY with respect to everything else; it never touches
constructive.exe / constructive_l144.* / l144_bnd_trace.py.

  <python> -u l144b_gate.py --sample s1 --cases 8 --profiles pool --gate
  <python> -u l144b_gate.py --sample s1 --cases 16 --profiles 0
  <python> -u l144b_gate.py --sample s1 --cases 16 --profiles 0 --min-n 101
"""
import argparse
import collections
import os
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

import torch                                                        # noqa: E402

import m67_oos_probe as m67                                         # noqa: E402
import m77_oos_probe as m77                                         # noqa: E402
import optimizer_constructive as oc                                 # noqa: E402
from optimizer_claude import _serialize_input                       # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402

EXE = _DIR / "constructive_l144b.exe"
SHIP = _DIR / "constructive.exe"


def _pct(a, b):
    return 100.0 * a / max(b, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--profiles", default="0")
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=10 ** 9)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--trace-on", action="store_true",
                    help="gate with ICCAD_BND_TRACE=1 on the probe side")
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = [s for s in m77._specs(a.sample)
             if a.min_n <= s[3] <= a.max_n][:a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    T = collections.Counter()
    runs = 0
    ns = []
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in byf[fk]:
            lay = m67._load_case(d, lay_id)
            tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            inp = _serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp)
            ns.append(n)
            idxs = (list(oc._pool_indices(n)) if a.profiles == "pool"
                    else [int(s) for s in a.profiles.split(",")])
            for i in idxs:
                env = dict(os.environ)
                env.update(oc._PROFILES[i])
                env.update(oc._profile_env(i, n))
                if a.gate:
                    x = subprocess.run([str(SHIP)], input=inp,
                                       capture_output=True, text=True,
                                       timeout=900, env=env)
                    e2 = dict(env)
                    if a.trace_on:          # stronger: trace ON must also be
                        e2["ICCAD_BND_TRACE"] = "1"   # stdout-identical
                    y = subprocess.run([str(EXE)], input=inp,
                                       capture_output=True, text=True,
                                       timeout=900, env=e2)
                    runs += 1
                    T["same" if x.stdout == y.stdout else "DIFF"] += 1
                    continue
                env["ICCAD_BND_TRACE"] = "1"
                r = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=900, env=env)
                runs += 1
                for line in r.stderr.splitlines():
                    if not line.startswith("BNDTRACE "):
                        continue
                    tag = line.split()[1]
                    for tok in line.split()[2:]:
                        if "=" in tok:
                            k, v = tok.split("=")
                            T[f"{tag}.{k}"] += float(v) if "." in v else int(v)

    print(f"\n=== L144b who-blocks: {a.sample}, {len(specs)} cases "
          f"(n {min(ns) if ns else 0}-{max(ns) if ns else 0}), "
          f"profiles={a.profiles}, {runs} solver runs ===\n")
    if a.gate:
        print(f"OFF-PATH GATE: identical {T['same']}/{runs}, "
              f"different {T['DIFF']}  -> "
              + ("PASS" if not T["DIFF"] else "*** FAIL ***"))
        return 0 if not T["DIFF"] else 1

    pl, av = T["cand.place"], T["cand.avail"]
    print(f"boundary-item placements            {pl}")
    print(f"  a bp==0 candidate EXISTED         {av:>8}  ({_pct(av, pl):.1f}%)")
    print(f"  chosen candidate WAS bp==0        {T['cand.took']:>8}  "
          f"({_pct(T['cand.took'], pl):.1f}%)")
    print(f"  NO compliant candidate at all     {pl - av:>8}  "
          f"({_pct(pl - av, pl):.1f}%)   <- classified below")

    F = T["who.fail"]
    print(f"\n--- classification of the {F} misses ---")
    rows = [
        ("(e) no exact-edge candidate offered", T["who.enogen"]),
        ("      members demand different coords", T["who.econflict"]),
        ("      item does not fit in the frame ", T["who.etoobig"]),
        ("      exact coord clamped out of range", T["who.eclamp"]),
        ("      other                           ", T["who.eother"]),
        ("(a) blocked by a PREPLACED block", T["who.apre"]),
        ("(b) blocked by a SAME-side boundary block", T["who.bsame"]),
        ("(c) blocked by an OTHER-side boundary block", T["who.cdiff"]),
        ("(d) blocked by a plain non-boundary block", T["who.dplain"]),
        ("    unexplained (should be 0)", T["who.unexpl"]),
    ]
    for lbl, v in rows:
        print(f"  {lbl:<46} {v:>8}  ({_pct(v, F):.1f}%)")
    print(f"  candidate-cap hit (probe truncation)         "
          f"{T['who.capped']:>8}")
    print(f"\n  compound (cluster) items among the misses   "
          f"{T['who.compound']:>8}  ({_pct(T['who.compound'], F):.1f}%)")
    print(f"    of them class e/a/b/c/d = "
          f"{T['blk.e']}/{T['blk.a']}/{T['blk.b']}/{T['blk.c']}/{T['blk.d']}")
    print(f"  corner items (>1 required side)             "
          f"{T['who.corner']:>8}  ({_pct(T['who.corner'], F):.1f}%)")

    bn = T["blk.n"]
    print(f"\n--- blocking BLOCKS on the least-blocked compliant candidate "
          f"({bn} total) ---")
    for lbl, k in (("preplaced", "blk.pre"), ("same-side boundary", "blk.same"),
                   ("other-side boundary", "blk.diff"),
                   ("plain non-boundary", "blk.plain")):
        print(f"  {lbl:<24} {T[k]:>8}  ({_pct(T[k], bn):.1f}%)")

    rn = T["run.n"]
    print(f"\n--- item along-edge extent vs largest free run on that strip "
          f"({rn} misses) ---")
    print(f"  a big enough run EXISTED   {T['run.fits']:>8}  "
          f"({_pct(T['run.fits'], rn):.1f}%)")
    print(f"  edge FRAGMENTED            {T['run.frag']:>8}  "
          f"({_pct(T['run.frag'], rn):.1f}%)")
    print(f"  run/extent  <0.25 {T['run.r00']}   0.25-0.5 {T['run.r25']}   "
          f"0.5-0.75 {T['run.r50']}   0.75-1.0 {T['run.r75']}   "
          f">=1.0 {T['run.r100']}")
    print(f"  mean run/extent = {T['run.sum'] / max(rn, 1):.3f}")
    print(f"  TOTAL free on the strip >= extent  {T['run.capfit']:>8}  "
          f"({_pct(T['run.capfit'], rn):.1f}%)   <- merely fragmented")
    print(f"  TOTAL free short of extent         {T['run.capshort']:>8}  "
          f"({_pct(T['run.capshort'], rn):.1f}%)   <- strip genuinely full")
    sn = T["strip.n"]
    print(f"\n--- blocks sitting in that strip ({sn} block-instances) ---")
    for lbl, k in (("preplaced", "strip.pre"),
                   ("same-side boundary", "strip.same"),
                   ("other-side boundary", "strip.diff"),
                   ("plain non-boundary", "strip.plain")):
        print(f"  {lbl:<24} {T[k]:>8}  ({_pct(T[k], sn):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
