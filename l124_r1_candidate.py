"""OFFLINE (never shipped): L124 R1 — produce the MIB-ON portfolio as a candidate.

WHY R1 EXISTS. L123 measured the GLOBAL overlay (every profile forced to one
setting) and it flipped sign across the two disjoint samples: s1 +0.6486%,
s2 -0.3730%. But the per-case oracle over {ON, OFF} is POSITIVE and large on
BOTH samples -- s1 +0.9002%, s2 +1.2289% -- with the same 34/80 split each time.
So the mechanism is real; what failed was forcing one setting on every case.

The deployable form is therefore a TWIN: keep both variants in the pool and let
the portfolio proxy arbitrate. That needs no per-case classifier (the thing M56
and M79 killed) because the proxy is oracle-perfect on heterogeneous candidates
(M76 full-union bit-identical to the 2-way oracle; M77 efficiency 100.0%).

R1 asks the one question that decides whether to pay the cache-rebuild chain:
DOES THE PROXY ACTUALLY REALISE THAT ORACLE? It is not rhetorical -- M76 measured
a +0.384% 2-way oracle realising only 41.8%, and escape tiers realising ~5%.

This script only produces the candidate. `m77_oos_probe.py score` then drops it
into the 35-profile pool and arbitrates with the REAL `_proxy_metrics`, which is
what makes the answer trustworthy rather than a re-derivation of my own oracle.

⚠️ Runs the L124 PROBE binary via ICCAD_CONSTRUCTIVE_BIN. The shipping exe is
never touched, so all four offline caches stay valid (that is the whole point of
developing in a probe copy).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

import m67_oos_probe as m67                                  # noqa: E402
import m77_oos_probe as m77                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s2", choices=["s1", "s2"])
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--bucket", default="1", choices=["0", "1"])
    ap.add_argument("--bin", default="constructive_l124.exe")
    ap.add_argument("--nmin", type=int, default=101,
                    help="only solve this band; the oracle was measured on n>100")
    a = ap.parse_args()

    exe = _DIR / a.bin
    if not exe.exists():
        sys.exit(f"missing probe binary {exe}")
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(exe)
    os.environ["ICCAD_MIB_BUCKET"] = a.bucket
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)

    import optimizer_constructive as oc
    specs = m77._specs(a.sample)
    sel = [(i, ck, fk, L, n) for i, (ck, fk, L, n) in enumerate(specs)
           if n >= a.nmin]
    print(f"[cfg] {a.sample} @{a.cores}c  MIB_BUCKET={a.bucket}  bin={a.bin}")
    print(f"[cfg] {len(sel)}/{len(specs)} cases with n>={a.nmin}")

    import torch
    opt = oc.MyOptimizer(verbose=False)
    # group by file so each .th is read once, mirroring m67's own solve loop
    byf = {}
    for i, ck, fk, L, n in sel:
        byf.setdefault(fk, []).append((i, ck, L))
    rows, t0, done = [], time.time(), 0
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for i, ck, L in sorted(byf[fk], key=lambda t: t[2]):
            lay = m67._load_case(d, L)
            lay["base"], _dev = m67._baseline_official(lay)
            pos, dt, _err = m67._solve_one(opt, lay)
            rows.append(dict(oos_id=i, key=ck,
                             positions=None if pos is None else
                             [list(map(float, p)) for p in pos],
                             runtime_seconds=dt))
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(sel)}  ({el:.0f}s, eta "
                      f"{el / done * (len(sel) - done):.0f}s)")

    # name by the SETTING, not by intent: an earlier revision hardcoded "mibon"
    # and a --bucket 0 run silently overwrote 334s of --bucket 1 output.
    tag = "on" if a.bucket == "1" else "off"
    out = _DIR / f"results_L124_mib{tag}_{a.sample}_c{a.cores}.json"
    out.write_text(json.dumps(dict(
        submission_name=f"L124 MIB_BUCKET={a.bucket} {a.sample} @{a.cores}c",
        sample=a.sample, test_results=rows)), encoding="utf-8")
    print(f"wrote {out.name}  ({len(rows)} cases, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
