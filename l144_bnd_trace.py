"""L144 - WHERE in the pipeline are the held-out boundary violations created?

L142 killed the scoring weight (inert in both directions) and L143, once its
availability test was corrected to the STRIP the block would actually occupy,
found that only 47.6% of the misses have a big enough gap even in the final
layout. Meanwhile the packer already sorts boundary items FIRST (`bscore` is the
primary sort key, `constructive.cpp:1815`), already prices bp in the greedy,
already weights `150000*bv` in `layout_score`, and already runs three boundary
repair passes. So before designing any mechanism, this asks the only question
that discriminates between the remaining explanations:

    of the boundary items the packer places, how often did a compliant (bp==0)
    candidate EXIST, and how often was it CHOSEN?

  * exists and chosen, yet the case still violates  -> created downstream
    (compaction / push / the frame that won), fix is in those stages;
  * exists and NOT chosen                           -> a scoring trade-off after
    all, and L142's inertness needs re-explaining;
  * does not exist                                  -> the edge is genuinely
    occupied at pack time, and the fix is upstream: frame shape, block aspect,
    or reserving the strip.

Driven with `constructive_l144.exe` (the L144 probe binary: adds stderr counters
only, ICCAD_BND_TRACE off => bit-identical). The shipping exe is never touched.

  <python> -u l144_bnd_trace.py --sample s1 --cases 40
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

EXE = _DIR / os.environ.get("L144_EXE", "constructive_l144.exe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--profiles", default="0",
                    help="comma-separated pool indices, or 'pool' for all")
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--gate", action="store_true",
                    help="off-path gate: probe binary with the trace OFF must be "
                         "byte-identical to the shipping exe")
    a = ap.parse_args()

    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    specs = m77._specs(a.sample)[:a.cases]
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        byf[fk].append((ck, lay_id, n))

    T = collections.Counter()
    runs = 0
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in byf[fk]:
            lay = m67._load_case(d, lay_id)
            tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            inp = _serialize_input(n, lay["at"], lay["b2b"], lay["p2b"],
                                   lay["pins"], lay["cons"], otp)
            idxs = (list(oc._pool_indices(n)) if a.profiles == "pool"
                    else [int(s) for s in a.profiles.split(",")])
            for i in idxs:
                env = dict(os.environ)
                env.update(oc._PROFILES[i])
                env.update(oc._profile_env(i, n))
                if a.gate:
                    x = subprocess.run([str(_DIR / "constructive.exe")],
                                       input=inp, capture_output=True,
                                       text=True, timeout=600, env=env)
                    y = subprocess.run([str(EXE)], input=inp,
                                       capture_output=True, text=True,
                                       timeout=600, env=env)
                    runs += 1
                    T["same" if x.stdout == y.stdout else "DIFF"] += 1
                    continue
                env["ICCAD_BND_TRACE"] = "1"
                r = subprocess.run([str(EXE)], input=inp, capture_output=True,
                                   text=True, timeout=600, env=env)
                runs += 1
                for line in r.stderr.splitlines():
                    if line.startswith("BNDTRACE cand"):
                        for tok in line.split()[2:]:
                            if "=" in tok:
                                k, v = tok.split("=")
                                T[k] += int(v)
                    elif line.startswith("BNDTRACE sel_prepost"):
                        p = line.split()
                        T["sel_pre_compact"] += int(p[2])
                        T["sel_post_compact"] += int(p[3])
                        T["sel_post_push"] += int(p[4])

    print(f"\n=== L144 boundary trace: {a.sample}, {len(specs)} cases, "
          f"{runs} solver runs ===\n")
    if a.gate:
        print(f"OFF-PATH GATE: identical {T['same']}/{runs}, "
              f"different {T['DIFF']}  -> "
              + ("PASS" if not T["DIFF"] else "*** FAIL ***"))
        return 0 if not T["DIFF"] else 1
    pl, av, tk = T["place"], T["avail"], T["took"]
    print(f"boundary-item placements            {pl}")
    print(f"  a bp==0 candidate EXISTED         {av:>8}  "
          f"({100 * av / max(pl, 1):.1f}%)")
    print(f"  the chosen candidate WAS bp==0    {tk:>8}  "
          f"({100 * tk / max(pl, 1):.1f}%)")
    print(f"  existed but was NOT chosen        {av - tk:>8}  "
          f"({100 * (av - tk) / max(pl, 1):.1f}%)")
    print(f"  no compliant candidate at all     {pl - av:>8}  "
          f"({100 * (pl - av) / max(pl, 1):.1f}%)")
    print(f"\nviolations per run, summed over runs:")
    print(f"  right after pack_in_frame         {T['pre']:>8}")
    print(f"  after the 3 repair passes         {T['post']:>8}"
          f"   ({T['post'] - T['pre']:+d})")
    print(f"  (over {T['frames']} run_frame calls)")
    print(f"\nselected layout, one line per solve:")
    print(f"  before compaction                 {T['sel_pre_compact']:>8}")
    print(f"  after compaction                  {T['sel_post_compact']:>8}"
          f"   ({T['sel_post_compact'] - T['sel_pre_compact']:+d})")
    print(f"  after hpwl_push + slide           {T['sel_post_push']:>8}"
          f"   ({T['sel_post_push'] - T['sel_post_compact']:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
