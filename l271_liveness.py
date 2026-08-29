"""L271 -- liveness with a REASON, not just a count.

`l252_identity.py --flags X` sets every flag to the string "1". A binary that
only accepts a different VALUE therefore reads as byte-identical and the gate
reports arm B PASS -- which for a mechanism means "silent no-op". That is exactly
the shape this project keeps getting bitten by, so liveness is checked here with
the value actually used, and every identical pair has to come with an explanation.

For each (case, profile): run stock and probe, compare raw stdout bytes, and read
the probe's own trace --

    L252TRY <i> <ok>   per frame; the count of ok=0 BEFORE the first ok=1 is the
                       number of failed frames, i.e. the size of L271's antecedent
    L271HIT <q> <new> <incumbent>   the retry packed; both layout_scores
    L271MISS <q>                    the retry did not pack

A byte-identical pair is acceptable only if the trace EXPLAINS it. There are
three legitimate explanations and they must be told apart:

    EMPTY   no frame failed before the first success -> nothing to retry
    MISS    the retry fired and the tight frame still did not pack
    LOST    the retry packed but lost to the refined incumbent on layout_score

Anything else is a real no-op. Note the incumbent printed by L271HIT is the RAW
pack; it gets ORDER_SWAP/REFINE afterwards, so "the retry beat the incumbent"
is measured against an unrefined rival and overstates the retry.

  <python> l271_liveness.py --env ICCAD_L271=1 --cases 2
"""
import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

DIR = Path(__file__).parent
STOCK = DIR / "constructive.exe"
_ARGV = list(sys.argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="constructive_l271.exe")
    ap.add_argument("--env", action="append", default=[],
                    help="K=V, repeatable; the arm's real values")
    ap.add_argument("--cases", type=int, default=2)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--nmin", type=int, default=101)
    a = ap.parse_args(_ARGV[1:])
    PROBE = DIR / a.probe
    ARM = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        ARM[k] = v
    print("[l271l] probe={}  arm={}".format(PROBE.name, ARM))

    sys.argv = ["x"]
    import torch                                             # noqa: F401
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(STOCK)
    import optimizer_constructive as oc

    if len(list(oc._pool_indices(120))) != 51:
        print("!! not the shipped pool")
        return 1
    specs = [(ck, fk, L, n) for ck, fk, L, n in m77._specs(a.sample) if n >= a.nmin]
    specs.sort(key=lambda t: -t[3])
    specs = specs[:a.cases]
    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)
    lock = threading.Lock()
    T = {"pairs": 0, "same": 0, "diff": 0, "hits": 0, "miss": 0,
         "empty": 0, "s_miss": 0, "s_lost": 0, "unexplained": [],
         "fails": [], "improved": 0, "worsened": 0}

    def spy(p, inp, bc):
        env = dict(os.environ)
        env.update(p)
        for k in ARM:
            env.pop(k, None)
        s = subprocess.run([str(STOCK)], input=inp, capture_output=True,
                           text=True, env=env, timeout=TO)
        env_on = dict(env)
        env_on.update(ARM)
        env_on["ICCAD_L252"] = "1"
        b = subprocess.run([str(PROBE)], input=inp, capture_output=True,
                           text=True, env=env_on, timeout=TO)
        nfail, seen_ok, hit, miss, imp, wor = 0, False, 0, 0, 0, 0
        for line in b.stderr.splitlines():
            if line.startswith("L252TRY "):
                _, _i, ok, _s = line.split()
                if ok == "1":
                    seen_ok = True
                elif not seen_ok:
                    nfail += 1
            elif line.startswith("L271HIT "):
                q = line.split()
                hit += 1
                if float(q[2]) < float(q[3]) - 1e-9:
                    imp += 1
                else:
                    wor += 1
            elif line.startswith("L271MISS "):
                miss += 1
        with lock:
            T["pairs"] += 1
            T["hits"] += hit
            T["miss"] += miss
            T["improved"] += imp
            T["worsened"] += wor
            T["fails"].append(nfail)
            if s.stdout == b.stdout:
                T["same"] += 1
                # ORDER MATTERS. Modes 5/6 retry the SAME frame, so they do not
                # need a failed one -- keying on nfail==0 first would file a real
                # LOST as EMPTY and understate the blast radius.
                if hit > 0:
                    T["s_lost"] += 1
                elif miss > 0:
                    T["s_miss"] += 1
                elif nfail == 0:
                    T["empty"] += 1
                elif len(T["unexplained"]) < 8:
                    T["unexplained"].append((bc, nfail, hit, miss))
            else:
                T["diff"] += 1
        return oc._parse_output(s.stdout, bc)

    orig = oc._run_profile
    oc._run_profile = spy
    try:
        byf = {}
        for ck, fk, L, n in specs:
            byf.setdefault(fk, []).append((ck, L, n))
        for fk in sorted(byf):
            d = torch.load(m67._path_of(fk))
            for ck, L, n in byf[fk]:
                m67._solve_one(oc.MyOptimizer(verbose=False), m67._load_case(d, L))
                print("   {} n={} done ({} pairs)".format(ck.split("/")[-1], n, T["pairs"]))
    finally:
        oc._run_profile = orig

    nf = T["fails"]
    print()
    print("=" * 68)
    print("L271 LIVENESS   {} (case, profile) pairs".format(T["pairs"]))
    print("=" * 68)
    print("  output differs from stock   {}/{}".format(T["diff"], T["pairs"]))
    print("  byte-identical              {}/{}".format(T["same"], T["pairs"]))
    print("    EMPTY  no frame failed -> nothing to retry            {}".format(T["empty"]))
    print("    MISS   retry fired, the tight frame did not pack       {}".format(T["s_miss"]))
    print("    LOST   retry packed but lost to the refined incumbent  {}".format(T["s_lost"]))
    print("    UNEXPLAINED  (a real no-op)                            {}"
          .format(len(T["unexplained"])))
    for x in T["unexplained"]:
        print("          n={} failed_frames={} hits={} miss={}".format(*x))
    print()
    print("  antecedent size (failed frames before the first success):")
    print("    zero {}   mean {:.2f}   max {}".format(
        sum(1 for x in nf if x == 0), sum(nf) / max(len(nf), 1), max(nf or [0])))
    print("  retry packed {}   retry failed to pack {}".format(T["hits"], T["miss"]))
    if T["hits"]:
        print("  retry beat the RAW incumbent on layout_score   {}/{}  ({:.0f}%)"
              "   [raw, not refined -- overstates]".format(
            T["improved"], T["hits"], 100.0 * T["improved"] / T["hits"]))
    ok = (T["diff"] > 0 and not T["unexplained"])
    print()
    print("  VERDICT: {}".format(
        "LIVE, and every identical pair is explained by the trace"
        if ok else
        "SUSPECT -- there are identical pairs the antecedent does not explain"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
