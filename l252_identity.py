"""L252 gate -- constructive_l252.exe IS the shipped placer.

Two comparisons per (case, profile), on the REAL deployment inputs (the spy sits
on _run_profile, so every env dict is the one the pool actually builds):

    A  stock            vs  probe, ICCAD_L252 unset   -> the recompile is clean
    B  stock            vs  probe, ICCAD_L252=1       -> the emitters are stderr-only

B is the one that matters, because B is the configuration the measurement runs
in. A is kept because if B fails, A says whether the cause is the compile or the
emitter. Compared on raw stdout BYTES, not on parsed positions.

  <python> l252_identity.py --cases 2
"""
import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

DIR = Path(__file__).parent
STOCK = DIR / "constructive.exe"
PROBE = DIR / "constructive_l252.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--nmin", type=int, default=101)
    ap.add_argument("--cases", type=int, default=2)
    ap.add_argument("--cores", type=int, default=48)
    ap.add_argument("--probe", default="constructive_l252.exe")
    ap.add_argument("--flags", default="ICCAD_L252",
                    help="comma-separated env flags to turn ON for arm B")
    a = ap.parse_args()
    global PROBE
    PROBE = DIR / a.probe
    FLAGS = [f for f in a.flags.split(",") if f]
    print("[l252] probe={}  flags={}".format(PROBE.name, FLAGS))

    for p in (STOCK, PROBE):
        if not p.exists():
            print("!! missing {}".format(p))
            return 1

    sys.argv = ["x"]
    # ORDER MATTERS -- m67_oos_probe.py:61-63 strips every ICCAD_* at import.
    import torch                                             # noqa: F401
    import m67_oos_probe as m67
    import m77_oos_probe as m77
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    os.environ["ICCAD_CONSTRUCTIVE_BIN"] = str(STOCK)
    import optimizer_constructive as oc

    npool = len(list(oc._pool_indices(120)))
    print("[l252] pool at n=120: {} profiles".format(npool))
    if npool != 51:
        print("!! not the shipped pool -- refusing to gate a config we do not ship")
        return 1

    specs = [(ck, fk, L, n) for ck, fk, L, n in m77._specs(a.sample) if n >= a.nmin]
    specs.sort(key=lambda t: -t[3])
    specs = specs[:a.cases]
    print("[l252] gate on {} cases: {}".format(
        len(specs), ", ".join("{}(n={})".format(ck.split("/")[-1], n)
                              for ck, _f, _L, n in specs)))

    TO = getattr(oc, "_PROFILE_TIMEOUT", 300.0)
    lock = threading.Lock()
    tally = {"pairs": 0, "A_same": 0, "B_same": 0, "A_bad": [], "B_bad": []}

    def run(binary, inp, env):
        return subprocess.run([str(binary)], input=inp, capture_output=True,
                              text=True, env=env, timeout=TO)

    def spy(p, inp, block_count):
        env = dict(os.environ)
        env.update(p)
        for f in FLAGS:
            env.pop(f, None)
        s = run(STOCK, inp, env)
        pa = run(PROBE, inp, env)
        env_on = dict(env)
        for f in FLAGS:
            env_on[f] = "1"
        pb = run(PROBE, inp, env_on)
        with lock:
            tally["pairs"] += 1
            if s.stdout == pa.stdout and s.returncode == pa.returncode:
                tally["A_same"] += 1
            elif len(tally["A_bad"]) < 5:
                tally["A_bad"].append((block_count, sorted(p.items())[:2]))
            if s.stdout == pb.stdout and s.returncode == pb.returncode:
                tally["B_same"] += 1
            elif len(tally["B_bad"]) < 5:
                tally["B_bad"].append((block_count, sorted(p.items())[:2]))
        return oc._parse_output(s.stdout, block_count)

    orig = oc._run_profile
    oc._run_profile = spy
    try:
        byf = {}
        for ck, fk, L, n in specs:
            byf.setdefault(fk, []).append((ck, L, n))
        for fk in sorted(byf):
            d = torch.load(m67._path_of(fk))
            for ck, L, n in byf[fk]:
                lay = m67._load_case(d, L)
                opt = oc.MyOptimizer(verbose=False)
                m67._solve_one(opt, lay)
                print("   {} n={} done, {} pairs so far"
                      .format(ck.split("/")[-1], n, tally["pairs"]))
    finally:
        oc._run_profile = orig

    print()
    print("=" * 60)
    print("L252 IDENTITY GATE   {} (case, profile) pairs".format(tally["pairs"]))
    print("=" * 60)
    for k, lab in (("A", "flag OFF  (recompile is clean)"),
                   ("B", "flag ON   (emitters are stderr-only)")):
        ok = tally[k + "_same"]
        print("  {}  {}   {}/{}   {}".format(
            k, lab, ok, tally["pairs"],
            "PASS" if ok == tally["pairs"] and tally["pairs"] else "FAIL"))
        for b in tally[k + "_bad"]:
            print("        mismatch n={} {}".format(*b))
    rc = 0 if (tally["pairs"] and tally["A_same"] == tally["pairs"]
               and tally["B_same"] == tally["pairs"]) else 1
    print("L252_IDENTITY_RC={}".format(rc))
    return rc


if __name__ == "__main__":
    sys.exit(main())
