"""L274 -- does the SHIP binary's code default equal the arm that was measured?

Two comparisons per (case, profile), on the real deployment inputs (the spy sits
on _run_profile, so every env dict is the one the pool actually builds), compared
on raw stdout BYTES.

  G1  constructive_ship.exe  with NO env
   == constructive_l270.exe  with ICCAD_L269=1 ICCAD_L269_PROBES=1

      This is the one that matters. The L267-L269 evidence chain attaches to the
      PROBE run with those flags; G1 is what transfers it to a binary that reads
      no environment at all. Without it, the shipped default is a mechanism that
      merely resembles the measured one.

  G2  constructive_ship.exe  with ICCAD_L269=0   (the kill switch)
   == constructive.exe       (stock, shipped)

      Proves nothing ELSE from the probe source leaked into the shipped path --
      the L252 emitters, the L268 ordering block, the pack counter and the
      trial-loop restructure are all inert when the mechanism is off.

Both must be 100 %. G1 failing means the default is not the measured arm; G2
failing means the branch carries a side effect nobody priced.

  <python> l274_gate.py --cases 2
"""
import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

DIR = Path(__file__).parent
STOCK = DIR / "constructive.exe"
SHIP = DIR / "constructive_ship.exe"
PROBE = DIR / "constructive_l270.exe"
ARM = {"ICCAD_L269": "1", "ICCAD_L269_PROBES": "1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1")
    ap.add_argument("--nmin", type=int, default=101)
    ap.add_argument("--cases", type=int, default=2)
    ap.add_argument("--cores", type=int, default=48)
    a = ap.parse_args()

    for p in (STOCK, SHIP, PROBE):
        if not p.exists():
            print("!! missing {}".format(p.name))
            return 1
    print("[l274] ship={}  probe={}  arm={}".format(SHIP.name, PROBE.name, ARM))

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
    T = {"pairs": 0, "g1": 0, "g2": 0, "g1bad": [], "g2bad": []}

    def run(binary, inp, env):
        return subprocess.run([str(binary)], input=inp, capture_output=True,
                              text=True, env=env, timeout=TO)

    def spy(p, inp, bc):
        base = dict(os.environ)
        base.update(p)
        for k in list(ARM) + ["ICCAD_L252"]:
            base.pop(k, None)

        # G1: ship default  vs  probe + the measured flags
        env_arm = dict(base)
        env_arm.update(ARM)
        s = run(SHIP, inp, base)
        q = run(PROBE, inp, env_arm)
        # G2: ship kill switch  vs  stock
        env_off = dict(base)
        env_off["ICCAD_L269"] = "0"
        k = run(SHIP, inp, env_off)
        t = run(STOCK, inp, base)

        with lock:
            T["pairs"] += 1
            if s.stdout == q.stdout and s.returncode == q.returncode:
                T["g1"] += 1
            elif len(T["g1bad"]) < 5:
                T["g1bad"].append((bc, sorted(p.items())[:2]))
            if k.stdout == t.stdout and k.returncode == t.returncode:
                T["g2"] += 1
            elif len(T["g2bad"]) < 5:
                T["g2bad"].append((bc, sorted(p.items())[:2]))
        return oc._parse_output(t.stdout, bc)

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
                print("   {} n={} done, {} pairs".format(ck.split("/")[-1], n, T["pairs"]))
    finally:
        oc._run_profile = orig

    print()
    print("=" * 68)
    print("L274 SHIP-DEFAULT GATE   {} (case, profile) pairs".format(T["pairs"]))
    print("=" * 68)
    ok = True
    for k, lab in (("g1", "ship default   ==  probe + ICCAD_L269=1 PROBES=1"),
                   ("g2", "ship L269=0    ==  stock shipped binary")):
        n = T[k]
        good = (n == T["pairs"] and T["pairs"] > 0)
        ok = ok and good
        print("  {}  {}   {}/{}   {}".format(
            k.upper(), lab, n, T["pairs"], "PASS" if good else "FAIL"))
        for b in T[k + "bad"]:
            print("        mismatch n={} {}".format(*b))
    print("L274_GATE_RC={}".format(0 if ok else 1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
