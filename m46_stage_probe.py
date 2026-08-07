"""M46 stage-budget probe (OFFLINE, never shipped).

Where does the wall-setter runtime go? After M45 the big-case (n>100) wall is
MAX-bound and set by the FREE-stack build profiles (#18/#2/#23/#17...); the only
RF continuation is making those RUNS faster. This driver measures the stage
budget three ways:

  verify  - byte-compare constructive_m46.exe (the timing-instrumented COPY,
            gate off) against the shipped constructive.exe stdout on
            cases x profiles. Must be identical before any timing is trusted.
  diff    - differential env-knob runs on the SHIPPED exe (zero code change):
            drop REFINE / FREE_ASPECT / CA / FC / post-proc / frames and read
            the wall delta. Trajectory-dependent (approximate, ranking only).
  timing  - run constructive_m46.exe with ICCAD_STAGE_TIMING=1 and parse the
            stderr TIMING lines (exact per-stage budget, sampled inner split).

Usage:  python -u m46_stage_probe.py verify|diff|timing
Run with the machine otherwise idle (wall timings feed the M46 go/no-go).
"""
import os, sys, time, subprocess
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator
from optimizer_claude import _serialize_input
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc

EXE = str(_DIR / "constructive.exe")
EXE46 = str(_DIR / "constructive_m46.exe")

# top-wall cases (n=120/119/118/110) x the dominant max-setter profiles
CASE_IDS = [99, 98, 97, 89]
PROF_IDS = [18, 2, 23]          # CA0.6+FREE+GM+PIN / FREE+GM / FC+FREE+GM+PIN
REPS = 2                        # take min wall over reps

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
CASES = {}


def prep(case_ids):
    for idx in case_ids:
        if idx in CASES:
            continue
        s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
        at, b2b, p2b, pins, cons = inp
        n = int((at != -1).sum().item())
        base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
        otp = build_opt_target_pos(tp, cons, n)
        CASES[idx] = dict(n=n, txt=_serialize_input(n, at, b2b, p2b, pins, cons,
                                                    otp, gnn_hint=None))


def run(exe, txt, env_over, reps=REPS):
    best_dt, out, err = 1e18, "", ""
    for _ in range(reps):
        env = dict(os.environ); env.update(env_over)
        t0 = time.perf_counter()
        r = subprocess.run([exe], input=txt, capture_output=True, text=True,
                           env=env, timeout=300)
        dt = time.perf_counter() - t0
        if dt < best_dt:
            best_dt, out, err = dt, r.stdout, r.stderr
    return best_dt, out, err


def variants(prof):
    """(tag, env) differential variants for one profile dict."""
    p = dict(prof)
    v = [("full", dict(p))]
    v.append(("noREFINE", {**p, "ICCAD_NO_REFINE": "1"}))
    v.append(("REFINE1", {**p, "ICCAD_REFINE_ITERS": "1"}))
    v.append(("REFINE4", {**p, "ICCAD_REFINE_ITERS": "4"}))
    if "ICCAD_FREE_ASPECT" in p:
        q = dict(p); del q["ICCAD_FREE_ASPECT"]
        v.append(("noFREE", q))
    q = dict(p); hit = False
    for k in ("ICCAD_CLUSTER_ASPECT", "ICCAD_FREE_CLUSTER"):
        if k in q: del q[k]; hit = True
    if hit:
        v.append(("noCA/FC", q))
    v.append(("noPOST", {**p, "ICCAD_NO_COMPACT": "1", "ICCAD_NO_PUSH": "1",
                         "ICCAD_NO_BND_PUSH": "1"}))
    v.append(("1frame", {**p, "ICCAD_FRAME_SCALES": "1.05",
                         "ICCAD_FRAME_ASPECTS": "1.0"}))
    return v


def mode_verify():
    ok = True
    for ci in CASE_IDS:
        for k in PROF_IDS:
            env = dict(oc._PROFILES[k])
            _, o1, _ = run(EXE, CASES[ci]["txt"], env, reps=1)
            _, o2, _ = run(EXE46, CASES[ci]["txt"], env, reps=1)
            same = (o1 == o2)
            ok = ok and same
            print(f"case {ci} prof #{k}: stdout {'IDENTICAL' if same else 'DIFFERS <-- FAIL'}")
            # gate ON must not change stdout either (timing only writes stderr)
            _, o3, _ = run(EXE46, CASES[ci]["txt"],
                           {**env, "ICCAD_STAGE_TIMING": "1"}, reps=1)
            same3 = (o1 == o3)
            ok = ok and same3
            if not same3:
                print(f"  gate-ON stdout DIFFERS <-- FAIL")
    print("VERIFY", "PASS" if ok else "FAIL")


def mode_diff():
    for ci in CASE_IDS:
        n = CASES[ci]["n"]
        for k in PROF_IDS:
            vs = variants(oc._PROFILES[k])
            base = None
            row = []
            for tag, env in vs:
                dt, _, _ = run(EXE, CASES[ci]["txt"], env)
                if tag == "full":
                    base = dt
                row.append((tag, dt))
            cells = "  ".join(f"{tag}={dt:5.2f}s({100*(dt-base)/base:+5.1f}%)"
                              for tag, dt in row)
            print(f"case {ci} n={n} #{k:<2} {cells}", flush=True)


def mode_timing():
    for ci in CASE_IDS:
        n = CASES[ci]["n"]
        for k in PROF_IDS:
            env = dict(oc._PROFILES[k]); env["ICCAD_STAGE_TIMING"] = "1"
            dt, _, err = run(EXE46, CASES[ci]["txt"], env, reps=1)
            print(f"\n== case {ci} n={n} prof #{k} wall={dt:.2f}s ==")
            for line in err.splitlines():
                if line.startswith("TIMING"):
                    print("  " + line)


def mode_verify100():
    """Full-dataset byte-compare over 6 path-representative profiles (base /
    aspect / CA / FC / FA / OS-swap), plus per-size-band alpha from the paired
    solo dt (validates the uniform-alpha assumption of the rf46 projection)."""
    profs = [0, 12, 18, 23, 25, 34]
    prep(range(100))
    bands = [(0, 40), (40, 60), (60, 100), (100, 120)]
    ratio = {b: [] for b in bands}
    bad = []
    for ci in range(100):
        for k in profs:
            env = dict(oc._PROFILES[k])
            d1, o1, _ = run(EXE, CASES[ci]["txt"], env, reps=1)
            d2, o2, _ = run(EXE46, CASES[ci]["txt"], env, reps=1)
            if o1 != o2:
                bad.append((ci, k))
            n = CASES[ci]["n"]
            for lo, hi in bands:
                if lo < n <= hi:
                    ratio[(lo, hi)].append((d1, d2))
        if ci % 20 == 19:
            print(f"  ...{ci+1}/100 done, mismatches so far: {len(bad)}", flush=True)
    print(f"\nVERIFY100 {'PASS (600/600 byte-identical)' if not bad else 'FAIL: ' + str(bad)}")
    print("per-band alpha (sum dt_m46 / sum dt_shipped, solo runs):")
    for b, rs in ratio.items():
        if rs:
            s1 = sum(r[0] for r in rs); s2 = sum(r[1] for r in rs)
            print(f"  n in ({b[0]},{b[1]}]: alpha={s2/s1:.3f}  "
                  f"(shipped {s1:.1f}s -> m46 {s2:.1f}s over {len(rs)} runs)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diff"
    if mode != "verify100":
        prep(CASE_IDS)
    {"verify": mode_verify, "diff": mode_diff, "timing": mode_timing,
     "verify100": mode_verify100}[mode]()
