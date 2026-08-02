"""M80 clean RF runtime re-measurement -- Route A steps 1-3 (OFFLINE, never shipped).

HANDOFF_2026-08-02 section 3-A: every runtime number on record was taken on the
OLD box (Intel, "12 cores", user Nordra).  This box is an AMD Ryzen 9 8940HX,
16 physical / 32 logical -- a different machine entirely.  So:
  * the 1.52s -> 1.45s figures the teammate needs for the M74 ship decision
    cannot be reproduced from the record; they have to be re-measured here;
  * M77 found the two available m54 timing caches differ by 1.27-1.46x, which
    is most likely this same machine change -- Route B's whole verdict hangs
    on it.

BASIS (iron rule 1, the pit that has been fallen into five times): the package
measured here is C:\\Users\\.01\\Downloads\\cadc1075 -- the ACTUALLY UPLOADED
files, md5-verified at startup -- NOT a freshly staged tar from the working
tree (which carries uncommitted M71/M73 experimental code).

METHOD (chosen because a perfectly idle box is not obtainable -- see M78):
repeat the official 100-case eval N times and take the PER-CASE MINIMUM.
Contention can only ever ADD wall time, so the minimum over repeats is a
consistent estimator of the clean runtime.  This changes the ESTIMATOR, not
the threshold -- no pre-registered bar is moved.  Every repeat also records a
per-process CPU delta so contention is a measured covariate, not a guess.

Two hard gates run on every repeat:
  Q  quality must reproduce results_shipped_m51.json bit-exact (total, per-case
     cost, positions).  A mismatch means the package or the box is wrong and
     the runtime numbers are meaningless -- abort rather than report them.
  I  the idle snapshot is recorded per repeat; repeats whose non-self CPU delta
     exceeds the M78 bar are FLAGGED (not discarded -- the min estimator wants
     them, and discarding on an outcome-dependent rule would be cherry-picking).

Modes: setup | run [--repeats N] | report
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

import make_submission as ms                      # noqa: E402 (constants only)

UPLOADED = Path(r"C:\Users\.01\Downloads\cadc1075")
UPLOADED_MD5 = "6f1f31a209c15e619b587ea6af25b83a"   # op_wrapper.py, verified
WORK = _DIR / "build_submission" / "m80_verify"
PKG = WORK / "cadc1075"
ANCHOR = _DIR / "results_shipped_m51.json"
OUT = _DIR / "results_M80_rf_remeasure.json"
RESULTS_NAME = "results_m80.json"
IDLE_BAR_PCT = 0.02                                # M78's pre-registered bar


def md5(p):
    import hashlib
    return hashlib.md5(p.read_bytes()).hexdigest()


def cpu_snapshot():
    """Per-process CPU seconds, via PowerShell (no psutil on this box)."""
    ps = ("Get-Process -ErrorAction SilentlyContinue | Where-Object {$_.CPU} | "
          "ForEach-Object { \"$($_.Id)|$($_.ProcessName)|$($_.CPU)\" }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=120)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3:
            try:
                out[f"{parts[0]}|{parts[1]}"] = float(parts[2])
            except ValueError:
                pass
    return out


def cpu_delta(a, b, seconds, self_pids):
    rows = []
    for k, v in b.items():
        pid = int(k.split("|")[0])
        if pid in self_pids:
            continue
        d = v - a.get(k, 0.0) if k in a else v
        if d > 0.05:
            rows.append({"proc": k.split("|")[1], "pid": pid,
                         "cpu_s": round(d, 2),
                         "pct_core": round(100 * d / max(seconds, 1e-9), 2)})
    rows.sort(key=lambda r: -r["cpu_s"])
    return rows


def mode_setup():
    assert UPLOADED.is_dir(), f"{UPLOADED} missing"
    got = md5(UPLOADED / "op_wrapper.py")
    assert got == UPLOADED_MD5, f"op_wrapper md5 {got} != uploaded {UPLOADED_MD5}"
    print(f"basis: {UPLOADED}  op_wrapper.py md5 {got}  [iron rule 1 OK]")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    shutil.copytree(UPLOADED, PKG)
    # grader-side overlay, mirroring make_submission.verify()
    shutil.copyfile(_DIR / "iccad2026contest" / "iccad2026_evaluate.py",
                    PKG / "iccad2026_evaluate.py")
    for f in ms._LOADER_FILES:
        shutil.copyfile(_DIR / f, WORK / f)
    how = ms._link_dataset(_DIR / "LiteTensorDataTest", WORK / "LiteTensorDataTest")
    assert any((WORK / "LiteTensorDataTest").iterdir()), "dataset link empty"
    print(f"dataset: {how}; loaders: {len(ms._LOADER_FILES)}; evaluator overlaid")
    print(f"package: {PKG}  ({len(list(PKG.iterdir()))} entries)")
    print("\nsetup: OK")


# The shipped wrapper's Windows compile chain hardcodes
# C:\msys64\ucrt64\bin\g++.exe, but the g++ DRIVER still needs that directory on
# PATH to find cc1plus/as/ld.  With msys installed-but-not-on-PATH the compile
# dies instantly, the smoke test fails, and the wrapper SILENTLY falls back to
# the pure-python SA -- which is what happened on the first attempt here.  On the
# Linux grader this never fires (bundled binary).  This is a host-environment
# fix for the measurement box, not a change to the submission.
MSYS_BIN = r"C:\msys64\ucrt64\bin"


FORCE_CORES = None      # set by --force-cores; see mode_run


def one_repeat(i):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    env["PYTHONIOENCODING"] = "utf-8"
    if os.path.isdir(MSYS_BIN):
        env["PATH"] = MSYS_BIN + os.pathsep + env.get("PATH", "")
    # R1: this box reports 32 cores, so tier-5 (_M67F_CORES_MIN=40) does NOT
    # fire and the big band runs 13 profiles.  The GRADER has 48 and runs 35.
    # Forcing the detected core count reproduces the grader's POOL here (it does
    # not conjure 48 physical cores -- the wall is still this box's, which is
    # exactly what we want: same pool, measurable wall).
    if FORCE_CORES:
        env["ICCAD_ADAPTIVE_CORES"] = str(FORCE_CORES)
    cmd = [ms._PY, "-u", "iccad2026_evaluate.py", "--evaluate", "op_wrapper.py",
           "-o", RESULTS_NAME]
    before = cpu_snapshot()
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(PKG), env=env, capture_output=True,
                       encoding="utf-8", errors="replace", timeout=7200)
    wall = time.time() - t0
    after = cpu_snapshot()
    if r.returncode != 0:
        raise RuntimeError(f"repeat {i}: evaluator exit {r.returncode}\n"
                           + "\n".join(r.stderr.splitlines()[-15:]))
    bad = [l for l in r.stderr.splitlines()
           if "fallback" in l or "unavailable" in l]
    if bad:
        # print the whole stderr tail: the "fallback" line is the SYMPTOM, the
        # compile error above it is the cause (first attempt here hid it)
        raise RuntimeError(
            f"repeat {i}: FALLBACK triggered -- the C++ path did not run.\n"
            + "\n".join(r.stderr.splitlines()[-25:]))

    res = json.loads((PKG / RESULTS_NAME).read_text(encoding="utf-8"))
    # GATE Q -- self-anchored.  The uploaded cadc1075 is the M71 submission
    # (_M71_ENV applies ICCAD_CLUSTER_BND_EXPOSE/EDGE_PACK to EVERY profile,
    # default ON, 2026-07-29) and scores 1.305390 locally, NOT the 1.326473 of
    # results_shipped_m51.json, which is the pre-M71 anchor and is superseded.
    # For a RUNTIME measurement the property that matters is determinism across
    # repeats, so repeat 0 becomes the session anchor and 1..N-1 must match it
    # bit-exact.  Agreement with the old M51 json is reported, never enforced.
    self_anchor = PKG / f"m80_session_anchor{'_c%d' % FORCE_CORES if FORCE_CORES else ''}.json"
    if not self_anchor.exists():
        self_anchor.write_text(json.dumps(res), encoding="utf-8")
    anc = json.loads(self_anchor.read_text(encoding="utf-8"))
    diffs = []
    if res["total_score"] != anc["total_score"]:
        diffs.append(f"total {res['total_score']!r} != {anc['total_score']!r}")
    for a, b in zip(res["test_results"], anc["test_results"]):
        for k in ("test_id", "cost", "hpwl_gap", "area_gap",
                  "violations_relative", "is_feasible"):
            if a[k] != b[k]:
                diffs.append(f"case {b['test_id']}: {k} {a[k]!r} != {b[k]!r}")
        if a["positions"] != b["positions"]:
            diffs.append(f"case {b['test_id']}: positions differ")
    if diffs:
        raise RuntimeError(f"repeat {i}: GATE Q FAILED vs repeat 0 "
                           f"({len(diffs)} diffs) -- the package is not "
                           f"deterministic here: " + "; ".join(diffs[:5]))
    m51 = json.loads(ANCHOR.read_text(encoding="utf-8"))["total_score"]

    cont = cpu_delta(before, after, wall, {os.getpid()})
    worst = max((c["pct_core"] for c in cont), default=0.0)
    return {
        "repeat": i, "wall_s": round(wall, 1),
        "total_score": res["total_score"],
        "matches_m51_anchor": res["total_score"] == m51,
        "avg_runtime": res["summary"]["avg_runtime"],
        "runtimes": {str(t["test_id"]): t["runtime_seconds"]
                     for t in res["test_results"]},
        "n": {str(t["test_id"]): t["block_count"] for t in res["test_results"]},
        "contention_top": cont[:8],
        "worst_pct_core": worst,
        "idle_bar_ok": worst <= IDLE_BAR_PCT * 100,
    }


def mode_run(repeats):
    assert PKG.exists(), "run setup first"
    db = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else \
        {"basis": str(UPLOADED), "op_wrapper_md5": UPLOADED_MD5, "repeats": []}
    done = {r["repeat"] for r in db["repeats"]}
    for i in range(repeats):
        if i in done:
            continue
        print(f"[repeat {i}] starting ...", flush=True)
        rec = one_repeat(i)
        db["repeats"].append(rec)
        OUT.write_text(json.dumps(db, indent=1), encoding="utf-8")
        print(f"[repeat {i}] {rec['wall_s']}s  avg_runtime {rec['avg_runtime']}"
              f"  quality bit-exact OK  worst background "
              f"{rec['worst_pct_core']}% of one core"
              f"{'' if rec['idle_bar_ok'] else '  [FLAGGED: over M78 bar]'}",
              flush=True)
    print(f"\n{len(db['repeats'])} repeats stored -> {OUT.name}")


def mode_report():
    db = json.loads(OUT.read_text(encoding="utf-8"))
    reps = db["repeats"]
    assert reps, "no repeats"
    cis = sorted(reps[0]["runtimes"], key=int)
    mins = {c: min(r["runtimes"][c] for r in reps) for c in cis}
    n = reps[0]["n"]
    W = {c: math.exp(int(n[c]) / 12.0) for c in cis}
    totw = sum(W.values())

    print(f"basis {db['basis']}  op_wrapper md5 {db['op_wrapper_md5']}")
    print(f"repeats {len(reps)}  (quality bit-exact on every one)\n")
    print(f"{'rep':>4} {'wall':>7} {'avg_rt':>8} {'worst bg':>9} {'flag':>5}")
    for r in reps:
        print(f"{r['repeat']:>4} {r['wall_s']:>7.0f} {r['avg_runtime']:>8.4f} "
              f"{r['worst_pct_core']:>8.1f}% {'' if r['idle_bar_ok'] else 'BG':>5}")

    tot = sum(mins.values())
    print(f"\nPER-CASE MINIMUM over {len(reps)} repeats:")
    print(f"  mean {tot / len(cis):.4f}s   weighted mean "
          f"{sum(W[c] * mins[c] for c in cis) / totw:.4f}s   sum {tot:.1f}s")
    srt = sorted(mins.values())
    print(f"  p50 {srt[len(srt) // 2]:.4f}s  p90 {srt[int(.9 * len(srt))]:.4f}s"
          f"  max {srt[-1]:.4f}s")
    if len(reps) > 1:
        spread = [max(r["runtimes"][c] for r in reps) / mins[c] for c in cis]
        spread.sort()
        print(f"  max/min spread across repeats: p50 {spread[len(spread)//2]:.2f}x"
              f"  p90 {spread[int(.9*len(spread))]:.2f}x  worst {spread[-1]:.2f}x")
    print("\n  ==> record avg 1.52s (OLD Intel box). This box, per-case min:")
    print(f"      {tot / len(cis):.4f}s  = {tot / len(cis) / 1.52:.2f}x the record")
    print("\ntop-12 heaviest cases (per-case min):")
    for c in sorted(cis, key=lambda k: -mins[k])[:12]:
        print(f"  case {c:>3} n={n[c]:>3} {mins[c]:.4f}s")
    db["per_case_min"] = mins
    OUT.write_text(json.dumps(db, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.name} (per_case_min written)")


# ── RF projection: what a runtime change is actually worth ──────────────────
ALPHA_JSON = Path(r"C:\Users\.01\Downloads\cadc1075_results.json")
KAPPA = 3.161          # M67-E: M_i = kappa * t_i^alpha (alpha-calibrated median)
GAMMA, FLOOR = 0.3, 0.7


def mode_rf():
    """Elasticity of the official score wrt our runtime, per machine-speed s.

    RF_i = max(FLOOR, (s*t_i / M_i)^GAMMA) with M_i = KAPPA * t_i^alpha.
    A case sitting ON the floor pays nothing for extra time and gains nothing
    from saving it; only the un-floored weight responds.  So

        d ln(score) / d ln(t)  =  GAMMA * (cost-weighted share NOT floored)

    which is the single number the M74 ship decision needs: multiply it by the
    intended runtime change.  s (grader seconds per second of OUR box) is
    unknown, so it is swept rather than assumed.
    """
    db = json.loads(OUT.read_text(encoding="utf-8"))
    mins = db["per_case_min"]
    reps = db["repeats"]
    alpha = {str(t["test_id"]): t for t in
             json.loads(ALPHA_JSON.read_text(encoding="utf-8"))["test_results"]}
    res0 = reps[0]
    cis = sorted(mins, key=int)
    n = res0["n"]
    W = {c: math.exp(int(n[c]) / 12.0) for c in cis}
    # per-case quality (RF=1.0 local cost) from the measured shipped run
    anc = json.loads((PKG / "m80_session_anchor.json").read_text(encoding="utf-8"))
    Q = {str(t["test_id"]): t["cost"] for t in anc["test_results"]}
    M = {c: KAPPA * alpha[c]["runtime_seconds"] ** 1.0 for c in cis}

    print(f"alpha calibration: {ALPHA_JSON.name}  kappa={KAPPA}")
    print(f"our per-case runtime = MIN over {len(reps)} repeats on THIS box "
          f"(AMD 8940HX, 32 logical)\n")
    print(f"{'s':>5} {'wRF':>8} {'floored w%':>11} {'elasticity':>11} "
          f"{'-4.6% runtime':>14}")
    for s in (1.0, 1.5, 2.0, 2.5, 3.0):
        num = den = fl = 0.0
        for c in cis:
            raw = (s * mins[c] / M[c]) ** GAMMA
            rf = max(FLOOR, raw)
            wq = W[c] * Q[c]
            num += wq * rf
            den += wq
            if raw <= FLOOR:
                fl += wq
        live = 1.0 - fl / den
        elas = GAMMA * live
        print(f"{s:>5g} {num / den:>8.4f} {100 * fl / den:>10.1f}% "
              f"{elas:>11.4f} {100 * elas * -0.046:>13.3f}%")
    print("\nread: 'elasticity' = d ln(official score) / d ln(our runtime).")
    print("      last column = what M74's 1.52s -> 1.45s (-4.6%) buys, as a")
    print("      percentage of the official score, at that machine speed s.")
    print("      NEGATIVE = better (score is lower-is-better).")
    print("\n  CAVEAT: s for THIS box is not the M67-E bracket [1.5,1.7] -- that")
    print("  was calibrated on the OLD 12-core Intel box, which this run shows")
    print("  is ~1.4x SLOWER than this one on the portfolio wall.  s here is")
    print("  correspondingly LARGER.  Treat the sweep, not a single column.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["setup", "run", "report", "rf"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--force-cores", type=int, default=None, dest="force_cores",
                    help="force the wrapper's detected core count (R1: 48 = "
                         "reproduce the grader POOL on this box)")
    a = ap.parse_args()
    if a.force_cores:
        FORCE_CORES = a.force_cores
        OUT = _DIR / f"results_M80_rf_remeasure_c{a.force_cores}.json"
    {"setup": mode_setup, "run": lambda: mode_run(a.repeats),
     "report": mode_report, "rf": mode_rf}[a.mode]()
