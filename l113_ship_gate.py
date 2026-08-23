"""OFFLINE (never shipped): L113 package-shaped gate for the >=40-core config.

WHY THIS EXISTS
---------------
Three shipped mechanisms are cores-gated and fire only at >= 40 detected cores:

    _m80_active()        M80 knob-cloud tier   (8 profiles)
    _M67F_CORES_MIN      tier-5 pool restore
    _route_a_default()   route A frame queue

`make_submission.py verify` runs the official command with `ICCAD_*` STRIPPED, on
whatever box you are on. On a <40-core dev box that means none of the three ever
fire, so verify validates the configuration the grader will NOT run. That blind
spot already cost one shipping-grade bug: route A resolved its binary as a
hardcoded `constructive_l108.exe`, a file the package does not contain, so on the
48-core grader every frame task raised FileNotFoundError -> every profile
returned None -> the case sank to the SA fallback. Measured on test 99 in a
package-shaped tree: 10.0000 instead of 1.2773, with a clean local verify.

WHAT IT DOES
------------
Stages the real package, extracts it, overlays the evaluator + loader closure +
a dataset link exactly as verify does, then runs the OFFICIAL command with
ICCAD_ADAPTIVE_CORES forced -- so the same package the grader unpacks runs the
same configuration the grader runs. Checks:

    G1 evaluator exits 0
    G2 stderr carries no fallback / "all profiles failed" / unavailable line
    G3 every case feasible
    G4 cost AND positions bit-equal to the >=40-core anchor
    G5 route A peak concurrency <= its queue size
    G6 a usable C++ binary is present (an on-site compile product, or the
       bundled Linux binary)

Usage:
    python l113_ship_gate.py [--cores 48] [--anchor results_M80_48c_anchor.json]
                             [--test-id N] [--keep]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import make_submission as ms

_REPO = Path(__file__).resolve().parent
_GATE = _REPO / "build_submission" / "gate_cores"
_RESULTS_NAME = "results_l113_gate.json"
# L153: "[constructive]" added. Every print carrying that tag in
# optimizer_constructive.py is a degradation or a failure notice (compile
# failed, binary unavailable, profiles failed, SA/row fallback, LP raised) --
# there is no benign one. Without it G2 saw only the LAST line of a cascade:
# on 2026-08-20 this gate reported "binary unavailable; falling back to python
# SA" and swallowed the two `[constructive] <g++> -O3 failed:` lines directly
# above it that named the cause. The cause was that C:\msys64\ucrt64\bin was
# not on PATH, so g++ exited 1 with EMPTY stderr; the run then scored
# 9.999916545892749 with a clean 100/100 feasible table.
_BAD_RE = ("fallback", "unavailable", "all profiles failed", "[constructive]")


def _load(p):
    j = json.loads(Path(p).read_text(encoding="utf-8"))
    return j, {r["test_id"]: r for r in j["test_results"]}


_MSYS = Path(r"C:\msys64\ucrt64\bin")


def _cxx_preflight(env: dict) -> None:
    """The package ships NO Windows binary, so this gate has to compile on site.

    L153: it could not, and the way it could not is the point. The msys64 g++
    is installed and is the first candidate in `_ensure_compiled`, but its
    directory was not on PATH, so the exe failed to load its own DLLs and
    exited 1 with COMPLETELY EMPTY stderr. Every case then fell to the pure
    Python SA and the gate ran to completion printing a normal-looking
    `feasible=100/100` next to a total of 9.999916545892749.

    Probing here turns a 7-minute mystery into one line before the run. On the
    grader this branch never executes -- POSIX takes bin/constructive_linux and
    skips the compile entirely -- so this is purely about keeping the LOCAL
    gate meaningful.
    """
    if os.name != "nt":
        return
    def _runs(exe):
        try:
            return subprocess.run([exe, "--version"], capture_output=True,
                                  env=env, timeout=60).returncode == 0
        except Exception:
            return False
    if _runs("g++") or _runs(str(_MSYS / "g++.exe")):
        print("  c++ preflight: OK (a compiler answers --version)")
        return
    if (_MSYS / "g++.exe").exists():
        env["PATH"] = str(_MSYS) + os.pathsep + env.get("PATH", "")
        if _runs(str(_MSYS / "g++.exe")):
            print(f"  c++ preflight: prepended {_MSYS} to PATH (msys64 g++ "
                  "needs its own bin dir to load its DLLs)")
            return
    print("  c++ preflight: WARNING -- no working C++ compiler reachable. "
          "This gate is about to measure the pure-Python SA fallback, not the "
          "package. Put a g++ on PATH first.")


def run(cores: int, anchor: Path, test_id, keep: bool, extra_env=None) -> bool:
    print(f"== L113 package gate (ICCAD_ADAPTIVE_CORES={cores}) ==")
    if not anchor.exists():
        print(f"  FAIL: anchor {anchor.name} missing -- generate it first with the "
              f"official eval at ICCAD_ADAPTIVE_CORES={cores}")
        return False
    if not ms.stage():
        print("  FAIL: stage() failed")
        return False

    import shutil
    import tarfile
    if _GATE.exists():
        shutil.rmtree(_GATE)
    _GATE.mkdir(parents=True)
    with tarfile.open(ms._TAR, "r:gz") as tf:
        tf.extractall(_GATE)
    pkg = _GATE / ms._TEAM
    assert (pkg / "op_wrapper.py").exists(), "extracted package incomplete"

    shutil.copyfile(_REPO / "iccad2026contest" / "iccad2026_evaluate.py",
                    pkg / "iccad2026_evaluate.py")
    for f in ms._LOADER_FILES:
        shutil.copyfile(_REPO / f, _GATE / f)
    how = ms._link_dataset(_REPO / "LiteTensorDataTest", _GATE / "LiteTensorDataTest")
    if not any((_GATE / "LiteTensorDataTest").iterdir()):
        print("  FAIL: dataset link empty")
        return False
    print(f"  dataset: {how}; loaders: {len(ms._LOADER_FILES)}; evaluator overlaid")

    stats = _GATE / "route_a_peak.txt"
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    env["PYTHONIOENCODING"] = "utf-8"
    env["ICCAD_ADAPTIVE_CORES"] = str(cores)
    env["ICCAD_ROUTE_A_STATS"] = str(stats)
    _cxx_preflight(env)
    # L137: the ambient ICCAD_* strip above is deliberate (profile_audit.py:180)
    # -- the package must be measured on shipped defaults, not on whatever the
    # shell happens to carry. `--env K=V` re-admits ONE knob at a time and prints
    # it, so an A/B is explicit in the transcript instead of ambient.
    for kv in (extra_env or []):
        k, _, v = kv.partition("=")
        env[k] = v
        print(f"  env override: {k}={v}")
    cmd = [ms._PY, "-u", "iccad2026_evaluate.py", "--evaluate", "op_wrapper.py",
           "-o", _RESULTS_NAME]
    if test_id is not None:
        cmd += ["--test-id", str(test_id)]
    print(f"  running (cwd={pkg.name}): {' '.join(cmd[1:])}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(pkg), env=env, capture_output=True,
                       encoding="utf-8", errors="replace", timeout=7200)
    print(f"  eval done in {time.time() - t0:.0f}s (exit {r.returncode})")

    errs = []
    # G1 ---------------------------------------------------------------------
    if r.returncode != 0:
        errs.append(f"G1 evaluator exit {r.returncode}")
        for line in r.stderr.splitlines()[-15:]:
            print(f"    ! {line}")
    # G2 ---------------------------------------------------------------------
    bad = [l for l in r.stderr.splitlines() if any(p in l for p in _BAD_RE)]
    if bad:
        errs.append(f"G2 {len(bad)} fallback line(s) on stderr")
        for line in bad[:8]:
            print(f"    ! {line}")
    # G3/G4 ------------------------------------------------------------------
    out = pkg / _RESULTS_NAME
    if not out.exists():
        errs.append("G3/G4 no results file produced")
    else:
        nj, nd = _load(out)
        aj, ad = _load(anchor)
        common = sorted(set(nd) & set(ad))
        nfeas = sum(1 for i in common if nd[i]["is_feasible"])
        if nfeas != len(common):
            errs.append(f"G3 feasible {nfeas}/{len(common)}")
        cost_bad = [i for i in common if nd[i]["cost"] != ad[i]["cost"]]
        pos_bad = [i for i in common if nd[i]["positions"] != ad[i]["positions"]]
        if cost_bad:
            errs.append(f"G4 cost differs on {len(cost_bad)} case(s): {cost_bad[:8]}")
        if pos_bad:
            errs.append(f"G4 positions differ on {len(pos_bad)} case(s): {pos_bad[:8]}")
        if test_id is None and nj["total_score"] != aj["total_score"]:
            errs.append(f"G4 total {nj['total_score']!r} != {aj['total_score']!r}")
        print(f"  total={nj['total_score']!r} feasible={nfeas}/{len(common)} "
              f"cost-equal={len(common) - len(cost_bad)}/{len(common)} "
              f"positions-equal={len(common) - len(pos_bad)}/{len(common)}")
    # G5 ---------------------------------------------------------------------
    if stats.exists():
        peak, qcores = stats.read_text().split()
        print(f"  route A: peak={peak} queue={qcores}")
        if int(peak) > int(qcores):
            errs.append(f"G5 route A peak {peak} > queue {qcores}")
    else:
        errs.append("G5 route A never ran (no stats file) -- the cores gate did "
                    "not fire, so this gate proved nothing")
    # G6 ---------------------------------------------------------------------
    compiled = (pkg / "constructive.exe").exists() or (pkg / "constructive").exists()
    bundled = (pkg / "bin" / "constructive_linux").exists()
    if not (compiled or bundled):
        errs.append("G6 no usable C++ binary in the package")
    print(f"  binary: compiled={compiled} bundled-present={bundled}")

    if not keep:
        import shutil as _sh
        _sh.rmtree(_GATE, ignore_errors=True)
    for e in errs:
        print(f"  GATE FAIL: {e}")
    return not errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=48,
                    help="value forced into ICCAD_ADAPTIVE_CORES (default 48)")
    # L137 RE-ANCHOR (2026-08-16): the tree is now L136 (L131 abutment snap +
    # L136 FRAME_EPS), which is what was uploaded on 08-16, so the pre-L136
    # anchors FAIL BY DESIGN on every case the fixes improved. Previous anchors,
    # kept because they still describe real shipped artefacts:
    #   results_M80_48c_anchor.json
    #   results_L114_48c_lp_anchor.json   1.2367916697725434  (the 08-15 upload)
    ap.add_argument("--anchor", default="results_L136_48c_anchor.json")
    ap.add_argument("--test-id", type=int, default=None,
                    help="single case (fast smoke); skips the total check")
    ap.add_argument("--keep", action="store_true", help="keep the extracted tree")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="re-admit one ICCAD_* knob past the ambient strip (L137)")
    a = ap.parse_args()
    ok = run(a.cores, _REPO / a.anchor, a.test_id, a.keep, a.env)
    print(f"\nL113 SHIP GATE: {'ALL PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
