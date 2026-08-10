"""OFFLINE (never shipped): M48 Beta cold-start dry-run for the submission package.

Simulates the official grader loading our optimizer in a CLEAN directory holding
only the three shipped files (optimizer_constructive.py, constructive.cpp,
optimizer_claude.py) with NO prebuilt .exe: `_load_optimizer` -> `MyOptimizer.
__init__` -> `_ensure_compiled` must compile OUTSIDE the scored window
(iccad2026_evaluate.py:845 vs :884) and reproduce the repo's per-case costs
digit-for-digit. Each phase runs in its OWN subprocess (a shared process would
reuse the repo's optimizer_claude from sys.modules and hide packaging bugs).

  phase 0  baseline: repo optimizer on 3 cases (smallest/median/largest n)
  phase 1  cold start: clean dir, no exe -> compile happens, costs digit-equal,
           exe created, stderr free of any fallback
  phase 2  foreign binary: exe overwritten with garbage + future mtime -> the
           M48 smoke test must reject it and recompile (costs digit-equal)
  phase 3  compile chain: ICCAD_CXX=<missing compiler> forced to the front ->
           the chain must skip it and still succeed via the next candidate
  phase 4  path hygiene: report absolute C:\\ paths in the shipped files

M67-B variant (`python m48_coldstart_dryrun.py opwrapper`): same four phases,
but the clean dir holds the Beta SUBMISSION layout instead -- op_wrapper.py
(generated live via make_submission.build_op_wrapper_text(), so the gate and
the packager share one merge implementation) + constructive.cpp +
optimizer_claude.cpp, with NO optimizer_claude.py. Phase 0 baseline stays the
repo optimizer, so digit-equality also proves the merged single-file form is
cost-identical to the shipped three-file form.

Never touches the repo constructive.exe (phases compile only inside a tempdir).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).parent
_PY = sys.executable
_SHIP = ("optimizer_constructive.py", "constructive.cpp", "optimizer_claude.py")


def _run_phase(args, env_over=None, timeout=900):
    env = dict(os.environ)
    env.update(env_over or {})
    # Explicit utf-8/replace: text=True decodes with the LOCALE codec, and
    # compiler/OS messages are not valid cp950 on a zh-TW box -- the reader
    # thread then dies and r.stderr comes back None, which phase 3 reads.
    r = subprocess.run([_PY, str(Path(__file__).resolve()), *args],
                       capture_output=True, text=True, timeout=timeout,
                       env=env, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"phase {args} exited {r.returncode}")
    costs = dict(re.findall(r"^PHASECOST (\d+) (\S+)$", r.stdout, re.M))
    return costs, r.stdout, r.stderr


def _phase_eval(opt_path: str, ids):
    """Child-process body: evaluate `opt_path` on `ids`, print PHASECOST lines."""
    sys.path.insert(0, str(_REPO / "iccad2026contest"))
    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator(data_path=str(_REPO), verbose=False)
    res = ev.evaluate(opt_path, test_ids=list(ids))
    for tr in res.test_results:
        assert tr.error is None, f"case {tr.test_id} errored: {tr.error}"
        assert tr.is_feasible, f"case {tr.test_id} infeasible"
        print(f"PHASECOST {tr.test_id} {tr.cost!r}")


def _pick_ids():
    sys.path.insert(0, str(_REPO / "iccad2026contest"))
    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator(data_path=str(_REPO), verbose=False)
    ev._load_dataset()
    ns = [int((ev.dataset[i]["input"][0] != -1).sum().item()) for i in range(100)]
    order = sorted(range(100), key=lambda i: ns[i])
    ids = (order[0], order[50], order[99])
    print(f"cases: " + ", ".join(f"idx={i} n={ns[i]}" for i in ids))
    return ids


def main():
    if "--eval" in sys.argv:                      # child-process entry
        k = sys.argv.index("--eval")
        _phase_eval(sys.argv[k + 1], [int(x) for x in sys.argv[k + 2:]])
        return

    opw = "opwrapper" in sys.argv[1:]             # M67-B submission-layout variant
    tag = " (op_wrapper)" if opw else ""
    ids = _pick_ids()
    sids = [str(i) for i in ids]
    scratch = Path(tempfile.mkdtemp(prefix="m48_coldstart_"))
    print(f"scratch: {scratch}{tag}")
    ok = True
    try:
        # ---- phase 0: repo baseline --------------------------------------
        t0 = time.time()
        base, _, err0 = _run_phase(["--eval", str(_REPO / "optimizer_constructive.py"), *sids])
        print(f"phase 0 baseline costs: {base}  ({time.time()-t0:.0f}s)")

        # ---- phase 1: cold start in a clean dir --------------------------
        if opw:
            import make_submission
            (scratch / "op_wrapper.py").write_text(
                make_submission.build_op_wrapper_text(), encoding="utf-8",
                newline="\n")
            for f in ("constructive.cpp", "optimizer_claude.cpp"):
                shutil.copy2(_REPO / f, scratch / f)
            target = scratch / "op_wrapper.py"
            scan_files = ("op_wrapper.py", "constructive.cpp", "optimizer_claude.cpp")
        else:
            for f in _SHIP:
                shutil.copy2(_REPO / f, scratch / f)
            target = scratch / "optimizer_constructive.py"
            scan_files = _SHIP
        exe = scratch / "constructive.exe"
        assert not exe.exists()
        t0 = time.time()
        c1, _, err1 = _run_phase(["--eval", str(target), *sids])
        p1 = (c1 == base and exe.exists()
              and "fallback" not in err1 and "unavailable" not in err1)
        print(f"phase 1 cold start: {'PASS' if p1 else 'FAIL'} "
              f"(exe={exe.exists()}, costs_equal={c1 == base}, {time.time()-t0:.0f}s)")
        ok &= p1

        # ---- phase 2: foreign/corrupt binary must be rejected ------------
        exe.write_bytes(b"MZ this is not a runnable binary")
        future = time.time() + 3600
        os.utime(exe, (future, future))
        garbage_size = exe.stat().st_size
        t0 = time.time()
        c2, _, err2 = _run_phase(["--eval", str(target), sids[0]])
        p2 = (c2.get(sids[0]) == base[sids[0]]
              and exe.stat().st_size != garbage_size and "fallback" not in err2)
        print(f"phase 2 foreign binary: {'PASS' if p2 else 'FAIL'} "
              f"(recompiled {garbage_size}B -> {exe.stat().st_size}B, {time.time()-t0:.0f}s)")
        ok &= p2

        # ---- phase 3: compile chain skips a missing compiler -------------
        exe.unlink()
        t0 = time.time()
        c3, _, err3 = _run_phase(["--eval", str(target), sids[0]],
                                 env_over={"ICCAD_CXX": "no-such-compiler-m48"})
        p3 = (c3.get(sids[0]) == base[sids[0]]
              and "no-such-compiler-m48" in err3 and "fallback" not in err3)
        print(f"phase 3 compile chain: {'PASS' if p3 else 'FAIL'} "
              f"(skipped bogus compiler, {time.time()-t0:.0f}s)")
        ok &= p3

        # ---- phase 4: absolute-path hygiene (report only) -----------------
        print("phase 4 absolute C:\\ paths in shipped files:")
        scan_root = scratch if opw else _REPO
        for f in scan_files:
            for ln, line in enumerate((scan_root / f).read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"[Cc]:[\\/]", line):
                    print(f"  {f}:{ln}: {line.strip()[:90]}")
        print("  (expected: only the try-guarded compiler candidates in both wrappers)"
              if not opw else
              "  (expected: only the nt-gated msys compiler candidate in op_wrapper.py)")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nM48 COLD-START DRY-RUN{tag}: {'ALL PASS' if ok else 'FAILURES -- see above'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
