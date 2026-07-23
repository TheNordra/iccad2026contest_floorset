"""OFFLINE (never shipped): M67-C Linux verification bundle builder.

Builds a self-contained WSL2 bundle (this box has no WSL; the GPU box's WSL2
Ubuntu-22.04 g++11 is the build+verify host, see memory
docker-linux-coldstart-verify) that:

  T1  builds the static Linux binaries (bin_out/constructive_linux,
      bin_out/sa_linux, + a full-static spare) with md5 + 1-block smoke
  T2  runs m48_coldstart_dryrun.py opwrapper (submission-layout compile
      fallback, 4 phases)
  T3  extracts the staged cadc1075.tar.gz, injects the T1 binaries, runs the
      OFFICIAL command `python iccad2026_evaluate.py --evaluate op_wrapper.py`
      and bit-compares all 100 cases vs results_shipped_m51.json with a
      <2e-9 ULP warn band (expected: case 84 only) -- plus the hard
      bundled-binary-first proof: no constructive.exe compile artifact
  T4  corrupts bin/constructive_linux and re-runs ONE case: the M67-A
      fallthrough must on-site-compile and stay digit-equal

verify_final_tar.sh is included from the start so round 2 (re-verifying the
FINAL tar that ships to Google Drive, binaries inside) needs no new bundle.

Output: C:/Users/Nordra/Downloads/m67c-linux-verify.tar.gz  (~510 MB; the
dataset dominates and is incompressible -> gzip level 1). The embedded
cadc1075.tar.gz is freshly staged via make_submission.stage() (no bin/ yet;
the "binaries missing" warning is expected at this point of M67-C).
"""
import hashlib
import io
import sys
import tarfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_OUT = Path(r"C:\Users\Nordra\Downloads\m67c-linux-verify.tar.gz")
_TOP = "m67c"

_SOURCES = (
    "constructive.cpp", "optimizer_claude.cpp",
    "optimizer_constructive.py", "optimizer_claude.py",
    "make_submission.py", "m48_coldstart_dryrun.py",
    "results_shipped_m51.json",
)

# ── embedded WSL2 scripts (written with LF endings) ──────────────────────────

_SETUP_ENV_SH = r"""#!/usr/bin/env bash
# M67-C WSL2 environment setup (idempotent). Run once from the m67c/ dir.
set -e
echo "== M67-C setup_env =="
command -v g++ >/dev/null || { sudo apt-get update && sudo apt-get install -y g++; }
VENV="$HOME/m67c_venv"
if [ ! -x "$VENV/bin/python" ]; then
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    rm -rf "$VENV"
    sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
    python3 -m venv "$VENV"
  fi
fi
"$VENV/bin/pip" install --quiet --upgrade pip
if ! "$VENV/bin/python" -c "import torch, numpy, shapely, tqdm, requests, matplotlib" 2>/dev/null; then
  "$VENV/bin/pip" install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/pip" install numpy==2.2.6 shapely==2.1.2 tqdm==4.67.3 requests matplotlib
fi
"$VENV/bin/python" -c "import torch, numpy, shapely, tqdm; print('deps OK: torch', torch.__version__, '| numpy', numpy.__version__, '| shapely', shapely.__version__, '| tqdm', tqdm.__version__)"
g++ --version | head -1
echo "setup_env: OK (venv=$VENV)"
"""

_RUN_ALL_SH = r"""#!/usr/bin/env bash
# M67-C WSL2 four-tier verification. Run from the extracted m67c/ dir:
#     bash run_all.sh 2>&1 | tee m67c_run.log
set -u
cd "$(dirname "$0")"
PY="$HOME/m67c_venv/bin/python"
[ -x "$PY" ] || { echo "FATAL: venv missing -- run setup_env.sh first"; exit 1; }
FAIL=0

echo "==== M67-C T0: environment ===="
uname -a
g++ --version | head -1
"$PY" --version
"$PY" -c "import torch,numpy,shapely,tqdm; print('torch',torch.__version__,'numpy',numpy.__version__,'shapely',shapely.__version__,'tqdm',tqdm.__version__)" \
  || { echo "FATAL: python deps missing -- run setup_env.sh first"; exit 1; }

echo; echo "==== M67-C T1: build static binaries ===="
rm -rf bin_out && mkdir bin_out
if g++ -O3 -std=c++17 -static-libstdc++ -static-libgcc -o bin_out/constructive_linux constructive.cpp 2> t1_cons.err; then
  echo "COMPILE_OK constructive_linux"
else
  echo "COMPILE_FAIL constructive_linux"; tail -20 t1_cons.err; FAIL=1
fi
if g++ -O3 -std=c++17 -static-libstdc++ -static-libgcc -o bin_out/sa_linux optimizer_claude.cpp 2> t1_sa.err; then
  echo "COMPILE_OK sa_linux"
else
  echo "COMPILE_FAIL sa_linux"; tail -20 t1_sa.err; FAIL=1
fi
if g++ -O3 -std=c++17 -static -o bin_out/constructive_linux_fullstatic constructive.cpp 2> t1_full.err; then
  echo "COMPILE_OK constructive_linux_fullstatic (spare, not shipped)"
else
  echo "COMPILE_SKIP constructive_linux_fullstatic (spare, non-blocking)"; tail -5 t1_full.err
fi
for b in constructive_linux sa_linux constructive_linux_fullstatic; do
  [ -f "bin_out/$b" ] || continue
  echo "-- $b:"
  file "bin_out/$b" | sed 's/^/   /'
  ldd "bin_out/$b" 2>&1 | sed 's/^/   /'
done
( cd bin_out && md5sum * > md5sums.txt && cat md5sums.txt )
"$PY" m67c_smoke.py || FAIL=1

echo; echo "==== M67-C T2: m48 coldstart (op_wrapper submission layout) ===="
"$PY" m48_coldstart_dryrun.py opwrapper || FAIL=1

echo; echo "==== M67-C T3: official 100-case bundled-first bit-exact ===="
"$PY" m67c_tier3.py t3 || FAIL=1

echo; echo "==== M67-C T4: broken bundled binary -> compile-chain fallthrough ===="
"$PY" m67c_tier3.py t4 || FAIL=1

echo; echo "======================================"
if [ "$FAIL" = "0" ]; then
  echo "M67-C WSL2 VERDICT: ALL GREEN"
else
  echo "M67-C WSL2 VERDICT: FAILURES -- see above"
fi
exit $FAIL
"""

_VERIFY_FINAL_SH = r"""#!/usr/bin/env bash
# M67-C round 2: end-to-end verify of the FINAL submission tar (binaries
# already inside). Usage:  bash verify_final_tar.sh /mnt/d/cadc1075.tar.gz
set -u
cd "$(dirname "$0")"
PY="$HOME/m67c_venv/bin/python"
[ -x "$PY" ] || { echo "FATAL: venv missing -- run setup_env.sh first"; exit 1; }
if [ -z "${1:-}" ] || [ ! -f "$1" ]; then
  echo "usage: bash verify_final_tar.sh <path-to-final-cadc1075.tar.gz>"; exit 2
fi
"$PY" m67c_tier3.py final "$1"
"""

_SMOKE_PY = r'''"""M67-C T1 smoke: each built binary must run a trivial 1-block case
end-to-end (same semantics as the M48 _binary_runs smoke)."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimizer_claude import _serialize_input, _parse_output  # noqa: E402

inp = _serialize_input(1, [1.0], None, None, None, None, None)
ok = True
for name, timeout in (("constructive_linux", 60), ("sa_linux", 90),
                      ("constructive_linux_fullstatic", 60)):
    p = Path(__file__).resolve().parent / "bin_out" / name
    spare = name.endswith("_fullstatic")
    if not p.exists():
        print(f"SMOKE SKIP {name} (not built)")
        ok &= spare
        continue
    p.chmod(p.stat().st_mode | 0o111)
    try:
        r = subprocess.run([str(p)], input=inp, capture_output=True,
                           text=True, timeout=timeout)
        good = r.returncode == 0 and len(_parse_output(r.stdout, 1)) == 1
    except Exception as e:
        print(f"SMOKE ERROR {name}: {e}")
        good = False
    print(f"SMOKE {'OK' if good else 'FAIL'} {name}")
    ok &= good or spare
sys.exit(0 if ok else 1)
'''

_TIER3_PY = r'''"""M67-C T3/T4 driver (standalone -- no make_submission dependency).

t3           extract cadc1075.tar.gz, overlay evaluator + loader closure +
             dataset symlink (mirrors make_submission.verify mechanics),
             inject the T1 binaries, run the OFFICIAL command, assert
             bundled-binary-first engaged (no on-site compile artifact),
             ULP-tolerant bit-compare vs results_shipped_m51.json
t4           fresh extract, corrupt bin/constructive_linux, run ONE case via
             ContestEvaluator: the M67-A fallthrough must on-site-compile
             (constructive.exe appears) and stay digit-equal
final <tar>  t3 flow on a given FINAL tar (binaries already inside; nothing
             injected)
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANCHOR = ROOT / "results_shipped_m51.json"
LOADERS = ("litetestLoader.py", "lite_dataset_test.py", "liteLoader.py",
           "lite_dataset.py", "prime_dataset.py", "cost.py", "utils.py",
           "visualize.py")
TEAM = "cadc1075"
BIN_NAMES = ("constructive_linux",)   # M67-G slim: sa_linux removed (unused).
                                      # The slim submission ships only the
                                      # constructive binary; the pure-Python SA
                                      # fallback needs no bundled binary.
ULP_TOL = 2e-9          # sub-2e-9 diffs are WARN (known: case 84 libm ULP)
TOTAL_TOL = 1e-9
T4_CASE = 50


def build_layout(workdir: Path, tar_path: Path) -> Path:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir()
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(workdir)
    pkg = workdir / TEAM
    assert (pkg / "op_wrapper.py").exists(), "extracted package incomplete"
    # grader-side overlay: evaluator inside the package dir; loader closure
    # and the dataset one level up (evaluator sys.path.insert(parent.parent)
    # + data_path default "../" resolved against cwd=pkg).
    shutil.copyfile(ROOT / "iccad2026contest" / "iccad2026_evaluate.py",
                    pkg / "iccad2026_evaluate.py")
    for f in LOADERS:
        shutil.copyfile(ROOT / f, workdir / f)
    os.symlink(ROOT / "LiteTensorDataTest", workdir / "LiteTensorDataTest",
               target_is_directory=True)
    return pkg


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def compare(new_path: Path) -> bool:
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old = json.loads(ANCHOR.read_text(encoding="utf-8"))
    fails, warns = [], []
    dt = abs(new["total_score"] - old["total_score"])
    print(f"   total new={new['total_score']!r}")
    print(f"   total old={old['total_score']!r}  |d|={dt:.3e}")
    if dt > TOTAL_TOL:
        fails.append(f"total_score |d|={dt:.3e} > {TOTAL_TOL:g}")
    tn, to = new["test_results"], old["test_results"]
    if len(tn) != len(to):
        fails.append(f"case count {len(tn)} != {len(to)}")
    nfeas = 0
    for a, b in zip(tn, to):
        cid = b["test_id"]
        for k in ("test_id", "block_count"):
            if a[k] != b[k]:
                fails.append(f"case {cid}: {k} {a[k]!r} != {b[k]!r}")
        nfeas += bool(a["is_feasible"])
        if not a["is_feasible"]:
            fails.append(f"case {cid}: INFEASIBLE")
        for k in ("hpwl_gap", "area_gap", "violations_relative", "cost"):
            if a[k] != b[k]:
                d = abs(a[k] - b[k])
                (warns if d < ULP_TOL else fails).append(
                    f"case {cid}: {k} |d|={d:.3e}")
        if a["positions"] != b["positions"]:
            if len(a["positions"]) != len(b["positions"]):
                fails.append(f"case {cid}: positions length differs")
            else:
                worst = max(abs(x - y)
                            for p, q in zip(a["positions"], b["positions"])
                            for x, y in zip(p, q))
                (warns if worst < ULP_TOL else fails).append(
                    f"case {cid}: positions worst |d|={worst:.3e}")
    print(f"   feasible {nfeas}/{len(tn)}  avg_runtime "
          f"{new['summary']['avg_runtime']}")
    for w in warns:
        print(f"   ULP-WARN {w}")
    for f_ in fails[:20]:
        print(f"   FAIL {f_}")
    if len(fails) > 20:
        print(f"   ... and {len(fails) - 20} more")
    if fails:
        return False
    print(f"   compare: OK ({len(tn)} cases; {len(warns)} sub-{ULP_TOL:g} "
          f"ULP warns, expected only case 84)")
    return True


def t3(tar_path: Path, inject: bool, workname: str) -> bool:
    print(f"-- layout from {tar_path.name} -> {workname}/")
    pkg = build_layout(ROOT / workname, tar_path)
    if inject:
        (pkg / "bin").mkdir(exist_ok=True)
        for b in BIN_NAMES:
            src = ROOT / "bin_out" / b
            if not src.exists():
                print(f"FAIL: bin_out/{b} missing -- T1 must pass first")
                return False
            shutil.copyfile(src, pkg / "bin" / b)
    for b in BIN_NAMES:
        p = pkg / "bin" / b
        if not p.exists():
            print(f"FAIL: package bin/{b} missing")
            return False
        p.chmod(0o755)
    res_name = f"results_m67c_{workname}.json"
    cmd = [sys.executable, "-u", "iccad2026_evaluate.py", "--evaluate",
           "op_wrapper.py", "-o", res_name]
    print(f"-- official cmd (cwd={TEAM}): {' '.join(cmd[1:])}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(pkg), env=clean_env(),
                       capture_output=True, encoding="utf-8",
                       errors="replace", timeout=3600)
    print(f"-- eval done in {time.time() - t0:.0f}s (exit {r.returncode})")
    for line in r.stdout.splitlines()[-6:]:
        print(f"   | {line}")
    if r.returncode != 0:
        print("FAIL: evaluator exited non-zero; stderr tail:")
        for line in r.stderr.splitlines()[-15:]:
            print(f"   ! {line}")
        return False
    ok = True
    bad = [l for l in r.stderr.splitlines()
           if "fallback" in l or "unavailable" in l
           or "[constructive]" in l or "[optimizer_claude]" in l]
    if bad:
        print("FAIL: fallback/compile-chain lines on stderr:")
        for line in bad[:10]:
            print(f"   ! {line}")
        ok = False
    exe = pkg / "constructive.exe"
    if exe.exists():
        print("FAIL: constructive.exe exists -> on-site compile happened "
              "(bundled-binary-first did NOT engage)")
        ok = False
    else:
        print("   bundled-first: OK (no on-site compile artifact)")
    ok &= compare(pkg / res_name)
    return ok


def t4() -> bool:
    pkg = build_layout(ROOT / "t4", ROOT / f"{TEAM}.tar.gz")
    (pkg / "bin").mkdir(exist_ok=True)
    sa = ROOT / "bin_out" / "sa_linux"
    if sa.exists():
        shutil.copyfile(sa, pkg / "bin" / "sa_linux")
    (pkg / "bin" / "constructive_linux").write_bytes(
        b"\x7fELF garbage -- deliberately not runnable (M67-C T4)")
    for p in (pkg / "bin").iterdir():
        p.chmod(0o755)
    sys.path.insert(0, str(ROOT / "iccad2026contest"))
    sys.path.insert(0, str(ROOT))
    for k in list(os.environ):
        if k.startswith("ICCAD_"):
            del os.environ[k]
    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator(data_path=str(ROOT), verbose=False)
    t0 = time.time()
    res = ev.evaluate(str(pkg / "op_wrapper.py"), test_ids=[T4_CASE])
    tr = res.test_results[0]
    anchor = {c["test_id"]: c
              for c in json.loads(ANCHOR.read_text(encoding="utf-8"))
              ["test_results"]}[T4_CASE]
    exe = pkg / "constructive.exe"
    d = abs(tr.cost - anchor["cost"])
    cost_ok = tr.cost == anchor["cost"] or d < ULP_TOL
    ok = (tr.error is None and tr.is_feasible and cost_ok and exe.exists())
    print(f"   case {T4_CASE}: cost new={tr.cost!r} anchor={anchor['cost']!r} "
          f"|d|={d:.3e} ({time.time() - t0:.0f}s incl. on-site compile)")
    print(f"   on-site compile artifact present: {exe.exists()} "
          f"(garbage bundled binary must fall through to the compile chain)")
    if tr.error is not None:
        print(f"   FAIL error: {tr.error}")
    return ok


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "t3":
        ok = t3(ROOT / f"{TEAM}.tar.gz", inject=True, workname="t3")
    elif mode == "t4":
        ok = t4()
    elif mode == "final" and len(sys.argv) > 2:
        ok = t3(Path(sys.argv[2]).resolve(), inject=False, workname="t_final")
    else:
        print(__doc__)
        sys.exit(2)
    print(f"M67C-TIER [{mode}]: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
'''

_SCRIPTS = {
    "setup_env.sh": _SETUP_ENV_SH,
    "run_all.sh": _RUN_ALL_SH,
    "verify_final_tar.sh": _VERIFY_FINAL_SH,
    "m67c_smoke.py": _SMOKE_PY,
    "m67c_tier3.py": _TIER3_PY,
}


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes, mode: int):
    ti = tarfile.TarInfo(arcname)
    ti.size = len(data)
    ti.mode = mode
    ti.mtime = int(time.time())
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    tf.addfile(ti, io.BytesIO(data))


def _tar_filter(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode = 0o755 if ti.isdir() else 0o644
    return ti


def main():
    print("== M67-C bundle build ==")
    import make_submission

    # 1. fresh cadc1075.tar.gz via the M67-B packager (no bin/ yet -> warning
    #    expected; hygiene must still pass)
    if not make_submission.stage():
        print("FATAL: make_submission.stage() failed")
        sys.exit(1)
    sub_tar = make_submission._TAR
    with tarfile.open(sub_tar, "r:gz") as tf:
        members = {m.name for m in tf.getmembers() if m.isfile()}
        opw = tf.extractfile(f"{make_submission._TEAM}/op_wrapper.py").read()
    want = {f"{make_submission._TEAM}/{f}" for f in make_submission._WHITELIST}
    if members != want:
        print(f"FATAL: embedded submission tar members unexpected:\n"
              f"  extra={sorted(members - want)}\n  missing={sorted(want - members)}")
        sys.exit(1)
    print(f"  embedded {sub_tar.name}: {len(members)} members (no bin/, OK for "
          f"this stage)  md5={_md5(sub_tar.read_bytes())}")
    print(f"  op_wrapper.py inside: md5={_md5(opw)} ({len(opw)} bytes)")

    # 2. sanity: every bundled repo file exists
    missing = [f for f in _SOURCES + make_submission._LOADER_FILES
               if not (_REPO / f).exists()]
    if missing or not (_REPO / "iccad2026contest" / "iccad2026_evaluate.py").exists() \
            or not (_REPO / "LiteTensorDataTest").is_dir():
        print(f"FATAL: missing repo files: {missing}")
        sys.exit(1)

    # 3. build the bundle tar (dataset is incompressible -> gzip level 1)
    t0 = time.time()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    if _OUT.exists():
        _OUT.unlink()
    with tarfile.open(_OUT, "w:gz", compresslevel=1) as tf:
        for name, text in _SCRIPTS.items():
            if "\r" in text:
                print(f"FATAL: CR in embedded script {name}")
                sys.exit(1)
            mode = 0o755 if name.endswith(".sh") else 0o644
            _add_bytes(tf, f"{_TOP}/{name}", text.encode("utf-8"), mode)
        for f in _SOURCES + make_submission._LOADER_FILES:
            tf.add(_REPO / f, arcname=f"{_TOP}/{f}", filter=_tar_filter)
        tf.add(_REPO / "iccad2026contest" / "iccad2026_evaluate.py",
               arcname=f"{_TOP}/iccad2026contest/iccad2026_evaluate.py",
               filter=_tar_filter)
        tf.add(sub_tar, arcname=f"{_TOP}/{sub_tar.name}", filter=_tar_filter)
        tf.add(_REPO / "LiteTensorDataTest",
               arcname=f"{_TOP}/LiteTensorDataTest", filter=_tar_filter)
    dt = time.time() - t0

    # 4. self-check: reopen, verify key members + LF-only .sh
    with tarfile.open(_OUT, "r:gz") as tf:
        names = set(tf.getnames())
        for name in _SCRIPTS:
            if f"{_TOP}/{name}" not in names:
                print(f"FATAL: bundle missing {name}")
                sys.exit(1)
            if name.endswith(".sh"):
                if b"\r" in tf.extractfile(f"{_TOP}/{name}").read():
                    print(f"FATAL: CRLF leaked into {name}")
                    sys.exit(1)
        n_files = sum(1 for m in tf.getmembers() if m.isfile())
    size_mb = _OUT.stat().st_size / 1e6
    print(f"  bundle: {_OUT}")
    print(f"  {size_mb:.0f} MB, {n_files} files, built in {dt:.0f}s")
    print(f"  bundle md5: {_md5(_OUT.read_bytes())}")
    print("M67-C BUNDLE: OK")


if __name__ == "__main__":
    main()
