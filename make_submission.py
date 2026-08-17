"""OFFLINE (never shipped): M67-B Beta submission packager + official-layout gate.

Produces the Problem C Beta package (naming is zero-tolerance -- a wrong file
name is a DQ and re-submission is not accepted):

    build_submission/cadc1075/
        op_wrapper.py         entry point (constructive portfolio wrapper with
                              the optimizer_claude.py SA fallback mechanically
                              merged at the tail -- no optimizer_claude.py file
                              may exist in the package)
        op_src.py             byte-identical copy of op_wrapper.py (backup)
        requirements.txt      0 bytes (official env already has torch/shapely)
        README.md             short compile-fallback explanation
        constructive.cpp      main placer source (on-site compile fallback)
        bin/constructive_linux   prebuilt Linux binary (M67-C output;
                              packaged only if present)
    build_submission/cadc1075.tar.gz

Merge mechanics (M67-B, see M67_PLAN.md): a NAIVE concatenation of
optimizer_claude.py would be fatal -- its module-level _BIN, _ensure_compiled
and MyOptimizer would shadow the constructive ones (every profile would run the
SA binary). Only two blocks are sliced out of optimizer_claude.py:
  B1  "# -- Fallback Python SA" .. before the GNN section   (python_sa_solve
      + Skyline/skyline_decode/violation_cost + COL_*/BOUNDARY_* constants)
  B2  "def _serialize_input("   .. before "# -- Optimizer class"
      (_serialize_input + _parse_output)
and the lazy `from optimizer_claude import ...` try/except in
optimizer_constructive.py is REMOVED (not just allowed to fail): with it kept,
any repo-layout gate run would silently import the repo's optimizer_claude via
sys.path and mask merge bugs.

Modes:
    python make_submission.py stage    build + stage + hygiene + tar
    python make_submission.py verify   extract tar into build_submission/verify/,
                                       overlay the evaluator + loader closure +
                                       a LiteTensorDataTest junction, run the
                                       OFFICIAL command (cwd=verify/cadc1075):
                                           python iccad2026_evaluate.py --evaluate op_wrapper.py
                                       and bit-compare all 100 cases against
                                       results_M74_default.json (1.293461035226291)
    python make_submission.py all      both (default)

Importable without side effects; m48_coldstart_dryrun.py's `opwrapper` variant
imports build_op_wrapper_text() so the coldstart gate and the package share one
merge implementation.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_TEAM = "cadc1075"
_BUILD = _REPO / "build_submission"
_STAGE = _BUILD / _TEAM
_TAR = _BUILD / f"{_TEAM}.tar.gz"
_VERIFY = _BUILD / "verify"
# M76 ship chain: the anchor follows what the tree actually produces. It was
# results_shipped_m71.json (1.305389893450635 = the uploaded Beta package); the
# tree has been M74 since 2026-07-30 and M74 is what this package now carries.
# L137 RE-ANCHOR (2026-08-16): the tree is now L136 -- the L131 abutment snap
# (optimizer_constructive._snap_group_abutment) plus the L136 FRAME_EPS fix in
# constructive.cpp -- and that is what was uploaded on 08-16. Against the M74
# anchor this gate FAILED BY DESIGN on every case the fixes improved, which
# makes it useless as a regression gate; re-anchored so FAIL means regression.
#   was: results_M74_default.json   1.293461035226291
_ANCHOR = _REPO / "results_L136_default.json"       # 1.2772224039603648
_RESULTS_NAME = "results_m67b_opwrapper.json"

_CONDA_PY = Path(r"C:\Users\Nordra\.conda\envs\iccadv\python.exe")
_PY = str(_CONDA_PY if _CONDA_PY.exists() else Path(sys.executable))

# ── merge markers ────────────────────────────────────────────────────────────
# The exact 4-line lazy import block M67-A left in optimizer_constructive.py.
_IMPORT_BLOCK = (
    "try:\n"
    "    from optimizer_claude import _serialize_input, _parse_output, python_sa_solve\n"
    "except ImportError:\n"
    "    pass  # merged single-file form (M67-B): these are defined at the file tail\n"
)
_IMPORT_REPL = (
    "# M67-B merged form: the SA fallback (_serialize_input / _parse_output /\n"
    "# python_sa_solve) is defined at the tail of this file; no optimizer_claude\n"
    "# module exists in the submission package.\n"
)
_B1_START = "# ── Fallback Python SA"
_B1_END = "# ── GNN-based initial-perm hint"
_B2_START = "def _serialize_input("
_B2_END = "# ── Optimizer class"

_MERGE_HEADER = '''

# ═════════════════════════════════════════════════════════════════════════════
# M67-B: SA fallback, mechanically sliced out of optimizer_claude.py -- the pure-
# Python SA (python_sa_solve + helpers) and the C++ I/O helpers
# (_serialize_input/_parse_output) that the portfolio wrapper above calls.
# The SA MyOptimizer class, the GNN blocks and the SA compile chain are
# intentionally NOT merged: the wrapper never calls them, and their module-
# level _BIN/_ensure_compiled/MyOptimizer would shadow the constructive ones.
#
# NOTE: no `import time` here. optimizer_constructive.py imports it already
# (route A's queue times its frame tasks), and a second module-level binding
# trips the collision assert below. That assert is what stops the SA slice
# from shadowing _BIN/_ensure_compiled/MyOptimizer, so it stays at full
# strength -- the redundant import goes instead. The `time` pin in
# build_op_wrapper_text() below keeps this from rotting silently.
# ═════════════════════════════════════════════════════════════════════════════
from io import StringIO

from iccad2026_evaluate import (calculate_hpwl_b2b, calculate_hpwl_p2b,
                                calculate_bbox_area)

'''

# ── package spec ─────────────────────────────────────────────────────────────
_WHITELIST = ("op_wrapper.py", "op_src.py", "requirements.txt", "README.md",
              "constructive.cpp")
_BIN_FILES = ("bin/constructive_linux",)
_TEXT_EXT = {".py", ".md", ".txt", ".cpp"}
_FORBIDDEN_EXT = {".pkl", ".json", ".log", ".exe", ".pyc"}
# Absolute-path scan: the m48 phase-4 pattern family (a broader [A-Za-z]:
# drive class false-positives on ":\n" escape sequences in f-strings).
_ABS_RE = re.compile(r"[Cc]:[\\/]|/home/|/Users/")
# The single allowed absolute path: the nt-gated msys compiler candidate
# (M67-A; dead code on the POSIX grader, load-bearing for Windows dev).
_ABS_ALLOW = r"C:\msys64\ucrt64\bin\g++.exe"

_README = """# ICCAD 2026 Problem C -- team cadc1075

Entry point: `op_wrapper.py` (class `MyOptimizer`), evaluated as
`python iccad2026_evaluate.py --evaluate op_wrapper.py`.
`op_src.py` is a byte-identical backup copy of the same source.

The solver is a deterministic C++ constructive placer (`constructive.cpp`)
driven by a Python portfolio wrapper. Binary resolution happens once at
optimizer load time (outside the scored per-case window):

1. bundled prebuilt Linux binary `bin/constructive_linux` -- chmod +x, then a
   1-block smoke test; used only if the smoke passes;
2. on-site compile fallback: g++ / clang++ / c++  x  -O3 / -O2
   (`g++ -O3 -std=c++17 -o constructive.exe constructive.cpp`), each candidate
   accepted only after the same 1-block smoke test;
3. pure-Python SA fallback (embedded in op_wrapper.py) if no binary runs.

`requirements.txt` is intentionally empty: only torch / shapely / numpy from
the official environment are used.
"""

# Loader closure the evaluator needs when it sits inside cadc1075/ (its
# sys.path.insert(parent.parent) points at verify/): evaluator ->
# litetestLoader(->lite_dataset_test, visualize), liteLoader(->lite_dataset),
# cost(->utils), utils(->prime_dataset, visualize).
_LOADER_FILES = ("litetestLoader.py", "lite_dataset_test.py", "liteLoader.py",
                 "lite_dataset.py", "prime_dataset.py", "cost.py", "utils.py",
                 "visualize.py")


# ── op_wrapper build ─────────────────────────────────────────────────────────
def _slice(text: str, start: str, end: str, name: str) -> str:
    if text.count(start) != 1:
        raise AssertionError(f"{name}: start marker not unique ({text.count(start)}x): {start!r}")
    if text.count(end) != 1:
        raise AssertionError(f"{name}: end marker not unique ({text.count(end)}x): {end!r}")
    i, j = text.index(start), text.index(end)
    if not i < j:
        raise AssertionError(f"{name}: marker order wrong")
    return text[i:j].rstrip() + "\n"


def _toplevel_names(text: str, name: str) -> set:
    """Module-level bound names (defs/classes/assigns/imports) for the
    collision assert between the constructive part and the merged tail."""
    names = set()
    for node in ast.parse(text, filename=name).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name == "*":
                    continue
                names.add(a.asname or a.name.split(".")[0])
    return names


def build_op_wrapper_text() -> str:
    cons = (_REPO / "optimizer_constructive.py").read_text(encoding="utf-8")
    if cons.count(_IMPORT_BLOCK) != 1:
        raise AssertionError("optimizer_constructive.py: the M67-A lazy "
                             "optimizer_claude import block was not found verbatim")
    cons = cons.replace(_IMPORT_BLOCK, _IMPORT_REPL)

    cla = (_REPO / "optimizer_claude.py").read_text(encoding="utf-8")
    b1 = _slice(cla, _B1_START, _B1_END, "B1")
    b2 = _slice(cla, _B2_START, _B2_END, "B2")
    tail = _MERGE_HEADER + b1 + "\n\n" + b2
    text = cons.rstrip() + "\n" + tail

    # merge asserts -----------------------------------------------------------
    compile(text, "op_wrapper.py", "exec")           # syntax
    for pat, want in (("class MyOptimizer", 1), ("def python_sa_solve(", 1),
                      ("def _serialize_input(", 1), ("def _parse_output(", 1),
                      ("def _ensure_compiled(", 1), ("def _binary_runs(", 1)):
        got = text.count(pat)
        if got != want:
            raise AssertionError(f"merged op_wrapper: {pat!r} count {got} != {want}")
    for pat in ("from optimizer_claude", "import optimizer_claude"):
        if pat in text:
            raise AssertionError(f"merged op_wrapper: leftover {pat!r}")
    for pat in ("_BIN =", "def _ensure_compiled(", "class MyOptimizer", "_GNN"):
        if pat in tail:
            raise AssertionError(f"merged tail: forbidden {pat!r} (would shadow "
                                 f"the constructive module state)")
    cons_names = _toplevel_names(cons, "cons")
    # The tail's SA code calls time.perf_counter but no longer imports time
    # itself (see _MERGE_HEADER); pin the dependency it now inherits from cons.
    for need in ("time",):
        if need not in cons_names:
            raise AssertionError(f"merged op_wrapper: tail needs {need!r} but "
                                 f"optimizer_constructive.py no longer imports it")
    clash = cons_names & _toplevel_names(tail, "tail")
    if clash:
        raise AssertionError(f"merged op_wrapper: module-level name collisions: {sorted(clash)}")
    return text


# ── staging + hygiene + tar ──────────────────────────────────────────────────
def _hygiene(stage: Path, expected: set) -> list:
    errs = []
    found = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    if found != expected:
        errs.append(f"file set mismatch: extra={sorted(found - expected)} "
                    f"missing={sorted(expected - found)}")
    pys = sorted(f for f in found if f.endswith(".py"))
    if pys != ["op_src.py", "op_wrapper.py"]:
        errs.append(f".py files must be exactly op_wrapper.py+op_src.py, got {pys}")
    for f in found:
        if Path(f).suffix.lower() in _FORBIDDEN_EXT:
            errs.append(f"forbidden file type in package: {f}")
    req = stage / "requirements.txt"
    if req.stat().st_size != 0:
        errs.append(f"requirements.txt must be 0 bytes, is {req.stat().st_size}")
    if (stage / "op_src.py").read_bytes() != (stage / "op_wrapper.py").read_bytes():
        errs.append("op_src.py is not byte-identical to op_wrapper.py")
    for f in sorted(found):
        if Path(f).suffix not in _TEXT_EXT:
            continue
        for ln, line in enumerate((stage / f).read_text(encoding="utf-8").splitlines(), 1):
            if _ABS_RE.search(line) and _ABS_ALLOW not in line:
                errs.append(f"absolute path in {f}:{ln}: {line.strip()[:90]}")
    return errs


def _tar_filter(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    if ti.isdir():
        ti.mode = 0o755
    elif ti.name.startswith(f"{_TEAM}/bin/"):
        ti.mode = 0o755                 # exec bit for the bundled binaries
    else:
        ti.mode = 0o644
    return ti


def stage() -> bool:
    print("== M67-B stage ==")
    if _STAGE.exists():
        shutil.rmtree(_STAGE)
    _STAGE.mkdir(parents=True)

    text = build_op_wrapper_text()
    (_STAGE / "op_wrapper.py").write_text(text, encoding="utf-8", newline="\n")
    shutil.copyfile(_STAGE / "op_wrapper.py", _STAGE / "op_src.py")
    (_STAGE / "requirements.txt").write_bytes(b"")
    (_STAGE / "README.md").write_text(_README, encoding="utf-8", newline="\n")
    shutil.copyfile(_REPO / "constructive.cpp", _STAGE / "constructive.cpp")

    expected = set(_WHITELIST)
    nbin = 0
    for rel in _BIN_FILES:
        src = _REPO / rel
        if src.exists():
            (_STAGE / "bin").mkdir(exist_ok=True)
            shutil.copyfile(src, _STAGE / rel)
            expected.add(rel)
            nbin += 1
    if nbin < len(_BIN_FILES):
        print(f"  WARNING: {len(_BIN_FILES) - nbin} bundled Linux binaries missing "
              f"(M67-C pending) -- packaging without them")

    errs = _hygiene(_STAGE, expected)
    for e in errs:
        print(f"  HYGIENE FAIL: {e}")
    if errs:
        return False
    print(f"  hygiene: OK ({len(expected)} files)")

    if _TAR.exists():
        _TAR.unlink()
    with tarfile.open(_TAR, "w:gz") as tf:
        tf.add(_STAGE, arcname=_TEAM, filter=_tar_filter)
    with tarfile.open(_TAR, "r:gz") as tf:
        members = {m.name for m in tf.getmembers() if m.isfile()}
    want = {f"{_TEAM}/{f}" for f in expected}
    if members != want:
        print(f"  TAR FAIL: member mismatch: extra={sorted(members - want)} "
              f"missing={sorted(want - members)}")
        return False
    print(f"  tar: {_TAR.name} OK ({_TAR.stat().st_size} bytes, "
          f"{len(members)} files under {_TEAM}/)")
    return True


# ── verify: official layout + official command + bit-exact compare ───────────
def _link_dataset(src: Path, dst: Path) -> str:
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(str(src), str(dst))
            return "junction"
        except Exception:
            pass
        try:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                           check=True, capture_output=True)
            return "mklink"
        except Exception:
            pass
    else:
        try:
            os.symlink(src, dst, target_is_directory=True)
            return "symlink"
        except Exception:
            pass
    shutil.copytree(src, dst)
    return "copy"


def _compare(new_path: Path, anchor_path: Path) -> bool:
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old = json.loads(anchor_path.read_text(encoding="utf-8"))
    diffs = []
    if new["total_score"] != old["total_score"]:
        diffs.append(f"total_score {new['total_score']!r} != {old['total_score']!r}")
    tn, to = new["test_results"], old["test_results"]
    if len(tn) != len(to):
        diffs.append(f"case count {len(tn)} != {len(to)}")
    for a, b in zip(tn, to):
        cid = b["test_id"]
        for k in ("test_id", "block_count", "is_feasible", "hpwl_gap",
                  "area_gap", "violations_relative", "cost"):
            if a[k] != b[k]:
                diffs.append(f"case {cid}: {k} {a[k]!r} != {b[k]!r}")
        if a["positions"] != b["positions"]:
            n_bad = sum(1 for p, q in zip(a["positions"], b["positions"]) if p != q)
            diffs.append(f"case {cid}: positions differ ({n_bad} blocks)")
        if not a["is_feasible"]:
            diffs.append(f"case {cid}: infeasible")
    nfeas = sum(1 for a in tn if a["is_feasible"])
    print(f"  new total_score={new['total_score']!r} feasible={nfeas}/{len(tn)} "
          f"avg_runtime={new['summary']['avg_runtime']}")
    print(f"  anchor total_score={old['total_score']!r}")
    if diffs:
        print(f"  BIT-EXACT FAIL: {len(diffs)} differences")
        for d in diffs[:20]:
            print(f"    {d}")
        if len(diffs) > 20:
            print(f"    ... and {len(diffs) - 20} more")
        return False
    print(f"  bit-exact: OK ({len(tn)} cases, total + costs + gaps + positions)")
    return True


def verify() -> bool:
    print("== M67-B verify (official layout, official command) ==")
    if not _TAR.exists():
        print(f"  FAIL: {_TAR} missing -- run stage first")
        return False
    if _VERIFY.exists():
        shutil.rmtree(_VERIFY)
    _VERIFY.mkdir(parents=True)

    with tarfile.open(_TAR, "r:gz") as tf:
        tf.extractall(_VERIFY)
    pkg = _VERIFY / _TEAM
    assert (pkg / "op_wrapper.py").exists(), "extracted package incomplete"

    # grader-side overlay: evaluator inside the package dir; loader closure and
    # the dataset one level up (evaluator sys.path.insert(parent.parent) +
    # data_path default "../" resolved against cwd=pkg).
    shutil.copyfile(_REPO / "iccad2026contest" / "iccad2026_evaluate.py",
                    pkg / "iccad2026_evaluate.py")
    for f in _LOADER_FILES:
        shutil.copyfile(_REPO / f, _VERIFY / f)
    how = _link_dataset(_REPO / "LiteTensorDataTest", _VERIFY / "LiteTensorDataTest")
    if not any((_VERIFY / "LiteTensorDataTest").iterdir()):
        print("  FAIL: dataset link empty")
        return False
    print(f"  dataset: {how}; loaders: {len(_LOADER_FILES)}; evaluator overlaid")

    env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    env["PYTHONIOENCODING"] = "utf-8"   # tqdm bars are UTF-8; cp1252 decode dies
    cmd = [_PY, "-u", "iccad2026_evaluate.py", "--evaluate", "op_wrapper.py",
           "-o", _RESULTS_NAME]
    print(f"  running (cwd={pkg.name}): {' '.join(cmd[1:])}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(pkg), env=env, capture_output=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    print(f"  eval done in {time.time() - t0:.0f}s (exit {r.returncode})")
    for line in r.stdout.splitlines()[-8:]:
        print(f"    | {line}")
    if r.returncode != 0:
        print("  FAIL: evaluator exited non-zero; stderr tail:")
        for line in r.stderr.splitlines()[-15:]:
            print(f"    ! {line}")
        return False
    # any fallback line on stderr means the package did not run the C++ path
    bad = [l for l in r.stderr.splitlines()
           if "fallback" in l or "unavailable" in l]
    if bad:
        print("  FAIL: fallback triggered during official run:")
        for line in bad[:10]:
            print(f"    ! {line}")
        return False
    return _compare(pkg / _RESULTS_NAME, _ANCHOR)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("stage", "verify", "all"):
        print(__doc__)
        sys.exit(2)
    ok = True
    if mode in ("stage", "all"):
        ok &= stage()
    if ok and mode in ("verify", "all"):
        ok &= verify()
    print(f"\nM67-B MAKE-SUBMISSION [{mode}]: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
