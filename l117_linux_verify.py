"""OFFLINE (never shipped): run the FINAL tar on Linux, under the official command.

Supersedes the m67c_make_linux_bundle.py detour. That builder existed because the
box had no WSL, so a ~510MB bundle had to be carried to a machine that did; this
box now runs Ubuntu 26.04 under WSL2, so the same four checks run in place. The
driver body is m67c's T3/T4 logic kept verbatim where it still holds -- what
changed is the anchors and where the extraction lands.

WHAT IT PROVES (the parts that no Windows-side gate can reach):

  final     default cores. On this box WSL nproc == Windows os.cpu_count() == 32,
            so the tier gating picks the same branch and the answer must match
            results_M74_default.json. This is the C-binary lane: it proves the
            bundled ELF actually EXECUTES on Linux rather than merely being an
            ELF, and that the score survives a different libc/libm.

  final48   ICCAD_ADAPTIVE_CORES=48. This is the lane the grader actually runs
            and the one Windows structurally cannot see: clean_env() strips
            ICCAD_* (so a local run never enters it) and this box has 32 cores
            (so it never enters it by nproc either). It is also the only lane
            where the shape LP fires, which makes it the first cross-platform
            test of scipy/HiGHS determinism -- and that test came back negative,
            so this lane is judged on the shipping invariant instead of on
            bit-equality. See judge48().

  t4        corrupt bin/constructive_linux and run one case: the fallthrough
            must on-site-compile with the package's own g++ chain and land on
            the same cost. Proves the package is not silently dependent on a
            binary that happens to be prebuilt.

Both t3-style modes fail loudly on any 'fallback'/'unavailable'/'[constructive]'
line on stderr and on the presence of a constructive.exe compile artifact -- the
two ways the package can quietly degrade to the pure-Python SA and still print a
plausible-looking table.

ANCHORS. m67c compared the 48-core lane against results_M74_cores48.json, which
is four milestones stale (M80 tier, L110 route A, L114 LP, L116 sep_trim all
landed after it). The current 48-core ship anchor is
results_L114_48c_lp_anchor.json = 1.2367916697725434.

USAGE (from WSL, with the venv built by the scratchpad setup):
    L117_WORK=~/l117 ~/iccadvenv/bin/python l117_linux_verify.py final   build_submission/cadc1075.tar.gz
    L117_WORK=~/l117 ~/iccadvenv/bin/python l117_linux_verify.py final48 build_submission/cadc1075.tar.gz
    L117_WORK=~/l117 ~/iccadvenv/bin/python l117_linux_verify.py t4      build_submission/cadc1075.tar.gz

L117_WORK keeps the extracted package off /mnt/c: DrvFs is slow and creating the
dataset symlink there is not reliable. It defaults to this directory.
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
WORK = Path(os.environ.get("L117_WORK", str(ROOT))).expanduser()

ANCHOR = ROOT / "results_M74_default.json"            # 1.293461035226291
ANCHOR48 = ROOT / "results_L114_48c_lp_anchor.json"   # 1.2367916697725434
BASE48 = ROOT / "results_M80_48c_anchor.json"         # 1.2666234250706565, LP off

LOADERS = ("litetestLoader.py", "lite_dataset_test.py", "liteLoader.py",
           "lite_dataset.py", "prime_dataset.py", "cost.py", "utils.py",
           "visualize.py")
TEAM = "cadc1075"
BIN_NAMES = ("constructive_linux",)
ULP_TOL = 2e-9          # sub-2e-9 diffs are WARN (known: case 84 libm ULP)
TOTAL_TOL = 1e-9
T4_CASE = 50


def build_layout(workdir: Path, tar_path: Path) -> Path:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(workdir)
    pkg = workdir / TEAM
    assert (pkg / "op_wrapper.py").exists(), "extracted package incomplete"
    # grader-side overlay: evaluator inside the package dir; loader closure and
    # the dataset one level up (evaluator sys.path.insert(parent.parent) +
    # data_path default "../" resolved against cwd=pkg).
    shutil.copyfile(ROOT / "iccad2026contest" / "iccad2026_evaluate.py",
                    pkg / "iccad2026_evaluate.py")
    for f in LOADERS:
        shutil.copyfile(ROOT / f, workdir / f)
    os.symlink(str((ROOT / "LiteTensorDataTest").resolve()),
               str(workdir / "LiteTensorDataTest"), target_is_directory=True)
    return pkg


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def compare(new_path: Path, anchor: Path) -> bool:
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old = json.loads(anchor.read_text(encoding="utf-8"))
    fails, warns = [], []
    dt = abs(new["total_score"] - old["total_score"])
    print(f"   anchor    {anchor.name}")
    print(f"   total new={new['total_score']!r}")
    print(f"   total old={old['total_score']!r}  |d|={dt:.3e}")
    if dt > TOTAL_TOL:
        fails.append(f"total_score |d|={dt:.3e} > {TOTAL_TOL:g}")
    tn, to = new["test_results"], old["test_results"]
    if len(tn) != len(to):
        fails.append(f"case count {len(tn)} != {len(to)}")
    nfeas = 0
    worst_cost = 0.0
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
                if k == "cost":
                    worst_cost = max(worst_cost, d)
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
          f"{new['summary']['avg_runtime']}  worst |dcost|={worst_cost:.3e}")
    for w in warns[:10]:
        print(f"   ULP-WARN {w}")
    if len(warns) > 10:
        print(f"   ... and {len(warns) - 10} more ULP warns")
    for f_ in fails[:20]:
        print(f"   FAIL {f_}")
    if len(fails) > 20:
        print(f"   ... and {len(fails) - 20} more")
    if fails:
        return False
    print(f"   compare: OK ({len(tn)} cases; {len(warns)} sub-{ULP_TOL:g} "
          f"ULP warns)")
    return True


def t3(tar_path: Path, workname: str, cores: int = None) -> bool:
    print(f"-- layout from {tar_path.name} -> {WORK / workname}"
          + (f"  [ICCAD_ADAPTIVE_CORES={cores}]" if cores else "  [default cores]"))
    pkg = build_layout(WORK / workname, tar_path)
    for b in BIN_NAMES:
        p = pkg / "bin" / b
        if not p.exists():
            print(f"FAIL: package bin/{b} missing")
            return False
        with open(p, "rb") as fh:
            magic = fh.read(4)
        print(f"   bin/{b}: {p.stat().st_size} bytes, magic={magic!r}")
        if magic != b"\x7fELF":
            print(f"FAIL: bin/{b} is not an ELF -- a Windows PE here would be "
                  "exec-failed by the grader and fall through to the SA")
            return False
        p.chmod(0o755)
    res_name = f"results_l117_{workname}.json"
    cmd = [sys.executable, "-u", "iccad2026_evaluate.py", "--evaluate",
           "op_wrapper.py", "-o", res_name]
    print(f"-- official cmd (cwd={TEAM}): {' '.join(cmd[1:])}")
    env = clean_env()
    if cores:
        env["ICCAD_ADAPTIVE_CORES"] = str(cores)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(pkg), env=env, capture_output=True,
                       encoding="utf-8", errors="replace", timeout=7200)
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
    ok &= (judge48(pkg / res_name) if cores
           else compare(pkg / res_name, ANCHOR))
    return ok


def judge48(new_path: Path) -> bool:
    """The 48-core lane is NOT bit-reproducible across platforms, so bit-equality
    is the wrong gate for it -- measured, not assumed.

    The shape LP is massively degenerate; scipy's bundled HiGHS differs between
    the Windows env (scipy 1.15.3 / numpy 2.2.6 / py3.10) and this Ubuntu 26.04
    venv (scipy 1.18.0 / numpy 2.5.2 / py3.14), so the two land on different
    optima of the same program. Observed: 92/100 cases agree to <1e-9 and 8 move,
    with positions differing by whole units on the movers (worst 11.5) -- orders
    above any ULP band, and not a defect.

    What must hold is the invariant the LP was shipped on, so that is what is
    checked: every case feasible, no case worse than the pre-LP M80 anchor, and
    the total still ahead of it. The bit-comparison against the Windows anchor is
    kept as reporting only.
    """
    new = json.loads(new_path.read_text(encoding="utf-8"))
    base = json.loads(BASE48.read_text(encoding="utf-8"))
    win = json.loads(ANCHOR48.read_text(encoding="utf-8"))
    n = {r["test_id"]: r for r in new["test_results"]}
    b = {r["test_id"]: r for r in base["test_results"]}
    w = {r["test_id"]: r for r in win["test_results"]}
    fails = []
    infeas = [i for i in n if not n[i]["is_feasible"]]
    if infeas:
        fails.append(f"infeasible cases: {infeas[:10]}")
    reg = [i for i in b if i in n and n[i]["cost"] > b[i]["cost"] + 1e-12]
    if reg:
        fails.append(f"{len(reg)} cases worse than the pre-LP anchor: {reg[:10]}")
    gain = 100 * (1 - new["total_score"] / base["total_score"])
    if gain <= 0:
        fails.append(f"total {new['total_score']!r} is not ahead of the pre-LP "
                     f"anchor {base['total_score']!r}")
    moved = sorted(((abs(n[i]["cost"] - w[i]["cost"]), i) for i in w if i in n),
                   reverse=True)
    nmoved = sum(1 for d, _ in moved if d > 1e-9)
    print(f"   pre-LP anchor  {base['total_score']!r}")
    print(f"   windows LP     {win['total_score']!r}   "
          f"{100 * (1 - win['total_score'] / base['total_score']):+.4f}%")
    print(f"   this run       {new['total_score']!r}   {gain:+.4f}%")
    print(f"   feasible {sum(1 for i in n if n[i]['is_feasible'])}/{len(n)}   "
          f"regressions vs pre-LP {len(reg)}   "
          f"cases differing >1e-9 vs windows {nmoved}/{len(w)}")
    for d, i in moved[:5]:
        if d > 1e-9:
            print(f"     mover case {i:3d}: win {w[i]['cost']:.6f}  "
                  f"here {n[i]['cost']:.6f}  |d|={d:.2e}")
    for f_ in fails:
        print(f"   FAIL {f_}")
    return not fails


def t4(tar_path: Path) -> bool:
    pkg = build_layout(WORK / "t4", tar_path)
    (pkg / "bin").mkdir(exist_ok=True)
    (pkg / "bin" / "constructive_linux").write_bytes(
        b"\x7fELF garbage -- deliberately not runnable (L117 t4)")
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
    if os.name != "posix":
        print("FATAL: this is the Linux-side verify -- run it from WSL")
        sys.exit(2)
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    tar = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    if tar is not None and not tar.exists():
        print(f"FATAL: tar not found: {tar}")
        sys.exit(2)
    if mode == "final" and tar:
        ok = t3(tar, workname="t_final")
    elif mode == "final48" and tar:
        ok = t3(tar, workname="t_final48", cores=48)
    elif mode == "t4" and tar:
        ok = t4(tar)
    else:
        print(__doc__)
        sys.exit(2)
    print(f"L117 LINUX-VERIFY [{mode}]: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
