"""OFFLINE (never shipped): run the FINAL tar on Linux, under the official command.

Supersedes the m67c_make_linux_bundle.py detour. That builder existed because the
box had no WSL, so a ~510MB bundle had to be carried to a machine that did; this
box now runs Ubuntu under WSL2, so the same checks run in place.

WHAT IT PROVES (the parts that no Windows-side gate can reach):

  final     default cores. On this box WSL nproc == Windows os.cpu_count() == 32,
            so the tier gating picks the same branch and the answer must match
            the default-cores anchor bit-for-bit. This is the C-binary lane: it
            proves the bundled ELF actually EXECUTES on Linux rather than merely
            being an ELF, and that the score survives a different libc/libm.

  final48   ICCAD_ADAPTIVE_CORES=48. This is the lane the grader actually runs
            and the one Windows structurally cannot see: clean_env() strips
            ICCAD_* (so a local run never enters it) and this box has 32 cores
            (so it never enters it by nproc either). It is also the only lane
            where the shape LP fires, which makes it the first cross-platform
            test of scipy/HiGHS determinism -- and that test came back negative
            (L119), so this lane is judged on the shipping invariant instead of
            on bit-equality. See judge48().

  t4        corrupt bin/constructive_linux and run one case: the fallthrough
            must on-site-compile with the package's own g++ chain and land on
            the same cost. Proves the package is not silently dependent on a
            binary that happens to be prebuilt.

Both t3-style modes fail loudly on any 'fallback'/'unavailable'/'[constructive]'
line on stderr and on the presence of a constructive.exe compile artifact -- the
two ways the package can quietly degrade to the pure-Python SA and still print a
plausible-looking table.

L153 REWRITE (2026-08-20) -- three changes, each closing a hole that would have
let this gate pass while proving nothing:

  1. --env K=V re-admits ONE knob past the ICCAD_* strip and PRINTS it. The
     L147 config is four env flags that default OFF; under the old clean_env()
     they were stripped on the way in, so `final48` would have measured the
     shipped band, compared it against a shipped-band anchor, printed PASS, and
     said nothing at all about L147. This is HANDOFF_2026-08-20 §4.3 (the
     binary-override strip) in a second location.
  2. --stats forces ICCAD_SHAPE_LP_STATS and gates on the kept-rate. scipy
     absent, a malformed flag (`_shape_lp` swallows ValueError and drops the
     whole tangent dict), or `_LP_IMPORTS_OK` False all silently disable the LP
     with no stderr line at all. The stats file is the only positive evidence
     that the lane ran.
  3. judge48 takes its anchors on the command line and adds a LIVENESS gate
     against a control arm. Anchors were hardcoded four milestones stale
     (results_L114_48c_lp_anchor / results_M80_48c_anchor); worse, the "no case
     worse than the pre-LP anchor" rule was UNSATISFIABLE as written -- the
     already-uploaded L136 is itself worse than the M80 anchor on 2 cases. The
     rule is now "no case worse than the pre-LP base by more than --budget",
     where the budget is measured, not invented: it is the worst per-case
     regression the ALREADY-SHIPPED band shows against the same base.

ANCHORS are no longer baked in. Generate them on the tree under test with
l153_anchors.sh and pass them; a stale anchor prices someone else's change into
this verdict (HANDOFF_2026-08-20 §4.4).

USAGE (from WSL):
    V=~/iccadvenv/bin/python
    L117_WORK=~/l153 $V l117_linux_verify.py final build_submission/cadc1075.tar.gz \\
        --anchor results_L153_default_L137.json

    L117_WORK=~/l153 $V l117_linux_verify.py final48 build_submission/cadc1075.tar.gz \\
        --tag lpoff --base results_L153_lpoff_L137.json --no-judge --env ICCAD_SHAPE_LP=0

    L117_WORK=~/l153 $V l117_linux_verify.py final48 build_submission/cadc1075.tar.gz \\
        --tag ctrl --base results_L153_lpoff_L137.json --stats

    L117_WORK=~/l153 $V l117_linux_verify.py final48 build_submission/cadc1075.tar.gz \\
        --tag arm --base results_L153_lpoff_L137.json --win results_L147_on_L137.json \\
        --ctrl <ctrl results path printed above> --budget <measured> --stats \\
        --env ICCAD_SHAPE_LP_R=1.5 --env ICCAD_SHAPE_LP_G=1.10 \\
        --env ICCAD_SHAPE_LP_TOL=0.006 --env ICCAD_SHAPE_LP_PRICE=1.0

L117_WORK keeps the extracted package off /mnt/c: DrvFs is slow and creating the
dataset symlink there is not reliable. It defaults to this directory.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("L117_WORK", str(ROOT))).expanduser()

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


def clean_env(overrides=()) -> dict:
    """Grader-shaped environment: every ambient ICCAD_* stripped.

    `overrides` re-admits knobs ONE AT A TIME and prints each, so an A/B is
    explicit in the transcript instead of ambient (same doctrine as
    l113_ship_gate.py --env). Without this the L147 flags never reach the run.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    env["PYTHONIOENCODING"] = "utf-8"
    for kv in overrides:
        k, _, v = kv.partition("=")
        env[k] = v
        print(f"   env override: {k}={v}")
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


def _lp_gate_table(pkg: Path):
    """_L196_LPGATE as shipped INSIDE this package, or None if it predates L196.

    Read from the package, never from the tree: the tree is what we edit, the
    package is what the grader runs, and the whole point of a liveness gate is
    to catch the two drifting apart."""
    for name in ("op_src.py", "op_wrapper.py"):
        f = pkg / name
        if not f.exists():
            continue
        m = re.search(r"^_L196_LPGATE = \{.*?^\}", f.read_text(encoding="utf-8"),
                      re.S | re.M)
        if m:
            try:
                return {int(k): int(v) for k, v in
                        eval(m.group(0).split("=", 1)[1]).items()}
            except Exception:
                return None
    return None


def _lp_liveness(stats: Path, ncases: int, gate=None, blocks=None) -> bool:
    """Positive evidence that the shape LP ran on EXACTLY the cases it should.

    `_shape_lp_on()` returns False when scipy/shapely are missing, and
    `_shape_lp` swallows ValueError on a malformed flag and drops the whole
    tangent dict -- both silent, both leaving a run that looks normal. One
    `<block_count> <kept>` line per case is the only thing that distinguishes
    "the LP ran and declined" from "the LP was never reachable".

    L199: since L196 the LP is DELIBERATELY skipped on 37 of 100 block counts,
    so "one line per case" is no longer the right assertion -- it would fail a
    correct package. The gate is repointed rather than relaxed: the multiset of
    block counts in the stats file must equal exactly the block counts the
    shipped `_L196_LPGATE` selects out of the ones this run actually solved.
    That is strictly STRONGER than the old count check (it catches a table that
    fires on the wrong 63 as well as one that fires on too few), and it is the
    only form that can tell a live gate from a table that silently kept its old
    values -- which passes determinism and the kill switch while changing
    nothing. `gate=None` (a pre-L196 package) keeps the old one-per-case rule.
    """
    if not stats.exists():
        print("   FAIL LP liveness: no stats file -- the LP lane never ran "
              "(scipy missing, or the cores gate did not fire)")
        return False
    lines = [l.split() for l in stats.read_text().splitlines() if l.strip()]
    kept = sum(1 for l in lines if len(l) > 1 and l[1] == "1")
    ok = True

    if gate and blocks:
        want = sorted(n for n in blocks if gate.get(int(n), 1))
        got = sorted(int(l[0]) for l in lines if l and l[0].lstrip("-").isdigit())
        print(f"   LP liveness: {len(lines)}/{ncases} cases entered the LP; "
              f"the shipped gate selects {len(want)} of {len(blocks)}")
        if got != want:
            extra = sorted((Counter(got) - Counter(want)).elements())
            miss = sorted((Counter(want) - Counter(got)).elements())
            print(f"   FAIL LP liveness: the LP did not run on the gate's set. "
                  f"ran-but-must-not={extra[:12]} must-but-did-not={miss[:12]}")
            ok = False
        else:
            print(f"   LP gate: OK -- ran on exactly the {len(want)} selected "
                  f"block counts, skipped {len(blocks) - len(want)}")
    else:
        print(f"   LP liveness: {len(lines)}/{ncases} cases entered the LP, "
              f"kept {kept}/{len(lines)}")
        if len(lines) < ncases:
            print(f"   FAIL LP liveness: {ncases - len(lines)} case(s) never "
                  "reached the LP")
            ok = False

    if kept < 0.90 * max(1, len(lines)):
        print("   FAIL LP kept-rate below 90% -- a rejected case loses the "
              "whole shipped LP gain, not just the increment")
        ok = False
    return ok


def judge48(new_path: Path, base_path: Path, win_path=None, budget=0.0,
            ctrl_path=None, live_min=0.5) -> bool:
    """The 48-core lane is NOT bit-reproducible across platforms, so bit-equality
    is the wrong gate for it -- measured, not assumed (L119).

    The shape LP is massively degenerate; scipy's bundled HiGHS differs between
    the Windows env (scipy 1.15.3 / numpy 2.2.6 / py3.10) and this Ubuntu venv
    (scipy 1.18.0 / numpy 2.5.2 / py3.14), so the two land on different optima
    of the same program. Observed at L119: 92/100 cases agree to <1e-9 and 8
    move, with positions differing by whole units on the movers (worst 11.5) --
    orders above any ULP band, and not a defect.

    So the invariant the LP was shipped on is what gets checked:

      G-A  every case feasible
      G-B  no case worse than the pre-LP base by more than `budget`. The strict
           budget=0 form is UNSATISFIABLE for a real package -- the shipped band
           itself regresses 2 cases against its own pre-LP base -- so the budget
           is passed in, measured from the shipped band on the SAME base.
      G-C  the total is still ahead of the pre-LP base
      G-D  LIVENESS: if a control arm is supplied, this run must be ahead of it
           by at least `live_min` percent. Without this, a config whose flags
           were dropped on the way in passes G-A..G-C on the shipped band's own
           merits and reports a gain that has nothing to do with the change
           under test.

    The bit-comparison against the Windows anchor is kept as REPORTING only.
    """
    new = json.loads(new_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    n = {r["test_id"]: r for r in new["test_results"]}
    b = {r["test_id"]: r for r in base["test_results"]}
    fails = []
    infeas = [i for i in n if not n[i]["is_feasible"]]
    if infeas:
        fails.append(f"G-A infeasible cases: {infeas[:10]}")
    over = sorted(((n[i]["cost"] - b[i]["cost"], i) for i in b if i in n
                   if n[i]["cost"] > b[i]["cost"] + budget + 1e-12), reverse=True)
    reg = [i for i in b if i in n and n[i]["cost"] > b[i]["cost"] + 1e-12]
    if over:
        fails.append(f"G-B {len(over)} case(s) worse than the pre-LP base by "
                     f"more than the {budget:.6g} budget: "
                     f"{[(i, round(d, 6)) for d, i in over[:6]]}")
    gain = 100 * (1 - new["total_score"] / base["total_score"])
    if gain <= 0:
        fails.append(f"G-C total {new['total_score']!r} is not ahead of the "
                     f"pre-LP base {base['total_score']!r}")
    print(f"   pre-LP base    {base['total_score']!r}   ({base_path.name})")
    print(f"   this run       {new['total_score']!r}   {gain:+.4f}%")
    print(f"   feasible {sum(1 for i in n if n[i]['is_feasible'])}/{len(n)}   "
          f"regressions vs pre-LP {len(reg)} (budget {budget:.6g} -> "
          f"{len(over)} over)")

    if ctrl_path is not None:
        ctrl = json.loads(Path(ctrl_path).read_text(encoding="utf-8"))
        live = 100 * (1 - new["total_score"] / ctrl["total_score"])
        print(f"   control arm    {ctrl['total_score']!r}   "
              f"this run is {live:+.4f}% vs control   ({Path(ctrl_path).name})")
        if live < live_min:
            fails.append(f"G-D liveness: {live:+.4f}% vs the control arm is "
                         f"below the {live_min:g}% floor -- the flags under "
                         "test did not take effect")

    if win_path is not None:
        win = json.loads(Path(win_path).read_text(encoding="utf-8"))
        w = {r["test_id"]: r for r in win["test_results"]}
        moved = sorted(((abs(n[i]["cost"] - w[i]["cost"]), i)
                        for i in w if i in n), reverse=True)
        nmoved = sum(1 for d, _ in moved if d > 1e-9)
        print(f"   windows arm    {win['total_score']!r}   "
              f"{100 * (1 - win['total_score'] / base['total_score']):+.4f}% "
              f"vs pre-LP   ({Path(win_path).name})")
        print(f"   REPORTING ONLY: cases differing >1e-9 vs windows "
              f"{nmoved}/{len(w)}")
        for d, i in moved[:5]:
            if d > 1e-9:
                print(f"     mover case {i:3d}: win {w[i]['cost']:.6f}  "
                      f"here {n[i]['cost']:.6f}  |d|={d:.2e}")

    for f_ in fails:
        print(f"   FAIL {f_}")
    return not fails


def t3(tar_path: Path, workname: str, cores=None, overrides=(), stats=False,
       anchor=None, base=None, win=None, budget=0.0, ctrl=None, judge=True,
       live_min=0.5) -> bool:
    print(f"-- layout from {tar_path.name} -> {WORK / workname}"
          + (f"  [ICCAD_ADAPTIVE_CORES={cores}]" if cores else "  [default cores]"))
    pkg = build_layout(WORK / workname, tar_path)
    for bn in BIN_NAMES:
        p = pkg / "bin" / bn
        if not p.exists():
            print(f"FAIL: package bin/{bn} missing")
            return False
        with open(p, "rb") as fh:
            magic = fh.read(4)
        print(f"   bin/{bn}: {p.stat().st_size} bytes, magic={magic!r}")
        if magic != b"\x7fELF":
            print(f"FAIL: bin/{bn} is not an ELF -- a Windows PE here would be "
                  "exec-failed by the grader and fall through to the SA")
            return False
        p.chmod(0o755)
    res_name = f"results_l117_{workname}.json"
    cmd = [sys.executable, "-u", "iccad2026_evaluate.py", "--evaluate",
           "op_wrapper.py", "-o", res_name]
    print(f"-- official cmd (cwd={TEAM}): {' '.join(cmd[1:])}")
    env = clean_env(overrides)
    if cores:
        env["ICCAD_ADAPTIVE_CORES"] = str(cores)
        print(f"   env override: ICCAD_ADAPTIVE_CORES={cores}")
    stats_path = WORK / workname / "lp_stats.txt"
    if stats:
        if stats_path.exists():
            stats_path.unlink()
        env["ICCAD_SHAPE_LP_STATS"] = str(stats_path)
        print(f"   env override: ICCAD_SHAPE_LP_STATS={stats_path}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(pkg), env=env, capture_output=True,
                       encoding="utf-8", errors="replace", timeout=7200)
    wall = time.time() - t0
    print(f"-- eval done in {wall:.0f}s (exit {r.returncode})")
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
    # L164: POSITIVE assertion, not another absence check. The LP lane needs
    # scipy, and losing it costs 5.4171% of score (measured: hide scipy and the
    # in-set total goes 1.191977686767963 -> 1.260246745790688, exactly the
    # pre-LP lane) while being COMPLETELY SILENT -- every case still solves,
    # all 100 stay feasible, nothing is printed. Two official documents
    # disagree about whether the grader even provides scipy, so the optimizer
    # now states which one it got and this gate refuses to pass without it.
    src = [l.split("source=", 1)[1].strip()
           for l in r.stderr.splitlines() if l.startswith("[scipy] source=")]
    if not src:
        print("FAIL: no '[scipy] source=' marker -- cannot confirm the LP lane "
              "has its dependency")
        ok = False
    elif "absent" in src:
        print("FAIL: scipy absent -- the whole LP lane is inert, worth -5.4171%")
        ok = False
    else:
        print(f"   scipy: {sorted(set(src))[0]}")
    exe = pkg / "constructive.exe"
    if exe.exists():
        print("FAIL: constructive.exe exists -> on-site compile happened "
              "(bundled-binary-first did NOT engage)")
        ok = False
    else:
        print("   bundled-first: OK (no on-site compile artifact)")
    out = pkg / res_name
    _rows = json.loads(out.read_text(encoding="utf-8"))["test_results"]
    ncases = len(_rows)
    if stats:
        # The kill-switch lane runs the LP everywhere ON PURPOSE, so it is
        # judged by the old one-line-per-case rule; every other lane is judged
        # against the gate table the package actually carries.
        _killed = any(o.replace(" ", "") == "ICCAD_LP_GATE=0" for o in overrides)
        _gate = None if _killed else _lp_gate_table(pkg)
        ok &= _lp_liveness(stats_path, ncases, _gate,
                           [c["block_count"] for c in _rows])
    if cores:
        if judge:
            ok &= judge48(out, base, win, budget, ctrl, live_min)
        else:
            j = json.loads(out.read_text(encoding="utf-8"))
            nf = sum(1 for c in j["test_results"] if c["is_feasible"])
            print(f"   no-judge: total={j['total_score']!r} "
                  f"feasible={nf}/{ncases}")
            if base is not None:
                bt = json.loads(Path(base).read_text(encoding="utf-8"))
                print(f"   vs {Path(base).name}: "
                      f"{100 * (1 - j['total_score'] / bt['total_score']):+.4f}%")
            ok &= (nf == ncases)
    else:
        ok &= compare(out, anchor)
    print(f"   RESULT FILE: {out}")
    return ok


def t4(tar_path: Path, anchor: Path) -> bool:
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
    a = {c["test_id"]: c
         for c in json.loads(anchor.read_text(encoding="utf-8"))
         ["test_results"]}[T4_CASE]
    exe = pkg / "constructive.exe"
    d = abs(tr.cost - a["cost"])
    cost_ok = tr.cost == a["cost"] or d < ULP_TOL
    ok = (tr.error is None and tr.is_feasible and cost_ok and exe.exists())
    print(f"   case {T4_CASE}: cost new={tr.cost!r} anchor={a['cost']!r} "
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("final", "final48", "t4"))
    ap.add_argument("tar")
    ap.add_argument("--tag", default=None,
                    help="workdir/result suffix, so arms do not overwrite")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="re-admit one ICCAD_* knob past the ambient strip")
    ap.add_argument("--anchor", default=None,
                    help="final: bit-equality anchor (default-cores)")
    ap.add_argument("--base", default=None,
                    help="final48: pre-LP base for the shipping invariant")
    ap.add_argument("--win", default=None,
                    help="final48: windows arm, REPORTING ONLY")
    ap.add_argument("--ctrl", default=None,
                    help="final48: control arm for the liveness gate")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="per-case regression budget vs --base (measured)")
    ap.add_argument("--live-min", type=float, default=0.5,
                    help="minimum %% ahead of --ctrl for the liveness gate")
    ap.add_argument("--stats", action="store_true",
                    help="force ICCAD_SHAPE_LP_STATS and gate on the kept-rate")
    ap.add_argument("--no-judge", action="store_true",
                    help="final48: record the arm, skip the invariant")
    a = ap.parse_args()

    tar = Path(a.tar).resolve()
    if not tar.exists():
        print(f"FATAL: tar not found: {tar}")
        sys.exit(2)

    def _p(v):
        return (ROOT / v) if v and not Path(v).is_absolute() else (Path(v) if v else None)

    if a.mode == "final":
        anchor = _p(a.anchor)
        if anchor is None or not anchor.exists():
            print("FATAL: final needs --anchor <default-cores anchor on THIS tree>")
            sys.exit(2)
        ok = t3(tar, workname=a.tag or "t_final", anchor=anchor,
                overrides=a.env)
    elif a.mode == "final48":
        base = _p(a.base)
        if base is None or not base.exists():
            print("FATAL: final48 needs --base <pre-LP anchor on THIS tree>")
            sys.exit(2)
        ok = t3(tar, workname=a.tag or "t_final48", cores=48, overrides=a.env,
                stats=a.stats, base=base, win=_p(a.win), budget=a.budget,
                ctrl=_p(a.ctrl), judge=not a.no_judge, live_min=a.live_min)
    else:
        anchor = _p(a.anchor)
        if anchor is None or not anchor.exists():
            print("FATAL: t4 needs --anchor <default-cores anchor on THIS tree>")
            sys.exit(2)
        ok = t4(tar, anchor)
    print(f"L117 LINUX-VERIFY [{a.mode}{'/' + a.tag if a.tag else ''}]: "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
