"""OFFLINE (never shipped): one-shot regression suite for the shipped chain.

Runs, in order, the pre-submission gates CLAUDE.md requires (Beta/Final §3):

  m48      m48_coldstart_dryrun.py        4 phases; PASS = exit 0 + "ALL PASS"
  rf       rf_score_model.py              internal asserts (M42 drift, M45
                                          tier-3/4 drift, alpha=1 sanity);
                                          PASS = exit 0 + "RF MODEL DONE"
  m49big   m49_refine_probe.py variant 4 big   control K=12 vs cache EXACT 100%
  m49mid8  m49_refine_probe.py variant 8 mid   + movers == pinned expectation
  m49mid4  m49_refine_probe.py variant 4 mid   (mid4 has none pinned -> review)
  m47b     m47b_proxy_equiv.py            PASS = exit 0 + "PASS: N comparisons,
                                          mismatches=0" (script never exits
                                          non-zero on its own -> parsed here)

Per-item PASS/FAIL summary at the end; exit 1 if anything failed. Touches no
shipped file (pure subprocess runner; m48 works in a tempdir, m49 only writes
its own m49_pm_cache.pkl probe cache). Child env is stripped of all ICCAD_*
variables so the gates see shipped defaults.

The two mid gates share one process by default (`variant 8,4 mid` -- mode_variant
runs the K=12 control once, mechanically identical to two separate runs);
--separate-mid restores the literal documented invocations.

Prereqs: fresh audit_cache.pkl for the current pool (rf/m49 refuse a stale
signature themselves) and the iccadv conda env. Expect ~25-45 min total.

Usage:
  python regression_suite.py                    # everything
  python regression_suite.py --only m49big,m47b
  python regression_suite.py --list
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
_CONDA_PY = Path(r"C:\Users\Nordra\.conda\envs\iccadv\python.exe")
_PY_DEFAULT = str(_CONDA_PY if _CONDA_PY.exists() else Path(sys.executable))
ALL_KEYS = ("m48", "rf", "m49big", "m49mid8", "m49mid4", "m47b")
TAIL = 15

# Pinned mover expectations for the m49 gates (case ids whose Path-B cost moves
# vs the K=12 control). Sources: M49 ship (big K=4 -> case 85 only) and M50 ship
# (mid K=8 -> the 6 predicted movers); both re-confirmed at the M51 re-gate.
# mid K=4 was never pinned as a gate criterion (M50 measured ~11 movers) ->
# None = report the list for human review, control-EXACT is still the hard gate.
EXPECT_MOVERS = {
    (4, "big"): {85},
    (8, "mid"): {49, 54, 61, 64, 68, 70},
    (4, "mid"): None,
}

_KHDR = re.compile(r"^K=(\d+): exact (\d+)/(\d+)")
_PATHB_ROW = re.compile(
    r"^\s+(\d+)\s+\d+\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"
    r"\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+(?:WIN|lose)\s*$")
_M47B_VERDICT = re.compile(r"^(PASS|FAIL): (\d+) comparisons, mismatches=(\d+)")


def child_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_streamed(key, cmd, log_path, env):
    print(f"\n{'=' * 72}\n[{key}] {' '.join(cmd)}\n{'=' * 72}", flush=True)
    t0 = time.time()
    lines = []
    with open(log_path, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, cwd=str(_DIR), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             bufsize=1)
        for line in p.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            lf.write(line + "\n")
            lf.flush()
            print(f"  | {line}", flush=True)
        rc = p.wait()
    dur = time.time() - t0
    print(f"[{key}] exit={rc} ({dur:.0f}s)", flush=True)
    return rc, lines, dur


def _grep(lines, sub):
    return [l for l in lines if sub in l]


# ── per-item checkers: (rc, lines) -> [(row_name, ok, detail), ...] ──────────
def check_m48(rc, lines):
    final = _grep(lines, "M48 COLD-START DRY-RUN:")
    ok = rc == 0 and any("ALL PASS" in l for l in final)
    detail = final[-1].strip() if final else f"no verdict line (exit={rc})"
    nfail = sum(1 for l in lines if re.match(r"^phase \d.*FAIL", l))
    if nfail:
        detail += f"; {nfail} phase FAIL line(s)"
    return [("m48 coldstart 4-phase", ok, detail)]


def check_rf(rc, lines):
    ok = rc == 0 and any(l.strip() == "RF MODEL DONE" for l in lines)
    keys = [l.strip() for l in lines
            if l.startswith("SANITY:") or "shipped-chain check" in l
            or l.startswith("sanity: alpha=1.0")]
    detail = "; ".join(keys) if keys else "no assert-checkpoint lines"
    if not ok:
        detail = f"exit={rc}, no 'RF MODEL DONE'; " + detail
    return [("rf_score_model asserts", ok, detail)]


def make_check_m49(band, ks):
    def fn(rc, lines):
        ctl = _grep(lines, "control K=12 vs cache")
        control_ok = bool(ctl) and "EXACT 100%" in ctl[-1]
        sections, cur = {}, None
        for l in lines:
            m = _KHDR.match(l)
            if m:
                cur = int(m.group(1))
                sections[cur] = {"exact": (int(m.group(2)), int(m.group(3))),
                                 "lines": []}
                continue
            if cur is not None:
                sections[cur]["lines"].append(l)
        rows = []
        for K in ks:
            name = f"m49 variant {K} {band}"
            expected = EXPECT_MOVERS.get((K, band))
            if K not in sections:
                rows.append((name, False,
                             f"K={K} section missing (exit={rc}; control bad "
                             f"or crash)"))
                continue
            movers = set()
            for l in sections[K]["lines"]:
                m = _PATHB_ROW.match(l)
                if m and m.group(2) != m.group(3):
                    movers.add(int(m.group(1)))
            a, b = sections[K]["exact"]
            ok = rc == 0 and control_ok
            det = [f"control {'EXACT 100%' if control_ok else 'NOT EXACT <- BAD SESSION'}",
                   f"exact pairs {a}/{b}", f"movers {sorted(movers)}"]
            if expected is not None:
                if movers == expected:
                    det.append("== expected")
                else:
                    ok = False
                    det.append(f"!= expected {sorted(expected)}")
            else:
                det.append("(no pinned expectation -> review)")
            rows.append((name, ok, "; ".join(det)))
        return rows
    return fn


def check_m47b(rc, lines):
    m = None
    for l in lines:
        mm = _M47B_VERDICT.match(l)
        if mm:
            m = mm
    ok = (rc == 0 and m is not None and m.group(1) == "PASS"
          and int(m.group(2)) > 0 and m.group(3) == "0")
    detail = m.group(0) if m else f"no verdict line (exit={rc})"
    speed = _grep(lines, "proxy time old=")
    if speed:
        detail += "; " + speed[-1].strip()
    return [("m47b proxy equivalence", ok, detail)]


def build_runs(sel, separate_mid, py):
    """-> [(key, cmd, checker)]; the combined mid run yields two result rows."""
    runs = []
    if "m48" in sel:
        runs.append(("m48", [py, "-u", "m48_coldstart_dryrun.py"], check_m48))
    if "rf" in sel:
        runs.append(("rf", [py, "-u", "rf_score_model.py"], check_rf))
    if "m49big" in sel:
        runs.append(("m49big", [py, "-u", "m49_refine_probe.py", "variant", "4", "big"],
                     make_check_m49("big", [4])))
    mid8, mid4 = "m49mid8" in sel, "m49mid4" in sel
    if mid8 and mid4 and not separate_mid:
        runs.append(("m49mid", [py, "-u", "m49_refine_probe.py", "variant", "8,4", "mid"],
                     make_check_m49("mid", [8, 4])))
    else:
        if mid8:
            runs.append(("m49mid8", [py, "-u", "m49_refine_probe.py", "variant", "8", "mid"],
                         make_check_m49("mid", [8])))
        if mid4:
            runs.append(("m49mid4", [py, "-u", "m49_refine_probe.py", "variant", "4", "mid"],
                         make_check_m49("mid", [4])))
    if "m47b" in sel:
        runs.append(("m47b", [py, "-u", "m47b_proxy_equiv.py"], check_m47b))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help=f"comma list of {','.join(ALL_KEYS)}")
    ap.add_argument("--separate-mid", action="store_true",
                    help="run the two mid gates as separate invocations "
                         "(literal documented form; duplicates the K=12 control)")
    ap.add_argument("--python", default=_PY_DEFAULT,
                    help=f"interpreter for the tools (default {_PY_DEFAULT})")
    ap.add_argument("--logdir", help="log directory (default regression_logs/<ts>)")
    ap.add_argument("--list", action="store_true", help="list items and exit")
    args = ap.parse_args()

    sel = set(args.only.split(",")) if args.only else set(ALL_KEYS)
    bad = sel - set(ALL_KEYS)
    if bad:
        ap.error(f"unknown item(s) {sorted(bad)}; choose from {ALL_KEYS}")

    runs = build_runs(sel, args.separate_mid, args.python)
    if args.list:
        for key, cmd, _ in runs:
            print(f"{key:<8} {' '.join(cmd[1:])}")
        return

    logdir = Path(args.logdir) if args.logdir else \
        _DIR / "regression_logs" / time.strftime("%Y%m%d_%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    env = child_env()
    stripped = sorted(k for k in os.environ if k.startswith("ICCAD_"))
    print(f"regression suite: {len(runs)} run(s) -> items {sorted(sel)}")
    print(f"python: {args.python}")
    print(f"logs:   {logdir}")
    if stripped:
        print(f"note: stripped from child env: {stripped}")

    results = []          # (row_name, ok, detail, key, dur)
    logs = {}             # key -> (log_path, lines)
    t0 = time.time()
    for key, cmd, checker in runs:
        log_path = logdir / f"{key}.log"
        try:
            rc, lines, dur = run_streamed(key, cmd, log_path, env)
        except Exception as e:          # spawn failure etc.
            results.append((key, False, f"launch error: {e}", key, 0.0))
            logs[key] = (log_path, [])
            continue
        logs[key] = (log_path, lines)
        for name, ok, detail in checker(rc, lines):
            results.append((name, ok, detail, key, dur))

    total = time.time() - t0
    print(f"\n{'=' * 72}\nREGRESSION SUITE SUMMARY  ({total:.0f}s total)\n{'=' * 72}")
    all_ok = True
    for name, ok, detail, key, dur in results:
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} [{key} {dur:>5.0f}s]  {detail}")
    for name, ok, detail, key, _ in results:
        if not ok and logs.get(key, (None, []))[1]:
            print(f"\n---- tail of {key}.log ({name}) ----")
            for l in logs[key][1][-TAIL:]:
                print(f"  {l}")
    print(f"\nlogs: {logdir}")
    print(f"OVERALL: {'ALL PASS' if all_ok else 'FAILURES'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
