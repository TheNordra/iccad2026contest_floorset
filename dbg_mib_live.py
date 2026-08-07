"""M37 liveness check: how many of the 100 cases have a RESHAPEABLE MIB group
(no-master / area-compatible square-fallback whose movable members are all interior)?
Runs constructive.exe with ICCAD_MIB_DBG=1 and counts the MIB_RESHAPEABLE stderr lines.
If ~0 cases, MIB-aspect is structurally dead -> skip to converge."""
import os, sys, subprocess
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator
from optimizer_claude import _serialize_input
from proxy_analysis import build_opt_target_pos
ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")

cases_with = 0; total_groups = 0; total_members = 0
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]; at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    env = dict(os.environ); env["ICCAD_MIB_DBG"] = "1"
    r = subprocess.run([EXE], input=txt, capture_output=True, text=True, env=env)
    lines = [l for l in r.stderr.splitlines() if l.startswith("MIB_RESHAPEABLE")]
    if lines:
        cases_with += 1; total_groups += len(lines)
        for l in lines:
            # MIB_RESHAPEABLE group=G size=S avg=A
            sz = int(l.split("size=")[1].split()[0]); total_members += sz
        print(f"case {idx:3d} n={n:3d}: {len(lines)} reshapeable group(s) -> {lines}")
print(f"\n=== {cases_with}/100 cases have >=1 reshapeable MIB group; "
      f"{total_groups} groups, {total_members} movable members total ===")
