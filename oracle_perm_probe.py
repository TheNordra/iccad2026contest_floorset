"""OFFLINE ONLY — oracle-perm ceiling probe (never shipped).

Injects an fp_sol-derived pack ORDER into constructive.exe (via ICCAD_ORDER_FILE)
and measures the true-cost oracle-min gain over the current portfolio, exactly
like profile_vs_portfolio.py. Decides whether pack-order still has ore:

    oracle-min gain >= 0.5%  -> ordering matters; pursue order-LNS / pair-reloc / ML
    oracle-min gain  < 0.2%  -> permanently close the ordering/ML branch

The order key is the ground-truth tp[i]=(x,y,w,h) of each block (the real layout
we are trying to reconstruct). Two axes:
  PROBE_KEY = yx (default) -> y*1e6 + x  (row-major sweep)
            = xy           -> x + y      (diagonal / BL sweep, teammate's cx+cy)
  PROBE_GLOBAL = 1         -> ignore bscore (pure global sweep, may blow up vBd)
              else         -> key is a within-bscore-class tiebreak (safe)

Usage:
  python oracle_perm_probe.py                    # bscore-internal, (y,x)
  PROBE_GLOBAL=1 python oracle_perm_probe.py     # global (y,x)
  PROBE_KEY=xy  python oracle_perm_probe.py      # bscore-internal, (x+y)
"""
import os, sys, math, json, subprocess, tempfile
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = os.environ.get("ICCAD_CONSTRUCTIVE_BIN", str(_DIR / "constructive.exe"))
PORT = json.load(open(_DIR / "iccad2026contest" / "optimizer_constructive_results.json"))
pcost = {t['test_id']: t['cost'] for t in PORT['test_results']}

GLOBAL = os.environ.get("PROBE_GLOBAL") == "1"
KEY = os.environ.get("PROBE_KEY", "yx")
print(f"probe: KEY={KEY} GLOBAL={GLOBAL}")

def keyfun(tx, ty):
    return (tx + ty) if KEY == "xy" else (ty * 1e6 + tx)

def run(idx):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    of = tempfile.NamedTemporaryFile("w", suffix=".ord", delete=False)
    for i in range(n):
        tx = float(tp[i][0]); ty = float(tp[i][1])
        of.write(f"{i} {keyfun(tx, ty):.6f}\n")
    of.close()
    env = dict(os.environ); env["ICCAD_ORDER_FILE"] = of.name
    if GLOBAL: env["ICCAD_ORDER_GLOBAL"] = "1"
    r = subprocess.run([EXE], input=txt, capture_output=True, text=True, env=env)
    os.unlink(of.name)
    ps = _parse_output(r.stdout, n)
    m = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                          b2b, p2b, pins, at[:n], target_positions=tp[:n], median_runtime=1.0)
    return n, m.cost

totW = 0.0; gain = 0.0; wins = []
for idx in range(100):
    n, cC = run(idx)
    w = math.exp(n / 12.0); totW += w
    pc = pcost[idx]
    if cC < pc - 1e-4:
        wins.append((w * (pc - cC), idx, n, pc, cC)); gain += w * (pc - cC)
wins.sort(reverse=True)
print("=== cases oracle-order BEATS current portfolio ===")
for g, idx, n, pc, cc in wins:
    print(f"  case {idx:3d} n={n:3d} port={pc:.4f} oracle={cc:.4f} gain={pc-cc:+.4f} wContr={g/totW*100:.3f}%")
print(f"\nportfolio Total = {PORT['total_score']:.4f}")
print(f"oracle-order oracle-min gain = {gain/totW*100:.3f}%  -> ~{PORT['total_score']-gain/totW:.4f}")
print("verdict:", ">=0.5% ordering HAS ore" if gain/totW >= 0.005
      else ("<0.2% close ordering/ML" if gain/totW < 0.002 else "0.2-0.5% marginal"))
