"""OFFLINE ONLY — M68 ML-seed oracle-ceiling probe (never shipped).

Stage 0 kill test for the ML-seed insertion point (plan ml-40-mighty-gray.md).
BEFORE any training, inject the fp_sol per-block CENTER as a greedy ANCHOR seed (the
strongest thing an ML pose head could ever predict) via constructive_m68.exe's
default-off ICCAD_ML_ANCHOR knob, and measure whether a PERFECT position seed lets the
greedy beat (a) the same profile WITHOUT the seed [isolates the anchor mechanism] and
(b) the shipped 41-profile portfolio [the real bar]. Mirrors oracle_perm_probe.py,
which closed the pack-ORDER axis at +0.005%.

Per-registered verdict on the weighted oracle-min gain vs portfolio (RF=1.0):
    >= 0.50%  -> position seeds have ore; escalate to LP + 1.2978 bar (Stage 0b)
    <  0.20%  -> greedy ignores position seeds -> ML-seed RED, do NOT train
    0.2-0.5%  -> marginal, escalate cautiously

Modes:
  gate0            byte-gate: constructive_m68.exe (no env) == constructive.exe
  oracle [K]       inject fp_sol center anchors on K cases (default 100), ANCHOR_W sweep

Usage:
  <python> m68_ml_seed_probe.py gate0
  <python> m68_ml_seed_probe.py oracle
  <python> m68_ml_seed_probe.py oracle 15
"""
import os, sys, math, json, subprocess, tempfile
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

EXE_SHIP = str(_DIR / "constructive.exe")
EXE_M68 = str(_DIR / "constructive_m68.exe")

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
PORT = json.load(open(_DIR / "iccad2026contest" / "optimizer_constructive_results.json"))
pcost = {t["test_id"]: t["cost"] for t in PORT["test_results"]}

# Ceiling sweep: how hard to pull each block to its (perfect) seed center. 0.1 = shipped
# default ANCHOR_W (mild); 20 = anchor term dominates the greedy score.
ANCHOR_WS = ["0.1", "1.0", "5.0", "20.0"]


def _case(idx):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    return n, at, b2b, p2b, pins, cons, base, tp, txt


def _run(exe, txt, env_extra=None):
    env = dict(os.environ)
    # strip any inherited ICCAD_* so the base run is truly the shipped default
    for k in list(env):
        if k.startswith("ICCAD_"): del env[k]
    if env_extra: env.update(env_extra)
    r = subprocess.run([exe], input=txt, capture_output=True, text=True, env=env)
    return r.stdout


def _anchor_file(n, tp):
    f = tempfile.NamedTemporaryFile("w", suffix=".anc", delete=False)
    for i in range(n):
        cx = float(tp[i][0]) + float(tp[i][2]) / 2.0
        cy = float(tp[i][1]) + float(tp[i][3]) / 2.0
        f.write(f"{i} {cx:.6f} {cy:.6f}\n")
    f.close(); return f.name


def _cost(ps, base, cons, b2b, p2b, pins, at, tp, n):
    m = evaluate_solution({"positions": ps, "runtime": 1.0}, base, cons[:n],
                          b2b, p2b, pins, at[:n], target_positions=tp[:n], median_runtime=1.0)
    return m.cost, m.is_feasible


def gate0():
    sample = [0, 25, 50, 78, 99]
    ok = True
    for idx in sample:
        n, at, b2b, p2b, pins, cons, base, tp, txt = _case(idx)
        a = _run(EXE_SHIP, txt); b = _run(EXE_M68, txt)
        same = (a == b)
        ok = ok and same
        print(f"  gate0 case {idx:3d} n={n:3d} byte-identical={same}")
    print("GATE0:", "PASS" if ok else "FAIL (m68 off-path diverges from shipped!)")
    return ok


def oracle(K=100):
    totW = 0.0
    gain_port = 0.0; gain_base = 0.0
    wins = []; infeas = 0; anchor_beats_base = 0
    for idx in range(K):
        n, at, b2b, p2b, pins, cons, base, tp, txt = _case(idx)
        # (0) same profile WITHOUT the seed = shipped base profile
        c_base, feas_base = _cost(_parse_output(_run(EXE_M68, txt), n),
                                  base, cons, b2b, p2b, pins, at, tp, n)
        # (1) best over the ANCHOR_W sweep WITH the perfect fp_sol seed
        af = _anchor_file(n, tp)
        best = None
        for aw in ANCHOR_WS:
            ps = _parse_output(_run(EXE_M68, txt, {"ICCAD_ML_ANCHOR": af, "ICCAD_ANCHOR_W": aw}), n)
            c, feas = _cost(ps, base, cons, b2b, p2b, pins, at, tp, n)
            if feas and (best is None or c < best): best = c
        os.unlink(af)
        if best is None: infeas += 1; best = 10.0
        w = math.exp(n / 12.0); totW += w
        pc = pcost[idx]
        if best < c_base - 1e-4: anchor_beats_base += 1
        if best < c_base - 1e-9: gain_base += w * (c_base - best)
        if best < pc - 1e-4:
            wins.append((w * (pc - best), idx, n, pc, best, c_base)); gain_port += w * (pc - best)
    wins.sort(reverse=True)
    print("=== cases where oracle-ANCHOR beats the 41-profile portfolio ===")
    for g, idx, n, pc, cc, cb in wins[:25]:
        print(f"  case {idx:3d} n={n:3d} port={pc:.4f} anchor={cc:.4f} base={cb:.4f} "
              f"gain={pc-cc:+.4f} wContr={g/totW*100:.3f}%")
    print(f"\ncases={K}  port-movers={len(wins)}  anchor-beats-base={anchor_beats_base}  infeasible={infeas}")
    print(f"[isolate anchor] weighted gain vs SAME-profile-no-seed = {gain_base/totW*100:.3f}%")
    print(f"[real bar]       weighted gain vs PORTFOLIO            = {gain_port/totW*100:.3f}%"
          f"  -> ~{PORT['total_score']-gain_port/totW:.4f}")
    v = gain_port / totW
    print("VERDICT:", ">=0.5% seeds HAVE ore -> escalate to LP+1.2978 (Stage 0b)" if v >= 0.005
          else ("<0.2% -> ML-seed RED, do NOT train" if v < 0.002 else "0.2-0.5% marginal"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate0"
    if mode == "gate0":
        gate0()
    elif mode == "oracle":
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        oracle(K)
    else:
        print("modes: gate0 | oracle [K]")
