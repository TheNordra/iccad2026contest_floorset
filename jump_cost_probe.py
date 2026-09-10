"""§3 probe: measure how much wall-time PUSH_JUMP / PUSH_SWAP cost on BIG cases,
and how much quality (cost) they buy. Single base profile, top-6 biggest cases.
If NO_JUMP saves negligible time -> §3 dead (jump already cheap/early-out).
"""
import os, sys, time, subprocess
from pathlib import Path
_DIR = Path(r"C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet")
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")

sizes = []
for idx in range(len(ev.dataset)):
    at = ev.dataset[idx]["input"][0]
    sizes.append((int((at != -1).sum().item()), idx))
sizes.sort(reverse=True)
big = sizes[:6]
print("big cases (n, idx):", big)

CONFIGS = {
    "baseline":      {},
    "NO_JUMP":       {"ICCAD_NO_JUMP": "1"},
    "NO_SWAP":       {"ICCAD_NO_SWAP": "1"},
    "NO_JUMP+SWAP":  {"ICCAD_NO_JUMP": "1", "ICCAD_NO_SWAP": "1"},
    "NO_PUSH":       {"ICCAD_NO_PUSH": "1"},
}

def run(txt, envx, reps=4):
    env = dict(os.environ); env.update(envx)
    best = 1e18; out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = subprocess.run([EXE], input=txt, capture_output=True, text=True, env=env)
        best = min(best, time.perf_counter() - t0); out = r.stdout
    return best, out

for (n, idx) in big:
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    print(f"\n=== case {idx}  n={n} ===")
    base_t = None
    for name, envx in CONFIGS.items():
        dt, out = run(txt, envx)
        ps = _parse_output(out, n)
        m = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                              b2b, p2b, pins, at[:n], target_positions=tp[:n],
                              median_runtime=1.0)
        if name == "baseline":
            base_t = dt
        rel = "" if name == "baseline" else f" ({100*(dt-base_t)/base_t:+5.0f}%)"
        print(f"  {name:14} {dt*1000:8.1f}ms{rel}   cost={m.cost:.5f}")
