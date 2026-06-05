"""Constructive-portfolio ceiling + offline proxy search.

Runs several env-var profiles of constructive.exe over all cases, records each
profile's TRUE contest cost AND baseline-free metrics (bbox_area, hpwl, vrel),
then reports:
  - each profile alone (Total Score)
  - ORACLE (true-best per case) = portfolio ceiling
  - PROXY selectors (baseline-free) vs oracle  -> what a live wrapper can realize

Proxy rationale: true cost = 0.5*(area/A + hpwl/H)*exp(2*vrel). vrel is exact
from (positions, constraints) with no baseline. area baseline A ~= 1.035*ΣblockA.
hpwl baseline H has no stable constant, so we use a per-case scale (min hpwl over
profiles) and sweep its relative weight RH. Everything here is baseline-free and
usable live.
"""
import os, sys, math, subprocess
from pathlib import Path
import torch
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution, compute_total_score
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos

PROFILES = {
    # current shipped 7
    "base":        {},
    "wire_hi":     {"ICCAD_WIRE_MULT": "2.0"},
    "anc_lo":      {"ICCAD_ANCHOR_W": "0.04"},
    "area_lean":   {"ICCAD_WIRE_MULT": "0.5", "ICCAD_ANCHOR_W": "0.20"},
    "aspect_hi":   {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286"},
    "aspect_xhi":  {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20"},
    "asp_wire":    {"ICCAD_LR_ASPECT": "3.5", "ICCAD_TB_ASPECT": "0.286", "ICCAD_WIRE_MULT": "2.0"},
    # candidate additions (does the oracle ceiling drop?)
    "aspect_v7":   {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143"},
    "aspect_v10":  {"ICCAD_LR_ASPECT": "10.0", "ICCAD_TB_ASPECT": "0.10"},
    "asp7_wire":   {"ICCAD_LR_ASPECT": "7.0", "ICCAD_TB_ASPECT": "0.143", "ICCAD_WIRE_MULT": "2.0"},
    "wire_xhi":    {"ICCAD_WIRE_MULT": "5.0"},
    "asp5_anclo":  {"ICCAD_LR_ASPECT": "5.0", "ICCAD_TB_ASPECT": "0.20", "ICCAD_ANCHOR_W": "0.04"},
}
PNAMES = list(PROFILES)

ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
EXE = str(_DIR / "constructive.exe")

# rec[i] = {p: {cost,area,hpwl,vrel}}, plus n, sumA
recs = []; ns = []
for idx in range(len(ev.dataset)):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
    sumA = float(sum(float(at[i]) for i in range(n)))
    rec = {}
    for p, envd in PROFILES.items():
        env = dict(os.environ); env.update(envd)
        r = subprocess.run([EXE], input=txt, capture_output=True, text=True, env=env)
        ps = _parse_output(r.stdout, n)
        m = evaluate_solution({'positions': ps, 'runtime': 1.0}, base, cons[:n],
                              b2b, p2b, pins, at[:n], target_positions=tp[:n],
                              median_runtime=1.0)
        # C++-emitted metrics (what the live wrapper actually selects on)
        vrel_c = m.violations_relative
        for ln in r.stderr.splitlines():
            if ln.startswith("METRICS "):
                _, _a, _h, vbd, vcl, vmb, nsoft = ln.split()
                vrel_c = (int(vbd)+int(vcl)+int(vmb)) / max(int(nsoft), 1)
        rec[p] = dict(cost=m.cost, area=m.bbox_area, hpwl=m.hpwl_total,
                      vrel=m.violations_relative, vrel_c=vrel_c, feas=int(m.is_feasible))
    recs.append((rec, sumA)); ns.append(n)
    if (idx + 1) % 25 == 0:
        print(f"  {idx+1}/100", file=sys.stderr)

def oracle(sub):
    return compute_total_score(
        [min(recs[i][0][p]["cost"] for p in sub) for i in range(len(ns))], ns)

def proxy_pick(i, sub, RH, vkey="vrel"):
    rec, sumA = recs[i]
    A = 1.035 * max(sumA, 1e-9)
    hmin = min(rec[p]["hpwl"] for p in sub) or 1.0
    return min(sub, key=lambda p: (rec[p]["area"]/A + RH*rec[p]["hpwl"]/hmin) * math.exp(2.0*rec[p][vkey]))

def proxy_score(sub, RH=1.0, vkey="vrel"):
    return compute_total_score(
        [recs[i][0][proxy_pick(i, sub, RH, vkey)]["cost"] for i in range(len(ns))], ns)

print("\n=== per-profile Total Score ===")
for p in PNAMES:
    print(f"  {p:11s} {compute_total_score([recs[i][0][p]['cost'] for i in range(len(ns))], ns):.4f}")

# weighted win share over the FULL set (which profiles are ever the true best)
wins = {p: 0.0 for p in PNAMES}
for i in range(len(ns)):
    wins[min(PNAMES, key=lambda p: recs[i][0][p]["cost"])] += math.exp(ns[i] / 12.0)
totw = sum(wins.values())
print("\n=== win share (weighted, full set) ===")
for p in sorted(PNAMES, key=lambda p: -wins[p]):
    print(f"  {p:11s} {100*wins[p]/totw:5.1f}%")

# marginal value: oracle + live-proxy(shapely vrel) for growing subsets
S7 = ["base","wire_hi","anc_lo","area_lean","aspect_hi","aspect_xhi","asp_wire"]
SUBSETS = {
    "S7 (shipped)":      S7,
    "S7+v7":             S7 + ["aspect_v7"],
    "S7+v7+v10":         S7 + ["aspect_v7","aspect_v10"],
    "S7+v7+v10+asp7w":   S7 + ["aspect_v7","aspect_v10","asp7_wire"],
    "ALL":               PNAMES,
}
print("\n=== subset:  oracle  /  proxy(shapely vrel, RH=1) ===")
for name, sub in SUBSETS.items():
    print(f"  {name:18s} {oracle(sub):.4f}  /  {proxy_score(sub):.4f}")

# C++ vrel (live-without-shapely) vs harness/shapely vrel, full set
flip = sum(1 for i in range(len(ns))
           if proxy_pick(i, PNAMES, 1.0, "vrel") != proxy_pick(i, PNAMES, 1.0, "vrel_c"))
print(f"\nC++ vs shapely vrel flips pick on {flip}/{len(ns)} cases "
      f"(proxy C++ vrel = {proxy_score(PNAMES, 1.0, 'vrel_c'):.4f})")
