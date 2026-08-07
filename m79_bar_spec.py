"""M79 Gate 0-C — invert the ship bar into a per-case spec for a 42nd candidate.

OFFLINE TOOL — never shipped. Pure arithmetic over audit_cache_ship.pkl plus the
already-dumped fp_sol verbatim costs; no solving.

Gate 0-A/0-B answer "is there headroom". This answers the other half a model
designer needs: WHERE the headroom would have to come from, HOW MUCH per case,
and HOW FAST inference has to be before the candidate starts paying for itself in
RuntimeFactor.

Three tables:
  1. QUOTA      for a target portfolio gain G, the uniform relative improvement a
                candidate must deliver if it only wins on the top-K weighted cases
  2. HEADROOM   per case, base - fp_sol (the most any candidate could ever take),
                cumulated by weight -> where the 14.4% ceiling actually lives
  3. DT BUDGET  at 48 cores the wall is the max-setter (M67-E, 100/100), so a
                candidate is FREE iff its dt stays under the incumbent max on that
                case. Reported per band, since that is the real inference budget.

Run:  <python> m79_bar_spec.py
"""
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in sorted(k for k in os.environ if k.startswith("ICCAD_")):
    del os.environ[_k]

from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402
import optimizer_constructive as oc                                 # noqa: E402

RH, GAMMA, CORES = 1.4, 0.3, 48
TARGETS = [0.05, 0.30, 1.00]            # % — in-set house bar, OOS ship bar, G0 bar
SHIP_CACHE = _DIR / "audit_cache_ship.pkl"

print("[m79c] loading dataset + ship audit cache ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
CASES = []
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _at, _b2b, _p2b, _pins, _cons = _s["input"]
    _n = int((_at != -1).sum().item())
    _base, _tp = _ev._extract_baseline(_idx, _s["label"], _b2b, _p2b, _pins, _n)
    build_opt_target_pos(_tp, _cons, _n)
    _sumA = sum(max(0.0, float(_at[i])) for i in range(_n))
    CASES.append(dict(idx=_idx, n=_n, A_hat=1.035 * max(_sumA, 1e-9),
                      w=math.exp(_n / 12.0), base=_base, tp=_tp,
                      at=_at, b2b=_b2b, p2b=_p2b, pins=_pins, cons=_cons))
TOTW = sum(c["w"] for c in CASES)

_ac = pickle.load(open(SHIP_CACHE, "rb"))
D = _ac["data"]
N_LIVE = oc._M55_BASE_LEN


def _pool(n):
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(CORES)
    try:
        return [i for i in oc._pool_indices(n) if i < N_LIVE]
    finally:
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)


def _cost(c, pos):
    m = evaluate_solution({"positions": pos, "runtime": 1.0}, c["base"],
                          c["cons"][:c["n"]], c["b2b"], c["p2b"], c["pins"],
                          c["at"][:c["n"]], target_positions=c["tp"][:c["n"]],
                          median_runtime=1.0)
    return float(m.cost)


def _proxy(c, pos):
    m = oc._proxy_metrics(pos, c["at"], c["b2b"], c["p2b"], c["pins"],
                          c["cons"], c["n"])
    return m["area"], m["hpwl"], m["vrel"]


# ── incumbent portfolio at 48 cores ─────────────────────────────────────────
rows = []
for c in CASES:
    ci, pool = c["idx"], _pool(c["n"])
    pms = {k: _proxy(c, D[(ci, k)][0]) for k in pool}
    hmin = min(v[1] for v in pms.values()) or 1.0
    k = min(pool, key=lambda k: (pms[k][0] / c["A_hat"] + RH * pms[k][1] / hmin)
            * math.exp(2 * pms[k][2]))
    ts = [D[(ci, j)][1] for j in pool]
    rows.append(dict(ci=ci, n=c["n"], w=c["w"], cost=_cost(c, D[(ci, k)][0]),
                     tmax=max(ts), tsum=sum(ts)))

WBASE = sum(r["w"] * r["cost"] for r in rows)
print(f"  incumbent (M74 @48c)   {WBASE / TOTW:.9f}")

# fp_sol floor, from the G0-D dump if present (else computed here)
_fp = _DIR / "m79_fpsol_verbatim.json"
if _fp.exists():
    lab = {int(t["test_id"]): float(t["cost"])
           for t in json.loads(_fp.read_text(encoding="utf-8"))["test_results"]}
else:
    lab = {c["idx"]: _cost(c, [tuple(float(v) for v in c["tp"][i])
                               for i in range(c["n"])]) for c in CASES}
print(f"  fp_sol floor           {sum(r['w'] * lab[r['ci']] for r in rows) / TOTW:.9f}")

byw = sorted(rows, key=lambda r: -r["w"])

# ── 1. QUOTA ────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("1. QUOTA — uniform relative gain needed if the candidate wins only the "
      "top-K\n   weighted cases (and ties everywhere else)")
print("=" * 78)
print(f"  {'K':>4} {'weight share':>13} " +
      " ".join(f"{f'need for {g:.2f}%':>15}" for g in TARGETS))
for K in (1, 3, 5, 10, 20, 40, 100):
    sub = byw[:K]
    wc = sum(r["w"] * r["cost"] for r in sub)
    cells = []
    for g in TARGETS:
        need = g / 100 * WBASE / wc
        cells.append(f"{100 * need:>14.2f}%" if need <= 1 else f"{'IMPOSSIBLE':>15}")
    print(f"  {K:>4} {100 * wc / WBASE:>12.1f}% " + " ".join(cells))

# ── 2. HEADROOM ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("2. HEADROOM — per-case distance to fp_sol, i.e. the most ANY candidate "
      "could take")
print("=" * 78)
hr = sorted(((r["w"] * (r["cost"] - lab[r["ci"]]) / WBASE, r) for r in rows),
            reverse=True)
cum = 0.0
print(f"  {'case':>5} {'n':>4} {'portfolio':>10} {'fp_sol':>9} {'rel':>7} "
      f"{'w*d share':>10} {'cum':>8}")
for share, r in hr[:15]:
    cum += share
    print(f"  {r['ci']:>5} {r['n']:>4} {r['cost']:>10.5f} {lab[r['ci']]:>9.5f} "
          f"{100 * (1 - lab[r['ci']] / r['cost']):>6.1f}% {100 * share:>9.3f}% "
          f"{100 * cum:>7.3f}%")
tot_hr = sum(s for s, _ in hr)
print(f"  ... total headroom to the label floor {100 * tot_hr:.3f}%")
for K in (5, 10, 20, 40):
    print(f"    top-{K:>3} weighted cases hold "
          f"{100 * sum(s for s, _ in hr[:K]):>6.3f}% of it")

# ── 3. DT BUDGET ────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"3. DT BUDGET @{CORES}c — a candidate is FREE while dt <= the incumbent "
      f"max-setter\n   (adding dt also lifts sum/cores; both terms shown)")
print("=" * 78)
BANDS = [(0, 60), (60, 100), (100, 10 ** 9)]
print(f"  {'band':>12} {'cases':>6} {'free dt (min tmax)':>20} "
      f"{'p50 tmax':>10} {'max tmax':>10}")
for lo, hi in BANDS:
    sub = [r for r in rows if lo < r["n"] <= hi]
    if not sub:
        continue
    tm = sorted(r["tmax"] for r in sub)
    lbl = f"({lo},{hi if hi < 10 ** 9 else 'inf'}]"
    print(f"  {lbl:>12} {len(sub):>6} {tm[0]:>19.2f}s {tm[len(tm) // 2]:>9.2f}s "
          f"{tm[-1]:>9.2f}s")
print("\n  cost of overshooting the budget (weighted dRF, uniform dt):")
print(f"  {'dt':>8} {'dRF@48c':>10}  {'cases setting a new wall':>26}")
for dt in (0.5, 1.0, 2.0, 4.0, 8.0):
    acc, over = 0.0, 0
    for r in rows:
        old = max(r["tmax"], r["tsum"] / CORES)
        new = max(max(r["tmax"], dt), (r["tsum"] + dt) / CORES)
        over += new > old + 1e-12
        acc += r["w"] * ((new / old) ** GAMMA - 1.0)
    print(f"  {dt:>7.1f}s {100 * acc / TOTW:>+9.3f}%  {over:>21}/100")
print("\n  read: any gain a model delivers must clear the dRF of its own dt.")
