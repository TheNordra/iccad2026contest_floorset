"""L125 — is a bounded beam in pack_in_frame worth anything on QUALITY?

OFFLINE PROBE — never shipped. Drives `constructive_l125.exe` only; the shipping
exe is never touched and its md5 never moves, so every offline cache stays valid.

WHAT B0 ALREADY SETTLED (2026-08-12, cache analysis, no code). At 48 cores the
wall is the max-setter on 100/100 cases, so a beam profile costs whatever it adds
to the WALL, not what it adds to itself, and beam dt ~ W x the beamed profile's
dt. Weighted dRF over `audit_cache_ship.pkl`:

    W=2, best source #33   +0.0002%      <- affordable
    W=3, best source #9    +0.2039%      <- two thirds of the 0.30% bar
    W=4, best source #31   +5.2830%      <- dead

So the only question left is quality at W=2, and this file asks it.

THE RISK, WRITTEN DOWN BEFORE THE CODE (from `l125-beam-affordable-only-at-w2`):
`layout_score` has no monotone relationship to final cost on partial layouts —
that is exactly what killed the frame-level B&B — so the beam's own pruning
criterion is unreliable and top-W need not retain the eventual winner. A beam
tolerates that better than a B&B (it keeps W, not 1) but this is where it fails
if it fails.

MODES
  offpath  the two bit-identity gates. ICCAD_BEAM unset must reproduce
           constructive.exe; and ICCAD_BEAM=1 ICCAD_BEAM_W=1 must ALSO reproduce
           it, which is what gates the duplicated placement logic in
           pack_in_frame_beam (a beam of width 1 IS the greedy).
  ab       liveness + solo quality + measured cost multiplier for one (W,K).
  sweep    ab over a small (W,K) grid on the heavy band only.

Run (PowerShell, and do NOT put the compiler on PATH for a probe — the wrapper's
compile chain would overwrite _BIN, which here is the probe binary):
  <python> -u l125_beam_probe.py offpath      > l125_offpath.log 2>&1
  <python> -u l125_beam_probe.py ab --w 2 --k 8 > l125_ab_w2k8.log 2>&1
"""
import argparse
import concurrent.futures
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# Shipped defaults only: a stray knob in this shell must not contaminate a gate.
for _k in sorted(k for k in os.environ if k.startswith("ICCAD_")):
    del os.environ[_k]

from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from optimizer_claude import _serialize_input, _parse_output        # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402
import optimizer_constructive as oc                                 # noqa: E402

WORKERS = 11
EXE_SHIP = _DIR / "constructive.exe"
EXE_BEAM = _DIR / "constructive_l125.exe"

# A spread of shipped profiles rather than one: the beam changes the packing
# order's downstream, so a recipe that already reshapes items (FREE_*) and one
# that does not can disagree about whether branching helps.
RECIPE_IDX = [int(x) for x in os.environ.get(
    "L125_RECIPES", "0,2,6,22,25,26").split(",")]
_SHIPPED = list(oc._PROFILES[:oc._M55_BASE_LEN])
RECIPES = [_SHIPPED[i] for i in RECIPE_IDX]

print("[l125] loading dataset ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()

CASES = []
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _at, _b2b, _p2b, _pins, _cons = _s["input"]
    _n = int((_at != -1).sum().item())
    _base, _tp = _ev._extract_baseline(_idx, _s["label"], _b2b, _p2b, _pins, _n)
    _otp = build_opt_target_pos(_tp, _cons, _n)
    CASES.append(dict(idx=_idx, n=_n, w=math.exp(_n / 12.0), base=_base, tp=_tp,
                      otp=_otp, at=_at, b2b=_b2b, p2b=_p2b, pins=_pins,
                      cons=_cons))

_CHILD_BASE = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}


def _overlay(n):
    """The shipping per-case overlay in the wrapper's precedence order.

    A gate whose input does not match the deployed form reports a fake number —
    the `gate-inputs-must-match-deployment` lesson. Note `otp` above carries the
    preplaced/fixed placements for the same reason: target_positions=None sinks
    every case to 10.0 and empties every anchored antecedent.
    """
    ov = dict(oc._band_env(n))
    ov.update(oc._m71_env())
    return ov


def _run(job):
    ci, ri, beam = job
    c = CASES[ci]
    env = dict(_CHILD_BASE)
    env.update(RECIPES[ri])
    env.update(_overlay(c["n"]))
    exe = EXE_SHIP if beam is None else EXE_BEAM
    if beam is not None:
        env.update(beam)
    txt = _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"], c["pins"],
                           c["cons"], c["otp"], gnn_hint=None)
    t0 = time.perf_counter()
    out = subprocess.run([exe], input=txt, capture_output=True, text=True,
                         env=env).stdout
    return job, _parse_output(out, c["n"]), time.perf_counter() - t0


def _map(jobs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(_run, jobs))


def _cost(c, pos):
    m = evaluate_solution({"positions": pos, "runtime": 1.0}, c["base"],
                          c["cons"][:c["n"]], c["b2b"], c["p2b"], c["pins"],
                          c["at"][:c["n"]], target_positions=c["tp"][:c["n"]],
                          median_runtime=1.0)
    return float(m.cost), bool(m.is_feasible)


def _weighted(per_case):
    num = sum(CASES[ci]["w"] * v for ci, v in per_case.items())
    den = sum(CASES[ci]["w"] for ci in per_case)
    return num / den


# ── modes ───────────────────────────────────────────────────────────────────
def mode_offpath(a):
    if not EXE_BEAM.exists():
        sys.exit("constructive_l125.exe missing")
    jobs = [(ci, ri, None) for ci in range(100) for ri in range(len(RECIPES))]
    ship = {(ci, ri): pos for (ci, ri, _), pos, _dt in _map(jobs)}
    ok = True
    for name, beam in (("ICCAD_BEAM unset", {}),
                       ("ICCAD_BEAM=1 ICCAD_BEAM_W=1", {"ICCAD_BEAM": "1",
                                                        "ICCAD_BEAM_W": "1"})):
        js = [(ci, ri, beam) for ci in range(100) for ri in range(len(RECIPES))]
        bad = 0
        for (ci, ri, _), pos, _dt in _map(js):
            if pos != ship[(ci, ri)]:
                bad += 1
                if bad <= 5:
                    print(f"  MISMATCH case {ci} recipe {RECIPE_IDX[ri]}")
        print(f"  {len(js) - bad}/{len(js)} bit-identical   [{name}]")
        ok = ok and bad == 0
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _ab(w, k, sel, cases, tag):
    beam = {"ICCAD_BEAM": "1", "ICCAD_BEAM_W": str(w), "ICCAD_BEAM_K": str(k)}
    if sel is not None:
        beam["ICCAD_BEAM_SEL"] = str(sel)
    jobs_off = [(ci, ri, None) for ci in cases for ri in range(len(RECIPES))]
    jobs_on = [(ci, ri, beam) for ci in cases for ri in range(len(RECIPES))]
    off = {(ci, ri): (pos, dt) for (ci, ri, _), pos, dt in _map(jobs_off)}
    on = {(ci, ri): (pos, dt) for (ci, ri, _), pos, dt in _map(jobs_on)}

    changed = 0
    ratios = []
    better = worse = 0
    # per case: best true cost over the recipes, for the three deployment forms
    #   off   today
    #   on    REPLACE  (every recipe beamed) -- the L123 form
    #   uni   TWIN     (both available, arbitrated per case) -- the L124 form
    per_off, per_on, per_uni = {}, {}, {}
    on_wins = 0
    for ci in cases:
        c = CASES[ci]
        bo = bn = float("inf")
        for ri in range(len(RECIPES)):
            po, dto = off[(ci, ri)]
            pn, dtn = on[(ci, ri)]
            if po != pn:
                changed += 1
            if dto > 0:
                ratios.append(dtn / dto)
            co, fo = _cost(c, po)
            cn, fn = _cost(c, pn)
            if not fo:
                co = 10.0
            if not fn:
                cn = 10.0
            if cn < co - 1e-12:
                better += 1
            elif cn > co + 1e-12:
                worse += 1
            bo, bn = min(bo, co), min(bn, cn)
        per_off[ci], per_on[ci] = bo, bn
        per_uni[ci] = min(bo, bn)
        if bn < bo - 1e-12:
            on_wins += 1

    tot = len(cases) * len(RECIPES)
    wo, wn, wu = _weighted(per_off), _weighted(per_on), _weighted(per_uni)
    print(f"\n=== {tag} ===")
    print(f"  changed (case,recipe): {changed}/{tot}")
    print(f"  solo true cost:        {better} better / {worse} worse "
          f"/ {tot - better - worse} equal")
    if ratios:
        print(f"  dt ratio ON/OFF:       p50 {statistics.median(ratios):.2f}  "
              f"p90 {sorted(ratios)[int(0.9 * (len(ratios) - 1))]:.2f}  "
              f"max {max(ratios):.2f}")
    # The multiplier is per (case,profile), not a scalar: l125_beam_price.py
    # prices a scalar, so dump the real distribution for it to be checked against.
    with open(_DIR / f"l125_dt_{tag.replace(' ', '_').replace('=', '')}.csv",
              "w") as f:
        f.write("case,n,recipe,dt_off,dt_on,ratio\n")
        for ci in cases:
            for ri in range(len(RECIPES)):
                dto, dtn = off[(ci, ri)][1], on[(ci, ri)][1]
                f.write(f"{ci},{CASES[ci]['n']},{RECIPE_IDX[ri]},"
                        f"{dto:.6f},{dtn:.6f},{dtn / dto if dto else 0:.4f}\n")
    print(f"  weighted, best over {len(RECIPES)} recipes by TRUE cost:")
    print(f"      OFF (today)     {wo:.6f}")
    print(f"      ON  (replace)   {wn:.6f}   {100 * (1 - wn / wo):+.4f}%")
    print(f"      TWIN (oracle)   {wu:.6f}   {100 * (1 - wu / wo):+.4f}%"
          f"   ON side wins {on_wins}/{len(cases)} cases")
    return wo, wn, wu


def mode_ab(a):
    cases = [c["idx"] for c in CASES if c["n"] >= a.nmin]
    if a.limit:
        cases = cases[:a.limit]
    print(f"[cfg] {len(cases)} cases (n>={a.nmin}) x {len(RECIPES)} recipes "
          f"{RECIPE_IDX}")
    _ab(a.w, a.k, a.sel, cases, f"W={a.w} K={a.k} SEL={a.sel}")
    return 0


def mode_sweep(a):
    cases = [c["idx"] for c in CASES if c["n"] >= a.nmin]
    if a.limit:
        cases = cases[:a.limit]
    print(f"[cfg] {len(cases)} cases (n>={a.nmin}) x {len(RECIPES)} recipes "
          f"{RECIPE_IDX}")
    for w, k in [(2, 2), (2, 8), (2, 24), (2, 999), (3, 8)]:
        _ab(w, k, a.sel, cases, f"W={w} K={k}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offpath", "ab", "sweep"])
    ap.add_argument("--w", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sel", type=int, default=None)
    ap.add_argument("--nmin", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    return {"offpath": mode_offpath, "ab": mode_ab, "sweep": mode_sweep}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
