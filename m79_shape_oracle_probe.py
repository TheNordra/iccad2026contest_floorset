"""M79 Gate 0 — what is a PERFECT per-block SHAPE worth inside our portfolio?

OFFLINE PROBE — never shipped. Uses fp_sol as an ORACLE INPUT only (same class as
M26 oracle-perm and M68 ML-seed); no model is ever trained on it.

WHY. M77 built the yardstick for an external (ML) candidate but no candidate ever
arrived. The one asymmetry that keeps the axis open is that ADDING a candidate has
no quality downside (the shapely proxy is oracle-perfect on heterogeneous pools —
M76 full-union, M77 efficiency 100.0%), so an ML only has to win SOMETIMES.

Which axis could it win on? Three perfect-information probes already bound it:

    perfect ORDER            +0.005%   M26 oracle-perm
    perfect POSITION seed    +0.001%   M68 ML-seed
    perfect SHAPE            never measured        <- this file
    fp_sol verbatim          ~14%      (1.293461 -> 1.1079)

Shape is the only one left, and it is where the classical line earned the most
(M29-M37's six free-aspect sub-axes: 1.3862 -> 1.3269). Supporting reading taken
before any placer ran: over the 6110 movable soft blocks of the in-set 100, the
label's aspect w/h is p05 0.400 / p50 1.067 / p95 2.500, sd(log)=0.530, and only
34% sit in the near-square band [0.8,1.25] — while our default for an interior
soft block is SOFT_ASPECT=1.0, an exact square. Stratifying by boundary code
barely moves sd(log) (0.44-0.55 per class), i.e. the ONE shape feature we
currently exploit (LR 2.50 / TB 0.40) explains almost none of that variance.

MODES
  coverage   how much of each case the zero-compile scout can reach (no solving)
  calib      G0-D: fp_sol verbatim as the 42nd candidate -> the absolute ceiling
  scout      G0-A pre-probe, NO C++ CHANGE. constructive.cpp:1585 already turns an
             is_fixed block's (tw,th) into dims[], and tw/th come from stdin
             (:1924). So marking a movable block fx=1 in the C++ INPUT ONLY (the
             evaluator still sees the true constraints) prescribes its shape.
             is_fixed has side effects — MIB master (:235), "cluster is mixed"
             (:1510), and the free-aspect gates (:812/:867/:1595) — so the scout
             is restricted to cluster==0 and mib==0 blocks and is a LOWER BOUND.
  oracle     G0-A proper. Needs constructive_m79.exe's ICCAD_DIMS_FILE, which sets
             dims[] without touching is_fixed and locks the block against every
             reshape path. Covers clustered and MIB blocks too.

Two numbers come out of scout/oracle, and they are different questions:
  ORACLE    per-case best recipe chosen by TRUE cost   -> the ceiling
  PROXY     per-case best recipe chosen by our own baseline-free proxy
            -> what a deployable single candidate would actually carry

Both are written as official results jsons; the verdict comes from
    m77_ml_candidate_probe.py score <json> --cores 48 --dt 0
with the pre-registered Gate-0 bar of +1.0% in-set portfolio delta (3x headroom
over the OOS NET bar of 0.30%, because an upper bound has to survive realization
loss, the in-set->OOS transfer, and dRF@48c).

SHAPES ARE PRESCRIBED AS ASPECT RATIOS, NOT RAW RECTANGLES: (w,h) is re-derived as
w=sqrt(area*r), h=area/w with r the label's w/h. Area then hits area_target
exactly (soft blocks only tolerate 1%), and it makes the quantity under test the
one an ML would actually emit — one scalar per block.

Run (PowerShell):
  <python> m79_shape_oracle_probe.py coverage
  <python> m79_shape_oracle_probe.py calib
  <python> -u m79_shape_oracle_probe.py scout  > m79_scout.txt  2>&1
  <python> -u m79_shape_oracle_probe.py oracle > m79_oracle.txt 2>&1
"""
import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# Shipped defaults only (same discipline as m77_ml_candidate_probe.py:55) — a
# stray knob in this shell must not contaminate an oracle measurement.
for _k in sorted(k for k in os.environ if k.startswith("ICCAD_")):
    del os.environ[_k]

import torch                                                        # noqa: E402
from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from optimizer_claude import _serialize_input, _parse_output        # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402
import optimizer_constructive as oc                                 # noqa: E402

RH = 1.4
WORKERS = 11                       # leave a core for this process
EXE_SHIP = _DIR / "constructive.exe"
EXE_M79 = _DIR / "constructive_m79.exe"
CACHE = _DIR / "m79_cache.pkl"
SCRATCH = _DIR / "m79_dims"        # per-case dims files for the oracle mode

# Recipes. With shapes prescribed, every aspect knob is a no-op on a locked block,
# so the set has to vary what still matters: pack ORDER (BFS/PIN/GM), FRAME scales,
# WIRE weight, and how clusters/anchored members are handled. Indices into the
# SHIPPED prefix of _PROFILES (see profile_audit.py's listing).
RECIPE_IDX = [int(x) for x in os.environ.get(
    "M79_RECIPES", "0,2,6,22,25,26").split(",")]


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


_SHIPPED = list(oc._PROFILES[:oc._M55_BASE_LEN])
RECIPES = [_SHIPPED[i] for i in RECIPE_IDX]

print("[m79] loading dataset ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()

CASES = []
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _at, _b2b, _p2b, _pins, _cons = _s["input"]
    _n = int((_at != -1).sum().item())
    _base, _tp = _ev._extract_baseline(_idx, _s["label"], _b2b, _p2b, _pins, _n)
    _otp = build_opt_target_pos(_tp, _cons, _n)
    _sumA = sum(max(0.0, float(_at[i])) for i in range(_n))
    CASES.append(dict(idx=_idx, n=_n, A_hat=1.035 * max(_sumA, 1e-9),
                      w=math.exp(_n / 12.0), base=_base, tp=_tp, otp=_otp,
                      at=_at, b2b=_b2b, p2b=_p2b, pins=_pins, cons=_cons))
TOTW = sum(c["w"] for c in CASES)


# ── which blocks may have their shape prescribed ────────────────────────────
def _movable(c, i):
    return int(c["cons"][i, 0]) == 0 and int(c["cons"][i, 1]) == 0 \
        and float(c["at"][i]) > 0


def targets(c, mode):
    """Block ids whose shape this mode can prescribe.

    scout  -> cluster==0 and mib==0 only. is_fixed is the vehicle, and it also
              decides the MIB master and whether a cluster counts as mixed, so
              touching a clustered/MIB block would swap out the M71 mechanism
              instead of only changing a shape.
    oracle -> every movable soft block (ICCAD_DIMS_FILE writes dims[] directly).
    """
    out = []
    for i in range(c["n"]):
        if not _movable(c, i):
            continue
        if mode == "scout" and (int(c["cons"][i, 2]) != 0
                                or int(c["cons"][i, 3]) != 0):
            continue
        out.append(i)
    return out


def label_dims(c, i):
    """(w,h) carrying the label's ASPECT at exactly area_target's area.

    Measured on the in-set 100: the label's w*h equals area_target to 0 ulp on all
    6110 movable soft blocks, and all 100 movable MIB groups share one shape
    exactly. So the verbatim rectangle already IS the aspect-at-exact-area answer,
    and returning it verbatim keeps MIB shape equality bit-exact — re-deriving it
    through sqrt() would inject an ulp and could manufacture a V_mib. The rescale
    branch is the honest fallback for a corpus where that does not hold."""
    _, _, lw, lh = c["tp"][i]
    lw, lh = float(lw), float(lh)
    A = float(c["at"][i])
    if lw <= 0 or lh <= 0 or A <= 0:
        s = math.sqrt(max(A, 1e-12))
        return s, s
    if abs(lw * lh - A) <= 1e-12 * A:
        return lw, lh
    w = math.sqrt(A * (lw / lh))
    return w, A / w


# ── serialization for each mode ─────────────────────────────────────────────
def _input_text(c, mode):
    if mode == "plain":
        return _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"], c["pins"],
                                c["cons"], c["otp"], gnn_hint=None)
    if mode == "scout":
        cons = c["cons"].clone()
        otp = c["otp"].clone()
        for i in targets(c, "scout"):
            w, h = label_dims(c, i)
            cons[i, 0] = 1                      # is_fixed -> dims[] = (tw,th)
            otp[i, 2], otp[i, 3] = w, h
        return _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"], c["pins"],
                                cons, otp, gnn_hint=None)
    raise ValueError(mode)


def _dims_file(c):
    SCRATCH.mkdir(exist_ok=True)
    p = SCRATCH / f"case{c['idx']:03d}.txt"
    with open(p, "w") as f:
        for i in targets(c, "oracle"):
            w, h = label_dims(c, i)
            f.write(f"{i} {w:.17g} {h:.17g}\n")
    return p


# ── runner ──────────────────────────────────────────────────────────────────
_CHILD_BASE = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}


def _overlay(n):
    """The shipping per-case overlay, in the wrapper's precedence order
    (optimizer_constructive.py: profile dict, then band, then M71). A gate whose
    input does not match the deployed form reports a fake number — that is what
    bit M75 (see the `gate-inputs-must-match-deployment` memory)."""
    ov = dict(oc._band_env(n))
    ov.update(oc._m71_env())
    return ov


def _run(job):
    ci, ri, mode = job
    c = CASES[ci]
    env = dict(_CHILD_BASE)
    env.update(RECIPES[ri])
    env.update(_overlay(c["n"]))
    if mode == "oracle":
        env["ICCAD_DIMS_FILE"] = str(SCRATCH / f"case{ci:03d}.txt")
        exe, txt = str(EXE_M79), _input_text(c, "plain")
    elif mode == "scout":
        exe, txt = str(EXE_SHIP), _input_text(c, "scout")
    else:                                                    # "plain" control
        exe, txt = str(EXE_SHIP), _input_text(c, "plain")
    t0 = time.perf_counter()
    out = subprocess.run([exe], input=txt, capture_output=True, text=True,
                         env=env).stdout
    dt = time.perf_counter() - t0
    return job, _parse_output(out, c["n"]), dt


def _true(c, pos):
    m = evaluate_solution({"positions": pos, "runtime": 1.0}, c["base"],
                          c["cons"][:c["n"]], c["b2b"], c["p2b"], c["pins"],
                          c["at"][:c["n"]], target_positions=c["tp"][:c["n"]],
                          median_runtime=1.0)
    return float(m.cost), bool(m.is_feasible)


def _proxy(c, pos):
    m = oc._proxy_metrics(pos, c["at"], c["b2b"], c["p2b"], c["pins"],
                          c["cons"], c["n"])
    return m["area"], m["hpwl"], m["vrel"]


def _sig(mode):
    exe = EXE_M79 if mode == "oracle" else EXE_SHIP
    return hashlib.md5(repr((
        "v2", mode, repr(RECIPES), _md5(exe),
        repr(sorted(oc._m71_env().items())),
        repr(sorted(oc._M49_REFINE_BAND)),
        repr(sorted(oc._M50_REFINE_LOWCORE)), oc._M45_CORES_MAX,
    )).encode()).hexdigest()


def _slot(mode):
    """Namespace the cache by mode AND signature so changing RECIPE_IDX adds a
    slot instead of discarding the previous measurement."""
    return f"{mode}-{_sig(mode)[:8]}"


def _load_cache(mode):
    sig, slot = _sig(mode), _slot(mode)
    if CACHE.exists():
        try:
            c0 = pickle.load(open(CACHE, "rb"))
            if c0.get(slot, {}).get("sig") == sig:
                return c0, c0[slot]["data"]
            return c0, {}
        except Exception:
            pass
    return {}, {}


def _save_cache(all_c, mode, data):
    all_c[_slot(mode)] = {"sig": _sig(mode), "data": data}
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(all_c, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE)


def _dump(path, name, rows, total):
    Path(path).write_text(json.dumps({
        "submission_name": name, "total_score": total, "test_results": rows,
    }), encoding="utf-8")
    print(f"  wrote {Path(path).name}   total_score {total:.9f}")


# ── modes ───────────────────────────────────────────────────────────────────
def mode_coverage(_):
    print("=" * 78)
    print("M79 G0 coverage — how much of each case a prescribed shape can reach")
    print("=" * 78)
    tot = {"scout": [0.0, 0], "oracle": [0.0, 0]}
    allA, allN = 0.0, 0
    for c in CASES:
        A = sum(float(c["at"][i]) for i in range(c["n"])
                if float(c["at"][i]) > 0)
        allA += c["w"] * A
        allN += c["n"]
        for m in ("scout", "oracle"):
            t = targets(c, m)
            tot[m][0] += c["w"] * sum(float(c["at"][i]) for i in t)
            tot[m][1] += len(t)
    print(f"  blocks total                {allN}")
    for m in ("scout", "oracle"):
        print(f"  {m:<7} prescribable blocks {tot[m][1]:>6}  "
              f"({100 * tot[m][1] / allN:.1f}%)   "
              f"weighted area share {100 * tot[m][0] / allA:.1f}%")
    print("\n  per-case scout coverage (weight-sorted, worst 10 shown):")
    rows = sorted(((len(targets(c, "scout")) / max(1, len(targets(c, "oracle"))),
                    c["idx"], c["n"]) for c in CASES))[:10]
    for f, ci, n in rows:
        print(f"    case {ci:>3} n={n:>4}  scout reaches {100 * f:5.1f}% "
              f"of the oracle's blocks")
    return 0


def mode_calib(_):
    """fp_sol verbatim as the 42nd candidate: the absolute ceiling, and a scale
    check on m77's arithmetic. No placer involved."""
    print("=" * 78)
    print("M79 G0-D — fp_sol verbatim as an external candidate (calibration)")
    print("=" * 78)
    rows, tot, nfeas = [], 0.0, 0
    for c in CASES:
        pos = [tuple(float(v) for v in c["tp"][i]) for i in range(c["n"])]
        cost, feas = _true(c, pos)
        nfeas += feas
        tot += c["w"] * cost
        rows.append(dict(test_id=c["idx"], block_count=c["n"], positions=pos,
                         cost=cost, is_feasible=feas, runtime_seconds=0.0))
    print(f"  feasible {nfeas}/100")
    _dump(_DIR / "m79_fpsol_verbatim.json", "fp_sol verbatim", rows, tot / TOTW)
    print("\n  next: m77_ml_candidate_probe.py score m79_fpsol_verbatim.json "
          "--cores 48 --dt 0")
    return 0


def _sweep(mode):
    """Run RECIPES x 100 cases under `mode`; return {(ci,ri): (pos, dt)}."""
    if mode == "oracle":
        if not EXE_M79.exists():
            sys.exit("constructive_m79.exe missing -> build it first "
                     "(PowerShell: g++ -O3 -std=c++17 -o constructive_m79.exe "
                     "constructive_m79.cpp)")
        for c in CASES:
            _dims_file(c)
    all_c, data = _load_cache(mode)
    jobs = [(ci, ri, mode) for ci in range(100) for ri in range(len(RECIPES))
            if (ci, ri) not in data]
    print(f"  {len(jobs)} of {100 * len(RECIPES)} runs to do "
          f"({100 * len(RECIPES) - len(jobs)} cached)", flush=True)
    if jobs:
        t0 = time.perf_counter()
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for (ci, ri, _), pos, dt in ex.map(_run, jobs):
                data[(ci, ri)] = (pos, dt)
                done += 1
                if done % 100 == 0:
                    print(f"    {done}/{len(jobs)}  "
                          f"{time.perf_counter() - t0:.0f}s", flush=True)
        _save_cache(all_c, mode, data)
        print(f"  sweep done in {time.perf_counter() - t0:.0f}s", flush=True)
    return data


def _select_and_dump(mode, data):
    """Two candidates per case: best-by-true-cost (ceiling) and best-by-proxy
    (what a single deployable candidate carries)."""
    o_rows, p_rows = [], []
    o_tot = p_tot = 0.0
    o_feas = p_feas = 0
    same = 0
    for c in CASES:
        ci = c["idx"]
        cand = [(ri, data[(ci, ri)][0], data[(ci, ri)][1])
                for ri in range(len(RECIPES))]
        costs = {ri: _true(c, pos) for ri, pos, _ in cand}
        pms = {ri: _proxy(c, pos) for ri, pos, _ in cand}
        hmin = min(v[1] for v in pms.values()) or 1.0

        def pscore(ri):
            a, h, v = pms[ri]
            return (a / c["A_hat"] + RH * h / hmin) * math.exp(2 * v)

        ro = min(costs, key=lambda r: costs[r][0])
        rp = min(pms, key=pscore)
        same += (ro == rp)
        for ri, rows, tot, feas in ((ro, o_rows, "o", "o"), (rp, p_rows, "p", "p")):
            pos, dt = data[(ci, ri)][0], data[(ci, ri)][1]
            cost, ok = costs[ri]
            rows.append(dict(test_id=ci, block_count=c["n"],
                             positions=[list(map(float, q)) for q in pos],
                             cost=cost, is_feasible=ok, runtime_seconds=dt))
        o_tot += c["w"] * costs[ro][0]
        p_tot += c["w"] * costs[rp][0]
        o_feas += costs[ro][1]
        p_feas += costs[rp][1]
    print(f"  recipe agreement (proxy picked the true best) {same}/100")
    print(f"  feasible: oracle-pick {o_feas}/100   proxy-pick {p_feas}/100")
    _dump(_DIR / f"m79_shape_{mode}_oraclepick.json",
          f"M79 {mode} shape (recipe by true cost)", o_rows, o_tot / TOTW)
    _dump(_DIR / f"m79_shape_{mode}_proxypick.json",
          f"M79 {mode} shape (recipe by proxy)", p_rows, p_tot / TOTW)
    print("\n  next:")
    for tag in ("oraclepick", "proxypick"):
        print(f"    m77_ml_candidate_probe.py score m79_shape_{mode}_{tag}.json "
              f"--cores 48 --dt 0")


def mode_scout(_):
    print("=" * 78)
    print("M79 G0-A pre-probe (scout) — prescribed shapes via the is_fixed path")
    print("  LOWER BOUND: cluster==0 and mib==0 blocks only, no C++ change")
    print("=" * 78)
    _select_and_dump("scout", _sweep("scout"))
    return 0


def mode_oracle(_):
    print("=" * 78)
    print("M79 G0-A — perfect per-block SHAPE, our placer picks everything else")
    print("=" * 78)
    _select_and_dump("oracle", _sweep("oracle"))
    return 0


def mode_control(_):
    """The control G0-A needs: the SAME 6 recipes with NO shape prescribed. m77
    adds a candidate to the full 41-profile pool, so part of any gain could just be
    'a re-selected 6-recipe subset' rather than the shape information. Subtracting
    this isolates what the shapes actually bought."""
    print("=" * 78)
    print("M79 G0-A control — same 6 recipes, shapes NOT prescribed")
    print("=" * 78)
    _select_and_dump("plain", _sweep("plain"))
    return 0


def mode_offpath(_):
    """constructive_m79.exe with ICCAD_DIMS_FILE unset must be bit-identical to
    constructive.exe (the M78 convention for a probe binary)."""
    if not EXE_M79.exists():
        sys.exit("constructive_m79.exe missing")
    print("=" * 78)
    print("M79 off-path bit-identity: constructive_m79.exe (no DIMS_FILE) "
          "vs constructive.exe")
    print("=" * 78)
    jobs = [(ci, ri, "plain") for ci in range(100)
            for ri in range(len(RECIPES))]
    ship = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for (ci, ri, _), pos, _dt in ex.map(_run, jobs):
            ship[(ci, ri)] = pos
    global EXE_SHIP
    keep, EXE_SHIP = EXE_SHIP, EXE_M79
    try:
        bad = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for (ci, ri, _), pos, _dt in ex.map(_run, jobs):
                if pos != ship[(ci, ri)]:
                    bad += 1
                    if bad <= 5:
                        print(f"  MISMATCH case {ci} recipe {ri}")
    finally:
        EXE_SHIP = keep
    print(f"\n  {len(jobs) - bad}/{len(jobs)} (case,recipe) bit-identical")
    print(f"  RESULT: {'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


def main():
    modes = {"coverage": mode_coverage, "calib": mode_calib, "scout": mode_scout,
             "oracle": mode_oracle, "control": mode_control,
             "offpath": mode_offpath}
    if len(sys.argv) < 2 or sys.argv[1] not in modes:
        sys.exit(f"usage: m79_shape_oracle_probe.py {'|'.join(modes)}")
    return modes[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
