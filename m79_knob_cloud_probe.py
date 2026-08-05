"""M79 Gate 0-B — what is a PER-CASE choice of env-knob vector worth?

OFFLINE PROBE — never shipped. Uses no labels except to score (evaluate_solution).

THE QUESTION. Our four dead ML insertion points are generation (M52), selector
(M56), seed (M68) and refinement (LP family). A fifth was never measured: a model
that, per case, emits a PROFILE — a point in the ~15-dim env-knob space the 41
shipped profiles live in — which our own C++ then decodes.

It dodges both of the known killers:
  * M52 died on a zero fault-tolerance band (one near-miss token -> wR 1.232).
    Here the decoder is the shipped placer, so ANY knob vector yields a legal
    layout; the tolerance band is the whole space.
  * M56 died because per-case pool PRUNING loses the winner out of sample
    (strict-breaks 30/200). Here the candidate is ADDED, and the proxy is
    oracle-perfect on heterogeneous pools (M76 full-union, M77 efficiency 100%),
    so a wrong prediction costs runtime, never quality.

So the only thing that can kill it is the CEILING: is there, per case, a knob
vector outside the 41 that beats the portfolio? This measures exactly that, with
no model involved — R random vectors per case, per-case min, fed to m77 as one
candidate.

Prior is RED-leaning: M30/M31 swept single new fixed profiles to saturation at
<=0.063%, and M53 L2's stochastic best-of-N union oracle was +0.2377%. But those
measured a FIXED new profile or per-decision noise; nobody measured a per-case
free choice of knob vector.

The proposal distribution is deliberately not uniform-wild: half the draws mutate
1-3 knobs of a randomly chosen shipped profile (the region where good recipes are
known to live), half are fresh draws from the per-knob priors. A loose proposal
would understate the ceiling.

ORDER_SWAP/ORDER_MOVE are excluded on purpose: they cost 5-12s/case (profile_audit
puts om16 at mean 9.71s / max 23.73s) so a candidate carrying them would set the
48-core wall by itself, and M41 already drops them on every case. Including them
would inflate a ceiling we could not spend.

Run (PowerShell):
  <python> -u m79_knob_cloud_probe.py run  [R]   > m79_knob_cloud.txt 2>&1
  <python>    m79_knob_cloud_probe.py report
then
  <python> m77_ml_candidate_probe.py score m79_knob_cloud_oraclepick.json \
           --cores 48 --dt <the candidate's own dt>
"""
import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

for _k in sorted(k for k in os.environ if k.startswith("ICCAD_")):
    del os.environ[_k]

from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from optimizer_claude import _serialize_input, _parse_output        # noqa: E402
from proxy_analysis import build_opt_target_pos                     # noqa: E402
import optimizer_constructive as oc                                 # noqa: E402

RH = 1.4
WORKERS = 11
SEED = 79
EXE = _DIR / "constructive.exe"
CACHE = _DIR / "m79_knob_cloud.pkl"

_RATIO_SETS = ["0.333,0.5,0.6667,1.0,1.5,2.0,3.0,4.0",
               "0.5,0.6667,1.0,1.5,2.0",
               "0.25,0.4,0.6667,1.0,1.5,2.5,4.0",
               "0.2,0.5,1.0,2.0,5.0"]
_SCALE_SETS = [None, "1.00,1.05,1.10,1.20", "1.00,1.10,1.25,1.45",
               "1.00,1.02,1.05,1.10"]
_ASPECT_SETS = [None, "1.0,0.75,1.35,2.2", "0.67,0.5,0.4,0.33", "1.0,1.6,2.6"]


def _lu(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _fresh(rng):
    """A draw from the per-knob priors, spanning what _PROFILES actually varies
    plus SOFT_ASPECT, which no shipped profile ever moves."""
    p = {}
    if rng.random() < 0.75:
        p["ICCAD_FREE_ASPECT"] = "1"
    p["ICCAD_WIRE_MULT"] = f"{_lu(rng, 0.5, 4.0):.4g}"
    if rng.random() < 0.35:
        p["ICCAD_ANCHOR_W"] = f"{rng.uniform(0.0, 0.30):.4g}"
    if rng.random() < 0.30:
        p["ICCAD_BP_WEIGHT"] = f"{_lu(rng, 3000, 300000):.6g}"
    if rng.random() < 0.55:
        p["ICCAD_LR_ASPECT"] = f"{_lu(rng, 1.0, 7.0):.4g}"
    if rng.random() < 0.55:
        p["ICCAD_TB_ASPECT"] = f"{_lu(rng, 0.14, 1.0):.4g}"
    if rng.random() < 0.35:
        p["ICCAD_SOFT_ASPECT"] = f"{_lu(rng, 0.5, 2.0):.4g}"
    if rng.random() < 0.60:
        p["ICCAD_CLUSTER_ASPECT"] = f"{_lu(rng, 0.25, 4.0):.4g}"
    if rng.random() < 0.30:
        p["ICCAD_MIB_ASPECT"] = f"{_lu(rng, 0.2, 6.0):.4g}"
    if rng.random() < 0.55:
        p["ICCAD_FREE_CLUSTER"] = "1"
        p["ICCAD_FREE_CLUSTER_RATIOS"] = rng.choice(_RATIO_SETS)
    if rng.random() < 0.45:
        p["ICCAD_FREE_ANCHORED"] = "1"
        p["ICCAD_FREE_ANCHORED_RATIOS"] = rng.choice(_RATIO_SETS)
        if rng.random() < 0.5:
            p["ICCAD_FREE_ANCHORED_BND"] = "1"
    for k, pr in (("ICCAD_WIRE_BFS", .6), ("ICCAD_BFS_PIN", .45),
                  ("ICCAD_WIRE_TIEBREAK", .55), ("ICCAD_GUIDE_MED", .35)):
        if rng.random() < pr:
            p[k] = "1"
    s = rng.choice(_SCALE_SETS)
    if s:
        p["ICCAD_FRAME_SCALES"] = s
    a = rng.choice(_ASPECT_SETS)
    if a:
        p["ICCAD_FRAME_ASPECTS"] = a
    return p


_NUMERIC = {"ICCAD_WIRE_MULT": (0.5, 4.0), "ICCAD_ANCHOR_W": (0.01, 0.30),
            "ICCAD_BP_WEIGHT": (3000, 300000), "ICCAD_LR_ASPECT": (1.0, 7.0),
            "ICCAD_TB_ASPECT": (0.14, 1.0), "ICCAD_SOFT_ASPECT": (0.5, 2.0),
            "ICCAD_CLUSTER_ASPECT": (0.25, 4.0), "ICCAD_MIB_ASPECT": (0.2, 6.0)}


def _mutate(rng, base):
    """1-3 knob edits to a shipped profile — the local half of the proposal."""
    p = dict(base)
    for _ in range(rng.randint(1, 3)):
        which = rng.random()
        if which < 0.55:
            k = rng.choice(list(_NUMERIC))
            lo, hi = _NUMERIC[k]
            if k in p:
                v = float(p[k]) * math.exp(rng.gauss(0, 0.35))
                p[k] = f"{min(max(v, lo), hi):.6g}"
            else:
                p[k] = f"{_lu(rng, lo, hi):.6g}"
        elif which < 0.80:
            k = rng.choice(["ICCAD_WIRE_BFS", "ICCAD_BFS_PIN",
                            "ICCAD_WIRE_TIEBREAK", "ICCAD_GUIDE_MED",
                            "ICCAD_FREE_ASPECT", "ICCAD_FREE_CLUSTER",
                            "ICCAD_FREE_ANCHORED", "ICCAD_FREE_ANCHORED_BND"])
            if k in p:
                del p[k]
            else:
                p[k] = "1"
                if k == "ICCAD_FREE_CLUSTER":
                    p["ICCAD_FREE_CLUSTER_RATIOS"] = rng.choice(_RATIO_SETS)
                if k == "ICCAD_FREE_ANCHORED":
                    p["ICCAD_FREE_ANCHORED_RATIOS"] = rng.choice(_RATIO_SETS)
        else:
            for key, sets in (("ICCAD_FRAME_SCALES", _SCALE_SETS),
                              ("ICCAD_FRAME_ASPECTS", _ASPECT_SETS),
                              ("ICCAD_FREE_CLUSTER_RATIOS", _RATIO_SETS)):
                if rng.random() < 0.34:
                    v = rng.choice(sets)
                    if v:
                        p[key] = v
                    else:
                        p.pop(key, None)
    for k in ("ICCAD_ORDER_SWAP", "ICCAD_ORDER_MOVE"):
        p.pop(k, None)                       # never: 5-12s/case, M41 drops them
    return p


def build_cloud(R):
    rng = random.Random(SEED)
    src = [p for p in oc._PROFILES[:oc._M55_BASE_LEN]
           if "ICCAD_ORDER_SWAP" not in p and "ICCAD_ORDER_MOVE" not in p]
    out, seen = [], set()
    while len(out) < R:
        p = _mutate(rng, rng.choice(src)) if rng.random() < 0.5 else _fresh(rng)
        k = repr(sorted(p.items()))
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


print("[m79b] loading dataset ...", flush=True)
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
                      w=math.exp(_n / 12.0), base=_base, tp=_tp,
                      txt=_serialize_input(_n, _at, _b2b, _p2b, _pins, _cons,
                                           _otp, gnn_hint=None),
                      at=_at, b2b=_b2b, p2b=_p2b, pins=_pins, cons=_cons))
TOTW = sum(c["w"] for c in CASES)

_CHILD_BASE = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}


def _overlay(n):
    ov = dict(oc._band_env(n))
    ov.update(oc._m71_env())
    return ov


CLOUD = []


def _run(job):
    ci, ki = job
    c = CASES[ci]
    env = dict(_CHILD_BASE)
    env.update(CLOUD[ki])
    env.update(_overlay(c["n"]))
    t0 = time.perf_counter()
    out = subprocess.run([str(EXE)], input=c["txt"], capture_output=True,
                         text=True, env=env).stdout
    dt = time.perf_counter() - t0
    return job, _parse_output(out, c["n"]), dt


def _true(c, pos):
    m = evaluate_solution({"positions": pos, "runtime": 1.0}, c["base"],
                          c["cons"][:c["n"]], c["b2b"], c["p2b"], c["pins"],
                          c["at"][:c["n"]], target_positions=c["tp"][:c["n"]],
                          median_runtime=1.0)
    return float(m.cost), bool(m.is_feasible)


def _pkey(p):
    return hashlib.md5(repr(sorted(p.items())).encode()).hexdigest()[:16]


def _proxy(c, pos):
    """The wrapper's selector metrics — shapely vrel, never the C++ union-find."""
    m = oc._proxy_metrics(pos, c["at"], c["b2b"], c["p2b"], c["pins"],
                          c["cons"], c["n"])
    return m["area"], m["hpwl"], m["vrel"]


def _sig():
    """Deliberately does NOT include the cloud: entries are keyed by
    (case, profile-hash), so growing R only pays for the new vectors — the same
    trick m77_oos_probe.py uses to make a third sample cheap. It DOES pin the
    exe md5 and the overlay constants (M74: the old repr(_PROFILES)-only key let
    a stale cache silently answer for a different binary)."""
    return hashlib.md5(repr((
        "v1", _md5(EXE),
        repr(sorted(oc._m71_env().items())),
        repr(sorted(oc._M49_REFINE_BAND)),
        repr(sorted(oc._M50_REFINE_LOWCORE)), oc._M45_CORES_MAX,
    )).encode()).hexdigest()


def _pool_at(n, cores=48):
    old = os.environ.get("ICCAD_ADAPTIVE_CORES")
    os.environ["ICCAD_ADAPTIVE_CORES"] = str(cores)
    try:
        return [i for i in oc._pool_indices(n) if i < oc._M55_BASE_LEN]
    finally:
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
        if old is not None:
            os.environ["ICCAD_ADAPTIVE_CORES"] = old


def mode_greedy(data, KEY, R, sig):
    """THE question G0-B's oracle cannot answer on its own: how much of the
    per-case gain survives if you are NOT allowed to choose per case?

    A fixed new profile is the classical move (M30's `profile_vs_portfolio`
    methodology, swept to saturation at <=0.063%). Per-case choice is what needs a
    model. Greedily adding the best fixed vectors from the same cloud separates
    the two: whatever the greedy curve reaches is free classical gain, and only
    the gap above it is a job for ML."""
    ship = pickle.load(open(_DIR / "audit_cache_ship.pkl", "rb"))["data"]
    pm = data.setdefault("pm", {})
    cost = data.setdefault("cost", {})

    def PM(ci, k):
        if (ci, k) not in pm:
            c = CASES[ci]
            pos = ship[(ci, k)][0] if isinstance(k, int) else data[(ci, k)][0]
            pm[(ci, k)] = _proxy(c, pos)
        return pm[(ci, k)]

    def CO(ci, k):
        if (ci, k) not in cost:
            c = CASES[ci]
            pos = ship[(ci, k)][0] if isinstance(k, int) else data[(ci, k)][0]
            cost[(ci, k)] = _true(c, pos)[0]
        return cost[(ci, k)]

    def sel(ci, pool):
        c = CASES[ci]
        hmin = min(PM(ci, k)[1] for k in pool) or 1.0
        return min(pool, key=lambda k: (PM(ci, k)[0] / c["A_hat"]
                                        + RH * PM(ci, k)[1] / hmin)
                   * math.exp(2 * PM(ci, k)[2]))

    pools = {c["idx"]: _pool_at(c["n"]) for c in CASES}
    print("  warming proxy/cost cache (16900 entries, minutes) ...", flush=True)
    for c in CASES:
        ci = c["idx"]
        for k in pools[ci]:
            PM(ci, k), CO(ci, k)
        for ki in range(R):
            PM(ci, KEY[ki]), CO(ci, KEY[ki])
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"sig": sig, "data": data}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE)

    def total(extra):
        return sum(CASES[ci]["w"] * CO(ci, sel(ci, pools[ci] + list(extra)))
                   for ci in range(100)) / TOTW

    t_b = total([])
    print(f"\n  incumbent portfolio @48c  {t_b:.9f}")

    solo = sorted(((1 - total([KEY[ki]]) / t_b) * 100, ki) for ki in range(R))
    print("\n  best SINGLE added vector (this is M30's `profile_vs_portfolio`, "
          "bar 0.05%):")
    for g, ki in sorted(solo, reverse=True)[:8]:
        print(f"    {g:+.3f}%   {json.dumps(CLOUD[ki], sort_keys=True)[:96]}")

    print("\n  greedy fixed-profile curve (no per-case choice allowed):")
    picked = []                                   # cloud indices, in greedy order
    for step in range(8):
        cur, ki = min(((total([KEY[j] for j in picked] + [KEY[i]]), i)
                       for i in range(R) if i not in picked),
                      key=lambda z: z[0])
        picked.append(ki)
        print(f"    +{step + 1} fixed profile(s): {cur:.9f}   "
              f"{(1 - cur / t_b) * 100:+.3f}%   (added #{ki})")
    print("\n  picked vectors:")
    for i in picked:
        print(f"    #{i:>3}  {json.dumps(CLOUD[i], sort_keys=True)}")
    return 0


def mode_loo(data, KEY, R, sig):
    """Gate-1 PREVIEW, free once `greedy` has warmed the cache.

    G0-B's +2% is an ORACLE: the winning vector is chosen with the label. M56 is
    the cautionary tale — its winner-only oracle was +3.87% and realized 0,
    because winner identity was case-idiosyncratic. So: is the winner predictable
    from label-free features at all? Three leave-one-out predictors, weakest
    first. Whatever the best of them scores is a LOWER bound on what a real model
    could do, and if they all land at ~0 this is M56 again."""
    pm, cost = data["pm"], data["cost"]
    ship = pickle.load(open(_DIR / "audit_cache_ship.pkl", "rb"))["data"]

    def PM(ci, k):
        if (ci, k) not in pm:
            pos = ship[(ci, k)][0] if isinstance(k, int) else data[(ci, k)][0]
            pm[(ci, k)] = _proxy(CASES[ci], pos)
        return pm[(ci, k)]

    def CO(ci, k):
        if (ci, k) not in cost:
            pos = ship[(ci, k)][0] if isinstance(k, int) else data[(ci, k)][0]
            cost[(ci, k)] = _true(CASES[ci], pos)[0]
        return cost[(ci, k)]

    def sel(ci, pool):
        c = CASES[ci]
        hmin = min(PM(ci, k)[1] for k in pool) or 1.0
        return min(pool, key=lambda k: (PM(ci, k)[0] / c["A_hat"]
                                        + RH * PM(ci, k)[1] / hmin)
                   * math.exp(2 * PM(ci, k)[2]))

    pools = {c["idx"]: _pool_at(c["n"]) for c in CASES}
    base = {ci: CO(ci, sel(ci, pools[ci])) for ci in range(100)}
    t_b = sum(CASES[ci]["w"] * base[ci] for ci in range(100)) / TOTW

    # What one added vector is WORTH on each case, after proxy arbitration. Not
    # its solo cost: M77's headline result is that solo score and portfolio value
    # are not monotonically related, so a predictor trained on mean solo cost
    # optimises the wrong thing (first cut of this probe did exactly that and
    # reported +0.000% for the global predictor).
    print("  precomputing per-vector portfolio effect (12800 selections) ...",
          flush=True)
    eff = [[CO(ci, sel(ci, pools[ci] + [KEY[ki]])) for ki in range(R)]
           for ci in range(100)]

    def feats(c):
        n = c["n"]
        cons, at = c["cons"], c["at"]
        pre = sum(1 for i in range(n) if int(cons[i, 1]) != 0)
        fix = sum(1 for i in range(n) if int(cons[i, 0]) != 0)
        clu = sum(1 for i in range(n) if int(cons[i, 3]) != 0)
        mib = sum(1 for i in range(n) if int(cons[i, 2]) != 0)
        bnd = sum(1 for i in range(n) if int(cons[i, 4]) != 0)
        ar = sorted(float(at[i]) for i in range(n) if float(at[i]) > 0)
        skew = ar[-1] / max(ar[len(ar) // 2], 1e-9) if ar else 1.0
        return [math.log(n), pre / n, fix / n, clu / n, mib / n, bnd / n,
                math.log(max(skew, 1e-9))]

    F = [feats(c) for c in CASES]
    mu = [sum(f[j] for f in F) / 100 for j in range(len(F[0]))]
    sd = [max(1e-9, (sum((f[j] - mu[j]) ** 2 for f in F) / 100) ** .5)
          for j in range(len(F[0]))]
    Z = [[(f[j] - mu[j]) / sd[j] for j in range(len(f))] for f in F]

    def band(n):
        return 0 if n <= 60 else (1 if n <= 100 else 2)

    def predict(ci, kind):
        tr = [j for j in range(100) if j != ci]
        if kind == "global":
            pool = tr
        elif kind == "band":
            pool = [j for j in tr if band(CASES[j]["n"]) == band(CASES[ci]["n"])]
        else:                                                       # knn(k=5)
            d = sorted(tr, key=lambda j: sum((a - b) ** 2
                                             for a, b in zip(Z[ci], Z[j])))
            pool = d[:5]
        return max(range(R), key=lambda ki: sum(
            CASES[j]["w"] * (base[j] - eff[j][ki]) for j in pool))

    print(f"\n  incumbent portfolio @48c  {t_b:.9f}")
    print("\n  leave-one-out predictors (portfolio delta of the PREDICTED "
          "vector as a 42nd candidate):")
    for kind in ("global", "band", "knn5"):
        tot, hit = 0.0, 0
        for ci in range(100):
            ki = predict(ci, kind)
            k = sel(ci, pools[ci] + [KEY[ki]])
            c = CO(ci, k)
            tot += CASES[ci]["w"] * c
            hit += c < base[ci] - 1e-12
        t = tot / TOTW
        print(f"    {kind:>7}  {t:.9f}   {(1 - t / t_b) * 100:+.3f}%   "
              f"wins on {hit}/100 cases")
    print("\n  (oracle over the same cloud was +2.025%; the gap is what a real "
          "model\n   would have to earn back with better features.)")

    # ---- 5-fold CV of the GREEDY FIXED set: does the classical half of the
    # finding survive being chosen on cases it is not scored on? Same discipline
    # as m55_dropset_cv.py, and the M76 warning applies: in-sample advantage has
    # transferred at ~5% before.
    print("\n" + "=" * 78)
    print("  5-fold CV of the greedy FIXED-profile set (picked on 80, scored "
          "on 20)")
    print("=" * 78)
    folds = [[ci for ci in range(100) if ci % 5 == f] for f in range(5)]
    KMAX = 8
    held = [0.0] * (KMAX + 1)
    denom = sum(CASES[ci]["w"] * base[ci] for ci in range(100))
    for f, te in enumerate(folds):
        tr = [ci for ci in range(100) if ci not in te]
        picked, cur = [], None
        for step in range(KMAX):
            best, bg = None, None
            for ki in range(R):
                if ki in picked:
                    continue
                ex = [KEY[j] for j in picked + [ki]]
                g = sum(CASES[ci]["w"] * (base[ci] - CO(ci, sel(ci, pools[ci] + ex)))
                        for ci in tr)
                if bg is None or g > bg:
                    best, bg = ki, g
            picked.append(best)
            ex = [KEY[j] for j in picked]
            held[step + 1] += sum(
                CASES[ci]["w"] * (base[ci] - CO(ci, sel(ci, pools[ci] + ex)))
                for ci in te)
        print(f"    fold {f}: picked {picked}", flush=True)
    print(f"\n  {'K':>3} {'held-out gain':>15}")
    for k in range(1, KMAX + 1):
        print(f"  {k:>3} {100 * held[k] / denom:>14.3f}%")
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    global CLOUD
    CLOUD = build_cloud(R)
    sig = _sig()
    data = {}
    if CACHE.exists():
        try:
            c0 = pickle.load(open(CACHE, "rb"))
            if c0.get("sig") == sig:
                data = c0["data"]
            else:
                print("  cache signature != current cloud/exe -> rebuilding")
        except Exception:
            pass

    print("=" * 78)
    print(f"M79 G0-B — per-case oracle over a {R}-point env-knob cloud")
    print("=" * 78)
    KEY = [_pkey(p) for p in CLOUD]
    if mode == "run":
        jobs = [(ci, ki) for ci in range(100) for ki in range(R)
                if (ci, KEY[ki]) not in data]
        print(f"  {len(jobs)} of {100 * R} runs to do "
              f"({100 * R - len(jobs)} cached)", flush=True)
        t0 = time.perf_counter()
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for (ci, ki), pos, dt in ex.map(_run, jobs):
                data[(ci, KEY[ki])] = (pos, dt)
                done += 1
                if done % 500 == 0:
                    el = time.perf_counter() - t0
                    print(f"    {done}/{len(jobs)}  {el:.0f}s  "
                          f"(eta {el / done * (len(jobs) - done):.0f}s)",
                          flush=True)
                    tmp = CACHE.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        pickle.dump({"sig": sig, "data": data}, f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp, CACHE)
        tmp = CACHE.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"sig": sig, "data": data}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, CACHE)
        print(f"  sweep done in {time.perf_counter() - t0:.0f}s", flush=True)

    missing = [(ci, ki) for ci in range(100) for ki in range(R)
               if (ci, KEY[ki]) not in data]
    if missing:
        sys.exit(f"{len(missing)} runs missing -> re-run `run`")

    if mode == "greedy":
        return mode_greedy(data, KEY, R, sig)
    if mode == "loo":
        return mode_loo(data, KEY, R, sig)

    o_rows, tot, feas, dts, hits = [], 0.0, 0, [], {}
    for c in CASES:
        ci = c["idx"]
        costs = {ki: _true(c, data[(ci, KEY[ki])][0]) for ki in range(R)}
        ko = min(costs, key=lambda k: costs[k][0])
        hits[ko] = hits.get(ko, 0) + 1
        pos, dt = data[(ci, KEY[ko])]
        tot += c["w"] * costs[ko][0]
        feas += costs[ko][1]
        dts.append(dt)
        o_rows.append(dict(test_id=ci, block_count=c["n"],
                           positions=[list(map(float, q)) for q in pos],
                           cost=costs[ko][0], is_feasible=costs[ko][1],
                           runtime_seconds=dt))
    print(f"  feasible {feas}/100   distinct winners {len(hits)}/{R}")
    print(f"  candidate dt: mean {sum(dts) / len(dts):.2f}s  max {max(dts):.2f}s")
    out = _DIR / "m79_knob_cloud_oraclepick.json"
    out.write_text(json.dumps({"submission_name": f"M79 knob cloud R={R} oracle",
                               "total_score": tot / TOTW,
                               "test_results": o_rows}), encoding="utf-8")
    print(f"  wrote {out.name}   total_score {tot / TOTW:.9f}")
    print(f"\n  next: m77_ml_candidate_probe.py score {out.name} --cores 48 "
          f"--dt {max(dts):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
