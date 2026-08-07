"""M80 — out-of-sample value of the M79 knob-cloud vectors as a FIXED pool tier.

OFFLINE PROBE — never shipped. Labels are used only to score.

WHY A SEPARATE TOOL. m77_oos_probe.py answers "what is ONE external candidate
worth out of sample" and its selftest pins that single-candidate path bit for bit
against the real wrapper. M80's mechanism is K fixed profiles added at once, and
K > 1 is not a repeat of K = 1:

  * the proxy's hmin is the min HPWL over the WHOLE pool, so each added vector
    shifts the normalizer for every other candidate — K vectors together are not
    the sum of K single additions (this is also why "adding candidates" is not
    monotone; M78 measured the same mechanism at -0.18% in one code path and
    +0.36% in another);
  * the wall is max(max_i dt_i, sum_i dt_i / cores), so K vectors share one
    max-term but pay K sum-terms. At 12 cores that second term is what makes the
    whole idea negative (M79: dRF +10.614% at K=8), which is the entire reason
    the shipped form has to be cores-gated.

So this tool reuses m77's 51.7 MB audit cache (2 samples x 240 cases x 35 shipped
profiles: positions + dt + proxy metrics) READ-ONLY, adds its own runs for the K
vectors, and reports the whole K = 0..Kmax prefix curve rather than one number.

THE NUMBER THAT DECIDES. NET = portfolio delta - dRF@48c, bar 0.30% (the
pre-registered M75/M76/M78 OOS bar). K is chosen on s1 and CONFIRMED on s2:
s1 (worker_0..9) is the corpus every historical OOS number lives on, s2
(worker_10..19) is disjoint. Both must agree in sign, because M76 measured that
an in-sample-chosen source set transfers at ~5%.

Run (PowerShell):
  <python> -u m80_tier_probe.py build --sample s1        # K x 240 solver runs
  <python> -u m80_tier_probe.py build --sample s2
  <python>    m80_tier_probe.py score --sample s1 --cores 48
  <python>    m80_tier_probe.py score --sample s2 --cores 48
  <python>    m80_tier_probe.py selftest --sample s1 --cores 48   # K=0 == anchor
"""
import argparse
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

for _k in sorted(k for k in os.environ if k.startswith("ICCAD_")):
    del os.environ[_k]

import m77_oos_probe as m77                                        # noqa: E402
import m67_oos_probe as m67                                        # noqa: E402
import optimizer_constructive as oc                                # noqa: E402

# HARD read-only guard, same reason m77 does it to m67: m77_oos_audit.pkl holds
# 2 x 240 x 35 solved combos and nothing here has the right to rewrite it. The
# file index both samples need is already built, so disabling the writer only
# costs a rescan in the case where it is missing.
m77._csave = lambda *a, **k: None                                  # noqa: E731

GAMMA = 0.3                        # RuntimeFactor exponent, evaluate.py:552
BAR_OOS_NET = 0.30                 # %, the M75/M76/M78 pre-registered OOS bar
WORKERS = 11                       # leave a core for this process (dt fidelity)
EXE = _DIR / "constructive.exe"
CACHE = _DIR / "m80_oos_cache.pkl"
VECFILE = _DIR / "m80_vectors.json"

# Same split as m77's SIG: pin everything that steers the cached POSITIONS, not
# the drop constants (those only pick WHICH cached profiles a pool uses at score
# time). The M80 vectors are deliberately NOT pinned -- entries are keyed by the
# vector's own hash, so growing K only pays for the new vectors.
SIG = repr(("m80oos", 1, m77._exe_md5(), oc._M55_BASE_LEN,
            repr(oc._PROFILES[:oc._M55_BASE_LEN]),
            repr(sorted(oc._M49_REFINE_BAND)),
            repr(sorted(oc._M50_REFINE_LOWCORE)),
            oc._M45_CORES_MAX,
            repr(sorted(oc._m71_env().items()))))

_CHILD_BASE = {k: v for k, v in os.environ.items() if not k.startswith("ICCAD_")}
_C = {"sig": SIG, "data": {}, "pm": {}, "cost": {}, "ahat": {}}


def _pkey(p):
    """Same hash m79_knob_cloud_probe.py keys its cloud by."""
    return hashlib.md5(repr(sorted(p.items())).encode()).hexdigest()[:16]


def _vectors(kmax=None):
    if not VECFILE.exists():
        sys.exit(f"{VECFILE.name} missing -> run "
                 f"`m79_knob_cloud_probe.py greedy <R> <KMAX>` first")
    j = json.loads(VECFILE.read_text(encoding="utf-8"))
    vecs = j["vectors"]
    if kmax:
        vecs = vecs[:kmax]
    return j, vecs


def _cload():
    global _C
    if not CACHE.exists():
        return
    try:
        c0 = pickle.load(open(CACHE, "rb"))
    except Exception as e:
        print(f"[cache] unreadable ({e!r}); starting fresh")
        return
    if c0.get("sig") == SIG:
        _C = c0
        for k in ("data", "pm", "cost", "ahat"):
            _C.setdefault(k, {})
    else:
        print("[cache] sig changed -> rebuilding (exe or shipped prefix moved)")


def _csave():
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_C, f, protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(6):                     # Windows: transient AV lock
        try:
            os.replace(tmp, CACHE)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))


# --------------------------------------------------------------------------- #
# build                                                                         #
# --------------------------------------------------------------------------- #
def _run_one(job):
    ck, pk, vec, txt, n, ov = job
    env = dict(_CHILD_BASE)
    env.update(vec)
    env.update(ov)                               # _band_env then _m71_env
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(EXE)], input=txt, capture_output=True, text=True,
                           env=env)
        dt = time.perf_counter() - t0
        pos = m77._parse_output(r.stdout, n) if r.returncode == 0 \
            and r.stdout.strip() else None
    except Exception:
        dt, pos = time.perf_counter() - t0, None
    return ck, pk, pos, dt


def mode_build(args):
    meta, vecs = _vectors(args.kmax)
    keys = [_pkey(v) for v in vecs]
    sample = args.sample
    specs = m77._specs(sample)
    if args.limit:
        specs = specs[:args.limit]

    print("=" * 78)
    print(f"M80 build   sample={sample}  K={len(vecs)}  exe md5 "
          f"{m77._exe_md5()[:12]}")
    print("=" * 78)
    print(f"  vectors from {VECFILE.name}: source={meta.get('source')} "
          f"R={meta.get('R')} order={meta.get('order')}")
    # A_hat is part of "complete": without it score() cannot select, and a case
    # whose runs are all cached would otherwise be skipped forever.
    todo_cases = [s for s in specs
                  if (sample, s[0]) not in _C["ahat"]
                  or any((sample, s[0], pk) not in _C["data"] for pk in keys)]
    print(f"  {len(vecs)} vectors x {len(specs)} cases = {len(vecs) * len(specs)} "
          f"combos; {len(specs) - len(todo_cases)} cases already complete")
    if not todo_cases:
        print("  nothing to do")
        return 0
    print(f"  overlay: n=80 {m77._band_overlay(80)}   "
          f"n=120 {m77._band_overlay(120)}")

    byfile = {}
    for ck, fk, L, n in todo_cases:
        byfile.setdefault(fk, []).append((ck, L, n))
    t0, done = time.time(), 0
    total = sum(1 for s in todo_cases for pk in keys
                if (sample, s[0], pk) not in _C["data"])
    fails = []
    for fi, fk in enumerate(sorted(byfile)):
        d = m77.torch.load(m67._path_of(fk))
        for ck, L, n in sorted(byfile[fk]):
            lay = m77._ctx(d, L, n)
            _C["ahat"][(sample, ck)] = m77._A_hat(lay)
            ov = m77._band_overlay(n)
            jobs = [(ck, pk, v, lay["txt"], n, ov)
                    for pk, v in zip(keys, vecs)
                    if (sample, ck, pk) not in _C["data"]]
            if not jobs:
                continue
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(_run_one, j) for j in jobs]
                for fut in concurrent.futures.as_completed(futs):
                    _ck, pk, pos, dt = fut.result()
                    _C["data"][(sample, _ck, pk)] = (pos, dt)
                    if pos is None:
                        fails.append((_ck, pk))
                    else:
                        # M47 pattern: proxy on this thread while the remaining
                        # subprocesses are still out, so it is ~free.
                        _C["pm"][(sample, _ck, pk)] = m77._pm_of(pos, lay)
                        _C["cost"][(sample, _ck, pk)] = _cost_of(pos, lay)
                    done += 1
            _csave()
        el = time.time() - t0
        print(f"[build] file {fi + 1}/{len(byfile)}  {done}/{total} combos  "
              f"({el:.0f}s, eta {el / max(done, 1) * (total - done):.0f}s)",
              flush=True)
    _csave()
    print(f"[build] done {done} combos in {time.time() - t0:.0f}s; "
          f"failed: {len(fails)} {fails[:6]}")
    return 0


def _cost_of(pos, lay):
    m = m67._cost(pos, lay)
    return (float(m.cost), bool(m.is_feasible))


# --------------------------------------------------------------------------- #
# score                                                                         #
# --------------------------------------------------------------------------- #
def _select(pm, A_hat, pool):
    """optimizer_constructive.py:1249-1255 verbatim: hmin over the WHOLE pool,
    first-best wins ties. The tier indices come LAST so a tie keeps the incumbent
    (the wrapper appends them at the end of _pool_indices() too)."""
    hmin = min(pm[k][1] for k in pool) or 1.0
    best, bestv = pool[0], float("inf")
    for k in pool:
        a, h, v = pm[k]
        val = (a / A_hat + m77.RH * h / hmin) * math.exp(2.0 * v)
        if val < bestv:
            bestv, best = val, k
    return best


def _rows(sample, specs, cores, keys, kmax, min_n):
    """Per case: the K = 0..kmax prefix curve of (winner, cost, wall)."""
    rows, need = [], {}
    for ck, _fk, _L, n in specs:
        want = m77._pool_at(n, cores)
        pool = [k for k in want
                if m77._C["data"].get((ck, k), (None,))[0] is not None]
        if len(pool) != len(want) or not pool:
            sys.exit(f"{ck}: m77 cache incomplete at cores={cores} -> run "
                     f"`m77_oos_probe.py build --sample {sample}`")
        pm = {k: m77._C["pm"][(ck, k)] for k in pool}
        dts = {k: m77._C["data"][(ck, k)][1] for k in pool}
        live = [pk for pk in keys if n > min_n]          # band gate
        for pk in live:
            if (sample, ck, pk) not in _C["pm"]:
                sys.exit(f"{ck}: vector {pk} not built -> run `build --sample "
                         f"{sample}`")
            pm[pk] = _C["pm"][(sample, ck, pk)]
            dts[pk] = _C["data"][(sample, ck, pk)][1]
        A_hat = _C["ahat"].get((sample, ck)) or m77._C["ahat"].get(ck)
        if A_hat is None:
            sys.exit(f"{ck}: A_hat missing -> run `build --sample {sample}`")
        r = dict(key=ck, n=n, pool=pool, wins=[], walls=[])
        ts = [dts[k] for k in pool]
        for K in range(kmax + 1):
            ex = live[:K]
            w = _select(pm, A_hat, pool + ex)
            r["wins"].append(w)
            tt = ts + [dts[pk] for pk in ex]
            r["walls"].append(max(max(tt), sum(tt) / cores))
            if isinstance(w, int) and (ck, w) not in m77._C["cost"] \
                    and (sample, ck, w) not in _C["cost"]:
                need.setdefault(ck, set()).add(w)
        rows.append(r)
    return rows, need


def _fill_costs(sample, specs, need):
    """Costs for shipped profiles that only become winners under some prefix.
    m77 fills its cost cache lazily for ITS winner, so a K > 0 winner is usually
    absent. One extra file walk, then cached forever."""
    if not need:
        return
    print(f"  scoring {sum(len(v) for v in need.values())} newly-winning shipped "
          f"profiles on {len(need)} cases (one file walk) ...", flush=True)

    def fn(ck, lay):
        for k in need[ck]:
            _C["cost"][(sample, ck, k)] = _cost_of(
                m77._C["data"][(ck, k)][0], lay)

    m77._walk(specs, set(need), fn)
    _csave()


def _cost(sample, ck, k):
    v = _C["cost"].get((sample, ck, k))
    if v is not None:
        return v[0]
    v = m77._C["cost"].get((ck, k))
    if v is not None:
        return v[0]
    raise KeyError((sample, ck, k))


def mode_score(args):
    meta, vecs = _vectors(args.kmax)
    keys = [_pkey(v) for v in vecs]
    sample, cores, kmax = args.sample, args.cores, len(vecs)
    specs = m77._specs(sample)
    rows, need = _rows(sample, specs, cores, keys, kmax, args.min_n)
    _fill_costs(sample, specs, need)

    print("=" * 78)
    print(f"M80 — OOS value of {kmax} fixed knob-cloud profiles as a pool tier")
    print("=" * 78)
    print(f"  sample    {sample} ({m77.SAMPLES[sample]['note']})")
    print(f"  vectors   {VECFILE.name}  source={meta.get('source')} "
          f"R={meta.get('R')}  order={meta.get('order')}")
    print(f"  pool      @{cores}c (tier-5 {'ON' if cores >= 40 else 'off'}, "
          f"tier-3 {'off' if cores > 16 else 'ON'})"
          + (f"   band gate n>{args.min_n}" if args.min_n else ""))
    anchor = m77.ANCHOR_OOS.get(sample, {}).get(cores)

    def tot(field):
        return m67._per_n_total([dict(n=r["n"], cost=r[field]) for r in rows])[0]

    for r in rows:
        r["c"] = [_cost(sample, r["key"], w) for w in r["wins"]]
    for r in rows:
        r["c0"] = r["c"][0]
    t0 = tot("c0")
    print(f"\n  shipped portfolio                  {t0:.9f}"
          + (f"   (known anchor {anchor:.6f})" if anchor else ""))

    print(f"\n  {'K':>3}{'total':>14}{'quality':>10}{'dRF':>9}{'NET':>10}"
          f"{'movers':>8}{'worse':>7}{'wall+':>7}")
    best = None
    for K in range(kmax + 1):
        for r in rows:
            r["cK"] = r["c"][K]
            r["rf"] = (r["walls"][K] / r["walls"][0]) ** GAMMA
            r["crf"] = r["cK"] * r["rf"]
        tK = tot("cK")
        q = (1 - tK / t0) * 100
        drf = (tot("crf") / tK - 1) * 100
        net = q - drf
        mv = sum(1 for r in rows if r["wins"][K] != r["wins"][0])
        wr = sum(1 for r in rows if r["c"][K] > r["c"][0] + 1e-12)
        wl = sum(1 for r in rows if r["walls"][K] > r["walls"][0] + 1e-12)
        print(f"  {K:>3}{tK:>14.9f}{q:>+9.3f}%{drf:>+8.3f}%{net:>+9.3f}%"
              f"{mv:>8}{wr:>7}{wl:>7}")
        if K and (best is None or net > best[1]):
            best = (K, net, q, drf)

    if best is None:
        sys.exit("no vectors in the json -> nothing to score")
    K, net, q, drf = best
    print(f"\n  best K = {K}   quality {q:+.3f}%   dRF@{cores}c {drf:+.3f}%   "
          f"NET {net:+.3f}%")
    print(f"  bar {BAR_OOS_NET:.2f}% (M75/M76/M78 OOS ship bar; M76 died at +0.10%)")
    verdict = "GREEN" if net >= BAR_OOS_NET else "RED (below the OOS bar)"
    print(f"  VERDICT: {verdict}")

    # band decomposition at the best K -- the band gate variant is decided here
    print(f"\n  band decomposition at K={K}")
    print(f"  {'band':<12}{'cases':>6}{'shipped':>12}{'+tier':>12}{'quality':>10}"
          f"{'dRF':>9}")
    for _b, lo, hi in m67.BANDS:
        sub = [r for r in rows if lo < r["n"] <= hi]
        if not sub:
            continue
        for r in sub:
            r["cK"] = r["c"][K]
            r["rf"] = (r["walls"][K] / r["walls"][0]) ** GAMMA
            r["crf"] = r["cK"] * r["rf"]

        def st(f):
            return m67._per_n_total([dict(n=r["n"], cost=r[f]) for r in sub])[0]
        a, c = st("c0"), st("cK")
        print(f"  ({lo:3d},{hi:3d}]{'':<3}{len(sub):>6}{a:>12.5f}{c:>12.5f}"
              f"{(1 - c / a) * 100:>+9.3f}%{(st('crf') / c - 1) * 100:>+8.3f}%")

    for r in rows:
        r["cK"] = r["c"][K]
    movers = sorted((r for r in rows if r["wins"][K] != r["wins"][0]),
                    key=lambda r: (r["cK"] - r["c0"]) * math.exp(r["n"] / 12.0))
    print(f"\n  top movers at K={K} (by weighted delta)")
    print(f"  {'case':<28}{'n':>4}{'shipped':>11}{'+tier':>11}{'d%':>8}{'win':>7}")
    for r in movers[:10] + ([] if len(movers) <= 14 else movers[-4:]):
        print(f"  {r['key']:<28}{r['n']:>4}{r['c0']:>11.5f}{r['cK']:>11.5f}"
              f"{(r['cK'] / r['c0'] - 1) * 100:>+8.2f}"
              f"{str(r['wins'][K])[:6]:>7}")

    p = _DIR / f"results_M80_oos_{sample}_c{cores}.json"
    json.dump(dict(sample=sample, cores=cores, kmax=kmax, min_n=args.min_n,
                   vectors=meta, shipped=t0, best_K=K, quality_pct=q,
                   drf_pct=drf, net_pct=net, bar_pct=BAR_OOS_NET,
                   verdict=verdict,
                   curve=[dict(K=k,
                               total=tot_k, quality=qk, drf=dk, net=qk - dk)
                          for k, tot_k, qk, dk in _curve(rows, t0, kmax, tot)]),
              open(p, "w"), indent=1)
    print(f"\n  wrote {p.name}")
    return 0 if net >= BAR_OOS_NET else 2


def _curve(rows, t0, kmax, tot):
    out = []
    for K in range(kmax + 1):
        for r in rows:
            r["cK"] = r["c"][K]
            r["rf"] = (r["walls"][K] / r["walls"][0]) ** GAMMA
            r["crf"] = r["cK"] * r["rf"]
        tK = tot("cK")
        out.append((K, tK, (1 - tK / t0) * 100, (tot("crf") / tK - 1) * 100))
    return out


def mode_inset(args):
    """The in-set K-prefix curve, rebuilt from the two caches M79 already paid
    for: m79_knob_cloud.pkl (positions + dt for every cloud vector on all 100
    validation cases) and audit_cache_ship.pkl (the same for the shipped pool
    under the shipping overlay). No solver runs.

    Two jobs. (1) It must reproduce m79_knob_cloud_probe.py's greedy curve
    exactly, which is what certifies that _M80_EXTRA is the set M79 measured.
    (2) It gives dRF at BOTH core counts — the 12-core column is the reason the
    shipped form is cores-gated and belongs in the report next to the 48-core
    one, not in a footnote."""
    import m79_knob_cloud_probe as m79                              # noqa: E402
    _meta, vecs = _vectors(args.kmax)
    keys = [_pkey(v) for v in vecs]
    c0 = pickle.load(open(_DIR / "m79_knob_cloud.pkl", "rb"))
    if c0.get("sig") != m79._sig():
        sys.exit("m79_knob_cloud.pkl signature != current exe/overlay")
    kd = c0["data"]
    ship = pickle.load(open(_DIR / "audit_cache_ship.pkl", "rb"))["data"]
    missing = [(c["idx"], pk) for c in m79.CASES for pk in keys
               if (c["idx"], pk) not in kd]
    if missing:
        sys.exit(f"{len(missing)} (case,vector) runs not in m79_knob_cloud.pkl "
                 f"{missing[:3]} -> re-run `m79_knob_cloud_probe.py run <R>`")
    pm, cost = c0["data"].setdefault("pm", {}), c0["data"].setdefault("cost", {})

    def POS(ci, k):
        return ship[(ci, k)][0] if isinstance(k, int) else kd[(ci, k)][0]

    def DT(ci, k):
        return ship[(ci, k)][1] if isinstance(k, int) else kd[(ci, k)][1]

    def PM(ci, k):
        if (ci, k) not in pm:
            pm[(ci, k)] = m79._proxy(m79.CASES[ci], POS(ci, k))
        return pm[(ci, k)]

    def CO(ci, k):
        if (ci, k) not in cost:
            cost[(ci, k)] = m79._true(m79.CASES[ci], POS(ci, k))[0]
        return cost[(ci, k)]

    print("=" * 78)
    print(f"M80 in-set — K-prefix curve on the 100 validation cases "
          f"(K max {len(vecs)})")
    print("=" * 78)
    for cores in (args.cores, 12):
        pools = {c["idx"]: m79._pool_at(c["n"], cores) for c in m79.CASES}
        base_t = None
        print(f"\n  @{cores}c pool (tier-5 {'ON' if cores >= 40 else 'off'}, "
              f"tier-3 {'off' if cores > 16 else 'ON'})")
        print(f"  {'K':>3}{'total':>16}{'quality':>10}{'dRF':>10}{'NET':>10}"
              f"{'movers':>8}{'wall+':>7}")
        for K in range(len(vecs) + 1):
            tot = trf = mv = wl = 0.0
            for c in m79.CASES:
                ci, pool = c["idx"], pools[c["idx"]]
                ex = keys[:K] if c["n"] > args.min_n else []
                cand = pool + ex
                hmin = min(PM(ci, k)[1] for k in cand) or 1.0
                w = min(cand, key=lambda k: (PM(ci, k)[0] / c["A_hat"]
                                             + m77.RH * PM(ci, k)[1] / hmin)
                        * math.exp(2 * PM(ci, k)[2]))
                ts = [DT(ci, k) for k in cand]
                wall = max(max(ts), sum(ts) / cores)
                if K == 0:                       # K=0 is this core count's base
                    c["_w0"], c["_win0"] = wall, w
                cc = CO(ci, w)
                tot += c["w"] * cc
                trf += c["w"] * cc * (wall / c["_w0"]) ** GAMMA
                mv += (w != c["_win0"])
                wl += (wall > c["_w0"] + 1e-12)
            t = tot / m79.TOTW
            if base_t is None:
                base_t = t
            q = (1 - t / base_t) * 100
            drf = (trf / tot - 1) * 100
            print(f"  {K:>3}{t:>16.9f}{q:>+9.3f}%{drf:>+9.3f}%{q - drf:>+9.3f}%"
                  f"{int(mv):>8}{int(wl):>7}")
    c0["data"]["pm"], c0["data"]["cost"] = pm, cost
    tmp = (_DIR / "m79_knob_cloud.pkl").with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(c0, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _DIR / "m79_knob_cloud.pkl")
    return 0


def mode_selftest(args):
    """K=0 must reproduce m77's shipped total exactly. If it does not, this tool's
    pool/selection/aggregation differs from the one every historical OOS number
    was measured with, and nothing below it means anything."""
    sample, cores = args.sample, args.cores
    specs = m77._specs(sample)
    rows, need = _rows(sample, specs, cores, [], 0, 0)
    _fill_costs(sample, specs, need)
    for r in rows:
        r["c0"] = _cost(sample, r["key"], r["wins"][0])
    ours = m67._per_n_total([dict(n=r["n"], cost=r["c0"]) for r in rows])[0]
    ref, _missing = m77._evaluate(specs, cores)
    theirs = m77._tot(ref, "base")
    anchor = m77.ANCHOR_OOS.get(sample, {}).get(cores)
    wdiff = sum(1 for a, b in zip(rows, ref) if a["wins"][0] != b["win"])
    print("=" * 78)
    print(f"M80 selftest   sample={sample}  cores={cores}")
    print("=" * 78)
    print(f"  winners differing from m77 : {wdiff}/{len(rows)}")
    print(f"  our K=0 total              : {ours:.9f}")
    print(f"  m77 shipped total          : {theirs:.9f}"
          + (f"   (known anchor {anchor:.6f})" if anchor else ""))
    ok = wdiff == 0 and abs(ours - theirs) < 1e-9
    print(f"\n  SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score", "selftest", "inset"])
    ap.add_argument("--sample", choices=sorted(m77.SAMPLES), default="s1")
    ap.add_argument("--cores", type=int, default=48,
                    help="pool shape to score at (48 = the grader's; M76 measured "
                         "a 2.7x difference vs 16)")
    ap.add_argument("--kmax", type=int, default=0,
                    help="use only the first K vectors (0 = all in the json)")
    ap.add_argument("--min-n", type=int, default=0, dest="min_n",
                    help="band gate: only cases with n > this get the tier")
    ap.add_argument("--limit", type=int, default=0,
                    help="build only: stop after N cases (smoke test)")
    args = ap.parse_args()
    _cload()
    m77._cload()
    return {"build": mode_build, "score": mode_score,
            "selftest": mode_selftest, "inset": mode_inset}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
