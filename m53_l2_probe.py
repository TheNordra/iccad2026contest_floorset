"""M53 L2 probe (OFFLINE, never shipped): stochastic best-of-N measurement.

L2 = gated noise in the greedy candidate scoring (constructive_l2.cpp copy,
ICCAD_NOISE / ICCAD_NOISE_W / ICCAD_SEED) -> N samples per case -> proxy picks.
Kill bar (user-set): weighted oracle gain < +0.3% of the L1 anchor -> RED.
Anchor = results_L1_final.json (1.3176, ICCAD_L1_POOL=1 + ADAPTIVE_POOL=0).

modes:
  gate0                 sigma=0 byte-identity gate: constructive_l2.exe vs the
                        shipped constructive.exe, 15 cases x 6 profiles, raw
                        stdout compare. MUST pass before anything else.
  phase0                15 cases x 84 L1 profiles (sigma=0, L2 exe): proxy
                        winner per case + anchor reproduction vs the json
                        (cost match + winner positions bit-match).
  phase1                sigma sweep on cases {99,96,94,89,85} x host profile:
                        mech B  ICCAD_NOISE   in {0.01,0.03,0.1,0.3}
                        mech A  ICCAD_NOISE_W in {0.05,0.15,0.3}
                        N=24 seeds each; oracle best-of-24 delta table.
  phase2 N SB SA        main curve: 15 cases x host (+2nd host on 89/85) x
                        seeds 1..N with ICCAD_NOISE=SB, ICCAD_NOISE_W=SA
                        (either may be 0). Checkpoints 8/16/32/64/128/256/...:
                        oracle + realizable curves + weighted verdict.
  report                reprint whatever phases the cache already holds.

Cache: m53_l2_cache.pkl keyed on (repr(pool), md5(constructive_l2.exe));
atomic .tmp+os.replace writes; every mode is resumable / incremental.
"""
import os, sys, math, time, json, pickle, hashlib, subprocess, concurrent.futures
from pathlib import Path

os.environ["ICCAD_L1_POOL"] = "1"        # BEFORE oc import (module-level extend)
os.environ["ICCAD_ADAPTIVE_POOL"] = "0"  # full pool; also disables _band_env

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc

EXE_SHIP = str(_DIR / "constructive.exe")
EXE_L2   = str(_DIR / "constructive_l2.exe")
CACHE    = _DIR / "m53_l2_cache.pkl"
ANCHORJ  = _DIR / "results_L1_final.json"
RH       = 1.4
WORKERS  = 11
PROBE    = list(range(85, 100))          # top-15 heaviest (n=106..120)
SWEEP    = [99, 96, 94, 89, 85]          # heaviest + the two cost outliers
GATE_KS  = [0, 20, 40, 41, 42, 83]       # base/mid/fa22/both L1 extras/K24 replica

PROFILES = list(oc._PROFILES)
assert len(PROFILES) == 84, f"L1 pool expected 84 profiles, got {len(PROFILES)}"

# ── dataset prep (weights for ALL 100; full case data only for the 15) ───────
ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
CASES, W = {}, {}
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    W[idx] = math.exp(n / 12.0)
    if idx not in PROBE:
        continue
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    sumA = sum(max(0.0, float(at[i])) for i in range(n))
    CASES[idx] = dict(idx=idx, n=n, A_hat=1.035 * max(sumA, 1e-9),
                      txt=_serialize_input(n, at, b2b, p2b, pins, cons, otp,
                                           gnn_hint=None),
                      base=base, tp=tp, at=at, b2b=b2b, p2b=p2b, pins=pins,
                      cons=cons)
TOTW = sum(W.values())

_aj = json.load(open(ANCHORJ))
ANCHOR_TOTAL = _aj["total_score"]
L1 = {t["test_id"]: t for t in _aj["test_results"]}

# ── cache ────────────────────────────────────────────────────────────────────
def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

SIG = repr((repr(PROFILES), _md5(EXE_L2)))
DB = {}
if CACHE.exists():
    try:
        _c = pickle.load(open(CACHE, "rb"))
        if _c.get("sig") == SIG:
            DB = _c["db"]
        else:
            print("[cache] signature mismatch (pool or exe changed) -> reset")
    except Exception:
        print("[cache] unreadable -> reset")


def save():
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"sig": SIG, "db": DB}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE)


# ── runners / lazy scoring ───────────────────────────────────────────────────
# job tuple j = (ci, k, sb, sa, seed); sb=ICCAD_NOISE, sa=ICCAD_NOISE_W.
def _env_of(k, sb, sa, seed):
    env = dict(os.environ); env.update(PROFILES[k])
    if sb > 0: env["ICCAD_NOISE"]   = f"{sb:g}"
    if sa > 0: env["ICCAD_NOISE_W"] = f"{sa:g}"
    if sb > 0 or sa > 0: env["ICCAD_SEED"] = str(seed)
    return env


def _run_raw(exe, ci, k, sb, sa, seed):
    t0 = time.perf_counter()
    r = subprocess.run([exe], input=CASES[ci]["txt"], capture_output=True,
                       text=True, env=_env_of(k, sb, sa, seed), timeout=600)
    return r.stdout, time.perf_counter() - t0


def ensure_runs(jobs, tag):
    todo = [j for j in jobs if ("run",) + j not in DB]
    if not todo:
        return
    t0, done = time.perf_counter(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_run_raw, EXE_L2, *j): j for j in todo}
        for fut in concurrent.futures.as_completed(futs):
            j = futs[fut]
            out, dt = fut.result()
            DB[("run",) + j] = (_parse_output(out, CASES[j[0]]["n"]), dt)
            done += 1
            if done % 50 == 0:
                save()
                print(f"  [{tag}] {done}/{len(todo)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    save()
    print(f"  [{tag}] done {len(todo)} runs in {time.perf_counter()-t0:.0f}s",
          flush=True)


def pm_of(j):
    kk = ("pm",) + j
    if kk not in DB:
        ci = j[0]; c = CASES[ci]; ps = DB[("run",) + j][0]
        m = oc._proxy_metrics(ps, c["at"], c["b2b"], c["p2b"], c["pins"],
                              c["cons"], c["n"])
        DB[kk] = (m["area"], m["hpwl"], m["vrel"])
    return DB[kk]


def cost_of(j):
    kk = ("cost",) + j
    if kk not in DB:
        ci = j[0]; c = CASES[ci]; ps = DB[("run",) + j][0]
        DB[kk] = evaluate_solution(
            {'positions': ps, 'runtime': 1.0}, c["base"], c["cons"][:c["n"]],
            c["b2b"], c["p2b"], c["pins"], c["at"][:c["n"]],
            target_positions=c["tp"][:c["n"]], median_runtime=1.0).cost
    return DB[kk]


def fill(jobs, fn, tag):
    t0, done = time.perf_counter(), 0
    for j in jobs:                       # serial: shapely proxies GIL-thrash
        fn(j)
        done += 1
        if done % 100 == 0:
            save()
            print(f"  [{tag}] {done}/{len(jobs)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    save()


def select_idx(ci, cand, pmmap):
    """Deployed _RH=1.4 proxy selector (mirrors optimizer_constructive)."""
    hmin = min(pmmap[j][1] for j in cand) or 1.0
    A = CASES[ci]["A_hat"]
    return min(cand, key=lambda j: (pmmap[j][0] / A + RH * pmmap[j][1] / hmin)
               * math.exp(2 * pmmap[j][2]))


def base_jobs(ci):
    return [(ci, k, 0.0, 0.0, 0) for k in range(len(PROFILES))]


# ── mode: gate0 ──────────────────────────────────────────────────────────────
def mode_gate0():
    jobs = [(ci, k) for ci in PROBE for k in GATE_KS]
    outs = {}
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_run_raw, exe, ci, k, 0.0, 0.0, 0): (exe, ci, k)
                for ci, k in jobs for exe in (EXE_SHIP, EXE_L2)}
        for fut in concurrent.futures.as_completed(futs):
            exe, ci, k = futs[fut]
            outs[(exe, ci, k)] = fut.result()[0]
    bad = [(ci, k) for ci, k in jobs
           if outs[(EXE_SHIP, ci, k)] != outs[(EXE_L2, ci, k)]]
    print(f"gate0: {len(jobs)-len(bad)}/{len(jobs)} byte-identical "
          f"({time.perf_counter()-t0:.0f}s)")
    if bad:
        print("  FAIL:", bad)
        sys.exit(1)
    print("  PASS: sigma=0 is bit-identical to shipped constructive.exe")


# ── mode: phase0 ─────────────────────────────────────────────────────────────
def mode_phase0():
    jobs = [j for ci in PROBE for j in base_jobs(ci)]
    ensure_runs(jobs, "phase0")
    fill(jobs, pm_of, "phase0-pm")
    print(f"\n{'ci':>3} {'n':>4} {'winner':>3} {'2nd':>3} "
          f"{'cost(win)':>12} {'json cost':>12} {'dcost':>10} pos")
    allok = True
    for ci in PROBE:
        cand = base_jobs(ci)
        pmm = {j: pm_of(j) for j in cand}
        win = select_idx(ci, cand, pmm)
        rest = [j for j in cand if j != win]
        sec = select_idx(ci, rest, {j: pmm[j] for j in rest})
        cw = cost_of(win)
        jc = L1[ci]["cost"]
        jpos = [tuple(map(float, p)) for p in L1[ci]["positions"]]
        opos = [tuple(map(float, p)) for p in DB[("run",) + win][0]]
        pos_ok = jpos == opos
        ok = pos_ok and abs(cw - jc) < 1e-9
        allok &= ok
        DB[("phase0", ci)] = (win[1], sec[1])
        print(f"{ci:>3} {CASES[ci]['n']:>4} {win[1]:>3} {sec[1]:>3} "
              f"{cw:>12.8f} {jc:>12.8f} {cw-jc:>10.2e} "
              f"{'MATCH' if pos_ok else 'DIFF'}{'' if ok else '  <-- CHECK'}")
    save()
    print("phase0:", "ALL MATCH — anchor reproduced, winners recorded"
          if allok else "MISMATCH -> investigate before phase1/2")


# ── mode: phase1 ─────────────────────────────────────────────────────────────
CFGS = [(0.01, 0.0), (0.03, 0.0), (0.10, 0.0), (0.30, 0.0),   # mech B
        (0.0, 0.05), (0.0, 0.15), (0.0, 0.30),                # mech A
        (0.0, 0.50), (0.0, 0.70)]                             # mech A ext (B RED)
N1 = 24


def mode_phase1():
    for ci in SWEEP:
        if ("phase0", ci) not in DB:
            sys.exit("phase0 winners missing -> run phase0 first")
    jobs = [(ci, DB[("phase0", ci)][0], sb, sa, s)
            for ci in SWEEP for sb, sa in CFGS for s in range(1, N1 + 1)]
    ensure_runs(jobs, "phase1")
    fill(jobs, cost_of, "phase1-cost")
    print(f"\nphase1: oracle best-of-{N1} true-cost delta vs L1 anchor "
          f"(negative = gain); host = phase0 winner")
    hdr = "  ".join(f"B{sb:g}" if sa == 0 else f"A{sa:g}" for sb, sa in CFGS)
    print(f"{'ci':>3} {'n':>4} {'host':>4} {'L1 cost':>10}  {hdr}")
    agg = {cfg: 0.0 for cfg in CFGS}
    for ci in SWEEP:
        host = DB[("phase0", ci)][0]
        row = []
        for sb, sa in CFGS:
            best = min(cost_of((ci, host, sb, sa, s)) for s in range(1, N1 + 1))
            d = best - L1[ci]["cost"]
            agg[(sb, sa)] += W[ci] * min(0.0, d)
            row.append(f"{d:+.4f}")
        print(f"{ci:>3} {CASES[ci]['n']:>4} {host:>4} "
              f"{L1[ci]['cost']:>10.6f}  " + "  ".join(f"{v:>8}" for v in row))
    print("\nweighted gain over the 5 sweep cases (% of anchor, "
          "best-of-24, improvements only):")
    for sb, sa in CFGS:
        tag = f"B={sb:g}" if sa == 0 else f"A={sa:g}"
        g = -agg[(sb, sa)] / TOTW / ANCHOR_TOTAL * 100
        print(f"  {tag:>7}: {g:+.4f}%")
    save()


# ── mode: phase2 ─────────────────────────────────────────────────────────────
def mode_phase2(N, sb, sa):
    for ci in PROBE:
        if ("phase0", ci) not in DB:
            sys.exit("phase0 winners missing -> run phase0 first")
    hosts = {ci: [DB[("phase0", ci)][0]] for ci in PROBE}
    for ci in (89, 85):                       # vrel-resample hypothesis: 2nd host
        hosts[ci].append(DB[("phase0", ci)][1])
    jobs = [(ci, k, sb, sa, s) for ci in PROBE for k in hosts[ci]
            for s in range(1, N + 1)]
    ensure_runs(jobs, "phase2")
    fill(jobs, cost_of, "phase2-cost")
    fill(jobs, pm_of, "phase2-pm")
    basepm = {ci: {j: pm_of(j) for j in base_jobs(ci)} for ci in PROBE}

    cks = [c for c in (8, 16, 32, 64, 128, 256, 500, 1000) if c <= N]
    if not cks or cks[-1] != N:
        cks.append(N)
    print(f"\nphase2: N={N} sb={sb:g} sa={sa:g}; oracle = min true cost over "
          f"samples; realizable = proxy pick over {{84 base + samples}}")
    print(f"{'ci':>3} {'host':>9} {'L1 cost':>10} " +
          " ".join(f"{'o@'+str(c):>9}" for c in cks) +
          f" {'r@'+str(N):>9} {'minVrel':>8}")
    wo = {c: 0.0 for c in cks}; wr = 0.0
    for ci in PROBE:
        sj = [(ci, k, sb, sa, s) for k in hosts[ci] for s in range(1, N + 1)]
        oc_ = {c: min(cost_of(j) for j in sj if j[4] <= c) for c in cks}
        pmm = dict(basepm[ci]); pmm.update({j: pm_of(j) for j in sj})
        rsel = select_idx(ci, list(pmm.keys()), pmm)
        rc = cost_of(rsel)
        mv = min(pm_of(j)[2] for j in sj)
        for c in cks:
            wo[c] += W[ci] * (oc_[c] - L1[ci]["cost"])
        wr += W[ci] * (rc - L1[ci]["cost"])
        print(f"{ci:>3} {'/'.join(map(str, hosts[ci])):>9} "
              f"{L1[ci]['cost']:>10.6f} " +
              " ".join(f"{oc_[c]-L1[ci]['cost']:>+9.4f}" for c in cks) +
              f" {rc-L1[ci]['cost']:>+9.4f} {mv:>8.4f}")
    print("\nweighted (% of anchor 1.3176; positive = gain):")
    for c in cks:
        print(f"  oracle  best-of-{c:>4}: {-wo[c]/TOTW/ANCHOR_TOTAL*100:+.4f}%")
    print(f"  realizable @N={N:>4}: {-wr/TOTW/ANCHOR_TOTAL*100:+.4f}%")
    g = -wo[cks[-1]] / TOTW / ANCHOR_TOTAL * 100
    print(f"\nVERDICT vs kill bar +0.3%: oracle {g:+.4f}% -> "
          f"{'GREEN candidate' if g >= 0.3 else 'RED (kill per user bar)'}")
    DB[("phase2", N, sb, sa)] = {"wo": wo, "wr": wr, "cks": cks}
    save()


# ── mode: union ──────────────────────────────────────────────────────────────
def mode_union():
    """Final deployable verdict: per case, oracle + proxy pick over the UNION
    of the 84 deterministic profiles and EVERY cached noisy sample (all sigma
    configs, all hosts, all seeds) — a sigma-mixture is a legal L2 ship form."""
    print(f"{'ci':>3} {'#smp':>5} {'L1 cost':>10} {'oracle d':>10} "
          f"{'realiz d':>10} {'minVrel':>8}")
    wo = wr = 0.0
    for ci in PROBE:
        sj = [k[1:] for k in DB
              if k[0] == "run" and k[1] == ci and (k[3] > 0 or k[4] > 0)]
        if not sj:
            continue
        fill([j for j in sj if ("cost",) + j not in DB], cost_of, f"u{ci}-cost")
        fill([j for j in sj if ("pm",) + j not in DB], pm_of, f"u{ci}-pm")
        ob = min(cost_of(j) for j in sj)
        pmm = {j: pm_of(j) for j in base_jobs(ci)}
        pmm.update({j: pm_of(j) for j in sj})
        rc = cost_of(select_idx(ci, list(pmm.keys()), pmm))
        do, dr = ob - L1[ci]["cost"], rc - L1[ci]["cost"]
        wo += W[ci] * do; wr += W[ci] * dr
        mv = min(pm_of(j)[2] for j in sj)
        print(f"{ci:>3} {len(sj):>5} {L1[ci]['cost']:>10.6f} {do:>+10.4f} "
              f"{dr:>+10.4f} {mv:>8.4f}")
    save()
    go = -wo / TOTW / ANCHOR_TOTAL * 100
    gr = -wr / TOTW / ANCHOR_TOTAL * 100
    print(f"\nUNION weighted (% of anchor): oracle {go:+.4f}%  "
          f"realizable {gr:+.4f}%")
    print(f"VERDICT vs kill bar +0.3%: "
          f"{'GREEN candidate' if go >= 0.3 else 'RED (kill per user bar)'}")


# ── mode: report ─────────────────────────────────────────────────────────────
def mode_report():
    p0 = [ci for ci in PROBE if ("phase0", ci) in DB]
    print(f"phase0 winners cached for {len(p0)}/15 cases:",
          {ci: DB[("phase0", ci)] for ci in p0})
    runs = [k for k in DB if k[0] == "run" and (k[3] > 0 or k[4] > 0)]
    print(f"noisy runs cached: {len(runs)}")
    for k in DB:
        if k[0] == "phase2":
            print("phase2 summary:", k, DB[k])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "gate0":
        mode_gate0()
    elif mode == "phase0":
        mode_phase0()
    elif mode == "phase1":
        mode_phase1()
    elif mode == "phase2":
        mode_phase2(int(sys.argv[2]),
                    float(sys.argv[3]) if len(sys.argv) > 3 else 0.10,
                    float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
    elif mode == "union":
        mode_union()
    else:
        mode_report()
