"""M58 probe (P5, M57_PLAN.md section 3): does switching compute_nsoft()'s MIB
term from distinct-shapes-1 (always 0 after apply_safe_mib_dims) to the
official evaluator's group_size-1 change compaction candidate selection and,
through it, the shipped-form score?

Uses constructive_m58.exe (copy of shipped constructive.cpp + env flag
ICCAD_NSOFT_OFFICIAL=1; flag off = same code path). All 100 eval cases carry
exactly one MIB group (official nsoft +2..+6), so the scan is the full set.

Modes:
  gate0   byte-gate: constructive_m58.exe (flag off) vs shipped
          constructive.exe -- 2 cases x 6 kept profiles, raw stdout compare.
  diff    100 cases x shipped-form kept pool (oc._pool_indices + oc._band_env,
          ADAPTIVE default-on) x both env sides -> per-profile stdout md5
          diff; winner-profile diff (spec criterion) + full-pool stats.
  eval    per-side _RH=1.4 proxy selection (wrapper parity: pool order,
          strict <) -> sanity of side-0 selection vs results_shipped_m51.json
          positions/cost -> strict official eval of both sides where the
          selected positions differ -> weighted delta ->
          results_M58_nsoft.json.
  all     gate0 + diff + eval.

Cache m58_cache.pkl keyed on (repr(pool), md5(constructive_m58.exe)).
Never shipped; quality-only probe (no runtime measurement).
"""
import os, sys, math, time, json, pickle, hashlib, subprocess, concurrent.futures
from pathlib import Path

# Clean ICCAD_* from the process env BEFORE importing oc (module-level reads)
# and before oc._pool_indices/_band_env call-time reads: shipped defaults only.
for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos
import optimizer_constructive as oc

EXE_SHIP = str(_DIR / "constructive.exe")
EXE_M58  = str(_DIR / "constructive_m58.exe")
CACHE    = _DIR / "m58_cache.pkl"
SHIPJ    = _DIR / "results_shipped_m51.json"
OUTJ     = _DIR / "results_M58_nsoft.json"
RH       = 1.4
WORKERS  = 11
GATE_CIS = [5, 95]                       # one small (n=26), one big (n=116)

PROFILES = list(oc._PROFILES)
assert len(PROFILES) == 41, f"shipped pool expected 41 profiles, got {len(PROFILES)}"

# ── dataset prep (all 100 cases) ─────────────────────────────────────────────
print("loading dataset...", flush=True)
ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
CASES, W = {}, {}
for idx in range(100):
    s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
    at, b2b, p2b, pins, cons = inp
    n = int((at != -1).sum().item())
    W[idx] = math.exp(n / 12.0)
    base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
    otp = build_opt_target_pos(tp, cons, n)
    sumA = sum(max(0.0, float(at[i])) for i in range(n))
    CASES[idx] = dict(idx=idx, n=n, A_hat=1.035 * max(sumA, 1e-9),
                      txt=_serialize_input(n, at, b2b, p2b, pins, cons, otp,
                                           gnn_hint=None),
                      base=base, tp=tp, at=at, b2b=b2b, p2b=p2b, pins=pins,
                      cons=cons,
                      kept=oc._pool_indices(n), band=oc._band_env(n))
TOTW = sum(W.values())
SHIP = {t["test_id"]: t for t in json.load(open(SHIPJ))["test_results"]}

# ── cache ────────────────────────────────────────────────────────────────────
def _md5b(b):
    return hashlib.md5(b).hexdigest()

SIG = repr((repr(PROFILES), _md5b(open(EXE_M58, "rb").read())))
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


# ── runners ──────────────────────────────────────────────────────────────────
def _env_of(ci, k, side):
    env = dict(os.environ)
    env.update(PROFILES[k])
    env.update(CASES[ci]["band"])        # band overrides profile (wrapper parity)
    if side:
        env["ICCAD_NSOFT_OFFICIAL"] = "1"
    return env


def _run_raw(exe, ci, k, side):
    r = subprocess.run([exe], input=CASES[ci]["txt"], capture_output=True,
                       text=True, env=_env_of(ci, k, side), timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"exe rc={r.returncode} ci={ci} k={k} side={side}")
    return r.stdout


def ensure_runs(jobs, tag):
    """jobs = [(ci, k, side)]; DB['run', ci, k, side] = (stdout_md5, positions)."""
    todo = [j for j in jobs if ("run",) + j not in DB]
    if not todo:
        return
    t0, done = time.perf_counter(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_run_raw, EXE_M58, *j): j for j in todo}
        for fut in concurrent.futures.as_completed(futs):
            j = futs[fut]
            out = fut.result()
            ps = _parse_output(out, CASES[j[0]]["n"])
            assert ps is not None, f"parse failed {j}"
            DB[("run",) + j] = (_md5b(out.encode()), ps)
            done += 1
            if done % 200 == 0:
                save()
                print(f"  [{tag}] {done}/{len(todo)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    save()
    print(f"  [{tag}] done {len(todo)} runs in {time.perf_counter()-t0:.0f}s",
          flush=True)


def all_jobs():
    return [(ci, k, side) for ci in range(100) for k in CASES[ci]["kept"]
            for side in (0, 1)]


# ── proxy / strict cost (deduped by stdout md5) ──────────────────────────────
def pm_of(ci, k, side):
    h, ps = DB[("run", ci, k, side)]
    kk = ("pm", ci, h)
    if kk not in DB:
        c = CASES[ci]
        m = oc._proxy_metrics(ps, c["at"], c["b2b"], c["p2b"], c["pins"],
                              c["cons"], c["n"])
        DB[kk] = (m["area"], m["hpwl"], m["vrel"])
    return DB[kk]


def cost_of_positions(ci, ps, h):
    kk = ("cost", ci, h)
    if kk not in DB:
        c = CASES[ci]
        tc = evaluate_solution({'positions': ps, 'runtime': 1.0}, c["base"],
                               c["cons"][:c["n"]], c["b2b"], c["p2b"],
                               c["pins"], c["at"][:c["n"]],
                               target_positions=c["tp"][:c["n"]],
                               median_runtime=1.0)
        DB[kk] = (tc.cost, bool(tc.is_feasible))
    return DB[kk]


def select_k(ci, side):
    """Deployed selector (wrapper parity): pool order, strict <."""
    c = CASES[ci]
    pms = {k: pm_of(ci, k, side) for k in c["kept"]}
    hmin = min(p[1] for p in pms.values()) or 1.0
    best_k, best_proxy = None, float("inf")
    for k in c["kept"]:
        a, hp, vr = pms[k]
        proxy = (a / c["A_hat"] + RH * hp / hmin) * math.exp(2.0 * vr)
        if proxy < best_proxy:
            best_proxy, best_k = proxy, k
    return best_k


# ── mode: gate0 ──────────────────────────────────────────────────────────────
def mode_gate0():
    print("=== gate0: constructive_m58.exe (flag off) vs shipped exe ===")
    jobs = [(ci, k) for ci in GATE_CIS for k in CASES[ci]["kept"][:6]]
    outs = {}
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_run_raw, exe, ci, k, 0): (exe, ci, k)
                for ci, k in jobs for exe in (EXE_SHIP, EXE_M58)}
        for fut in concurrent.futures.as_completed(futs):
            exe, ci, k = futs[fut]
            outs[(exe, ci, k)] = fut.result()
    bad = [(ci, k) for ci, k in jobs
           if outs[(EXE_SHIP, ci, k)] != outs[(EXE_M58, ci, k)]]
    print(f"gate0: {len(jobs)-len(bad)}/{len(jobs)} byte-identical "
          f"({time.perf_counter()-t0:.0f}s)")
    if bad:
        print("  FAIL:", bad)
        sys.exit(1)
    print("  PASS: flag-off m58 exe is byte-identical to shipped constructive.exe")


# ── mode: diff ───────────────────────────────────────────────────────────────
def mode_diff():
    print("=== diff: 100 cases x kept pool x both sides ===")
    ensure_runs(all_jobs(), "diff")
    per_case = {}
    for ci in range(100):
        ch = [k for k in CASES[ci]["kept"]
              if DB[("run", ci, k, 0)][0] != DB[("run", ci, k, 1)][0]]
        per_case[ci] = ch
    n_cases = sum(1 for ch in per_case.values() if ch)
    n_profs = sum(len(ch) for ch in per_case.values())
    n_runs  = sum(len(CASES[ci]["kept"]) for ci in range(100))
    print(f"full-pool diff: {n_profs}/{n_runs} profile runs changed, "
          f"{n_cases}/100 cases touched")

    print("computing side-0 proxies for winner identification...", flush=True)
    t0 = time.perf_counter()
    win_changed = []
    for ci in range(100):
        k0 = select_k(ci, 0)
        DB[("win0", ci)] = k0
        if DB[("run", ci, k0, 0)][0] != DB[("run", ci, k0, 1)][0]:
            win_changed.append((ci, k0))
        if ci % 20 == 19:
            save()
            print(f"  proxies {ci+1}/100 ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    save()
    print(f"winner-profile diff (spec criterion): {len(win_changed)}/100 cases "
          f"changed -> {win_changed}")
    DB[("diff_summary",)] = dict(per_case=per_case, win_changed=win_changed)
    save()
    if n_profs == 0:
        print("KILL GATE: zero position changes anywhere -> RED")
    return per_case, win_changed


# ── mode: eval ───────────────────────────────────────────────────────────────
def mode_eval():
    print("=== eval: per-side selection, sanity, strict eval, weighted delta ===")
    ensure_runs(all_jobs(), "eval")
    if ("diff_summary",) not in DB:
        mode_diff()
    per_case = DB[("diff_summary",)]["per_case"]

    rows, sane_pos, sane_cost = [], 0, 0
    t0 = time.perf_counter()
    for ci in range(100):
        c = CASES[ci]
        k0 = DB[("win0", ci)]
        k1 = select_k(ci, 1)
        h0, ps0 = DB[("run", ci, k0, 0)]
        h1, ps1 = DB[("run", ci, k1, 1)]
        cost0, feas0 = cost_of_positions(ci, ps0, h0)

        # sanity: side-0 selection must reproduce the shipped official eval
        sp = SHIP[ci]["positions"]
        pos_ok = (len(sp) == len(ps0) and
                  all(abs(sp[i][d] - ps0[i][d]) == 0.0
                      for i in range(len(sp)) for d in range(4)))
        cost_ok = abs(cost0 - SHIP[ci]["cost"]) < 1e-12
        sane_pos += pos_ok; sane_cost += cost_ok
        if not (pos_ok and cost_ok):
            print(f"  SANITY MISMATCH ci={ci}: pos_ok={pos_ok} "
                  f"cost0={cost0!r} shipped={SHIP[ci]['cost']!r}")

        if h1 == h0:
            cost1, feas1 = cost0, feas0
        else:
            cost1, feas1 = cost_of_positions(ci, ps1, h1)
        rows.append(dict(idx=ci, n=c["n"], w=W[ci], k0=k0, k1=k1,
                         changed_profiles=per_case[ci],
                         moved=(h1 != h0), cost0=cost0, cost1=cost1,
                         feasible0=feas0, feasible1=feas1))
        if ci % 20 == 19:
            save()
            print(f"  eval {ci+1}/100 ({time.perf_counter()-t0:.0f}s)", flush=True)
    save()

    print(f"sanity: positions {sane_pos}/100 bit-identical to shipped json, "
          f"cost {sane_cost}/100 within 1e-12")

    t_0 = sum(r["w"] * r["cost0"] for r in rows) / TOTW
    t_1 = sum(r["w"] * r["cost1"] for r in rows) / TOTW
    delta_pct = (t_1 - t_0) / t_0 * 100.0
    movers = [r for r in rows if r["cost1"] != r["cost0"]]
    flips  = [r for r in rows if r["k1"] != r["k0"]]
    print(f"\nweighted total side0={t_0:.10f} side1={t_1:.10f} "
          f"delta={delta_pct:+.4f}%  (negative = official denominator wins)")
    print(f"selection flips: {len(flips)}  cost movers: {len(movers)}")
    for r in sorted(movers, key=lambda r: -abs(r["w"] * (r["cost1"] - r["cost0"]))):
        wc = r["w"] * (r["cost1"] - r["cost0"]) / TOTW
        print(f"  case {r['idx']:3d} n={r['n']:3d} k {r['k0']:2d}->{r['k1']:2d} "
              f"cost {r['cost0']:.6f}->{r['cost1']:.6f} "
              f"dW={wc:+.3e} feas={r['feasible1']}")

    verdict = ("RED (no position change anywhere)" if not any(per_case.values())
               else "RED (weighted |delta| < 0.05%)" if abs(delta_pct) < 0.05
               else ("GREEN (improvement >= 0.05%)" if delta_pct <= -0.05
                     else "RED-adverse (official denominator LOSES >= 0.05%)"))
    print(f"kill-gate verdict: {verdict}")

    json.dump(dict(sig=SIG, total_side0=t_0, total_side1=t_1,
                   weighted_delta_pct=delta_pct, verdict=verdict,
                   sanity_positions=sane_pos, sanity_cost=sane_cost,
                   diff_profiles=sum(len(v) for v in per_case.values()),
                   diff_cases=sum(1 for v in per_case.values() if v),
                   winner_profile_changed=DB[("diff_summary",)]["win_changed"],
                   rows=rows),
              open(OUTJ, "w"), indent=1)
    print(f"wrote {OUTJ}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("gate0", "all"):
        mode_gate0()
    if mode in ("diff", "all"):
        mode_diff()
    if mode in ("eval", "all"):
        mode_eval()
