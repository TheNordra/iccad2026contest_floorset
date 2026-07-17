"""M61 diagnostic probe (OFFLINE, never shipped): blocked exact-edge candidate
-> event frames.

M57_PLAN.md section 6 (P4). Hypothesis: frame_candidates() (constructive.cpp
:416-441) is obstacle-blind -- it only sees total area / max block dims /
preplaced extents. A RIGHT/TOP boundary block's exact-origin candidate (the
only x resp. y that touches the wall) can be rejected by overlap in EVERY
generated frame; a frame grown by exactly the blocker clearance (an "event
frame" fw+dW or fh+dH) might open a new feasible topology no existing frame
reaches. LEFT/BOTTOM origins sit at 0 and do not move with frame growth.

Pipeline (spec: trace -> event frames, <=2 per base, dedup vs existing ->
FORCE_FRAME each -> official strict eval -> offline proxy arbitration):
  Phase A per target case {62,65,85,88,89,99}: winner host ksel via the
    m59/m60 winner_host mirror (asserted against the anchor json); byte-gate
    constructive_m61.exe (both envs off) == m53_l3_cache run positions; trace
    run ICCAD_FRAME_EVENT_TRACE=<file> with stdout still bit-identical; parse
    FRM (full frame_candidates list, incl. beyond max_trials -- the M51 ledger
    already proved "letting frame #5 be tried = 0 gain", so an event equal to
    an untried existing frame is a duplicate, not a discovery) and EVT records
    (src=I items loop / src=A anchored first-pass; d = min growth past all
    current blockers of that rejected exact-origin candidate).
  Phase B: group EVT by base frame; per base take min dW -> (fw+dW, fh) and
    min dH -> (fw, fh+dH) (= "<=2 per base"); drop d<=1e-9; dedup at the C++
    llround*1e6 key vs the FRM list and already-collected events.
  Phase C per event frame: run winner host env + ICCAD_FORCE_FRAME=WxH (single
    frame, rest of the per-frame pipeline untouched); new-topology check
    (positions != cached run); official STRICT eval (evaluate_solution with
    target_positions); proxy arbitration in wrapper semantics: _proxy_metrics
    of the new layout joined with the 84 cached ("pm",ci,k) entries, hmin
    recomputed, selected iff strict argmin (realizable); oracle = min official
    cost regardless of proxy.
  Phase D kill gate (spec): all events duplicate existing frames / no new
    feasible topology / best single-case weighted gain < 0.05% -> RED.
    Primary baseline = results_L1_final.json per-case cost (the pre-LP L1
    pool's realizable outcome -- LIVE semantics; M61 is a potential live
    profile shot, M51-type). The winner host's own pre-LP cost would
    overstate gains (post-LP re-selection can pick a pre-LP-worse host), so
    it is kept as info only. Post-LP offline anchor cost listed for A6.
    Verdict split: GREEN needs the bar met by a proxy-SELECTED event
    (realizable); oracle-only signal (proxy not selecting) -> YELLOW.

Honest scope: single host profile per case (winner only); d only guarantees
clearing the blockers present at that rejection (the shifted footprint may hit
new blockers -- absorbed by the FORCE_FRAME re-run); events beyond the two
min-d per base are not explored. Pure quality diagnostic -- no runtime
measurement.

Skeleton (dataset load / winner_host / run_exe / cache) copied from
m60_anchored_deficit.py; cost_eval copied from m59_refine_seed_probe.py per
the global no-import-and-patch rule.
Caches: m61_cache.pkl (own); m53_l3_cache.pkl READ-ONLY.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import subprocess
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

os.environ["ICCAD_L1_POOL"] = "1"        # BEFORE oc import (84-profile L1 pool)
os.environ["ICCAD_ADAPTIVE_POOL"] = "0"

from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
import optimizer_constructive as oc  # noqa: E402
from optimizer_claude import _serialize_input, _parse_output  # noqa: E402
from proxy_analysis import build_opt_target_pos  # noqa: E402

RH = 1.4
DMIN = 1e-9        # events below this growth are numerical noise
BAR_PCT = 0.05     # spec kill-gate bar on weighted single-case gain
TARGETS = (62, 65, 85, 88, 89, 99)
ANCHOR_JSON = _DIR / "results_L3_port_top32_area.json"
L1_JSON = _DIR / "results_L1_final.json"
L3_CACHE = _DIR / "m53_l3_cache.pkl"          # READ-ONLY
M61_CACHE = _DIR / "m61_cache.pkl"
EXE61 = str(_DIR / "constructive_m61.exe")
EXE_SHIPPED = str(_DIR / "constructive.exe")
SCRATCH = Path(os.environ.get(
    "M61_SCRATCH",
    r"C:\Users\Nordra\AppData\Local\Temp\claude"
    r"\C--Users-Nordra-Downloads-ICCAD2026-FloorSet-FloorSet"
    r"\778ec0ac-85d6-4c2f-a81f-7c77ccb62c6b\scratchpad"))

PROFILES = list(oc._PROFILES)
assert len(PROFILES) == 84, f"L1 pool expected 84 profiles, got {len(PROFILES)}"

# ── dataset (copied from m60_anchored_deficit.py) ────────────────────────────
print("[load] dataset ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
CASES, W = {}, {}
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _inp, _lab = _s["input"], _s["label"]
    _at, _b2b, _p2b, _pins, _cons = _inp
    _n = int((_at != -1).sum().item())
    W[_idx] = math.exp(_n / 12.0)
    _base, _tp = _ev._extract_baseline(_idx, _lab, _b2b, _p2b, _pins, _n)
    CASES[_idx] = dict(
        idx=_idx, n=_n, base=_base, tp=_tp, at=_at, b2b=_b2b, p2b=_p2b,
        pins=_pins, cons=_cons,
        cn=[[int(v) for v in _cons[i].tolist()] for i in range(_n)],
    )
TOTW = sum(W.values())
print("[load] 100 cases ready", flush=True)


def cost_eval(ci, ps):
    """Official strict scoring (target_positions passed -> hard checks on).
    Copied from m59_refine_seed_probe.py."""
    c = CASES[ci]
    return evaluate_solution(
        {"positions": ps, "runtime": 1.0}, c["base"], c["cons"][: c["n"]],
        c["b2b"], c["p2b"], c["pins"], c["at"][: c["n"]],
        target_positions=c["tp"][: c["n"]], median_runtime=1.0)


def winner_host(ci, db, anch_cost):
    """Mirror mode_portfull's selection (copied from m60_anchored_deficit.py):
    top-32 by pre-LP proxy, post-LP proxy re-selection -> ksel."""
    c = CASES[ci]
    A_hat = 1.035 * max(sum(max(0.0, float(c["at"][i]))
                            for i in range(c["n"])), 1e-9)
    pm = {k: db[("pm", ci, k)] for k in range(84)}
    hmin = min(v[1] for v in pm.values()) or 1.0
    prox = {k: (pm[k][0] / A_hat + RH * pm[k][1] / hmin)
            * math.exp(2 * pm[k][2]) for k in pm}
    top = sorted(pm, key=lambda k: prox[k])[:32]
    res = {k: db[("lp", ci, k, 8, True)] for k in top}
    h2 = {k: res[k][1] for k in top}
    hmin2 = min(h2.values()) or 1.0
    ab = float(c["base"].get("area_baseline", 1.0))
    prox2 = {k: ((1 + res[k][2]) * ab / A_hat + RH * h2[k] / hmin2)
             * math.exp(2 * res[k][3]) for k in top}
    ksel = min(top, key=lambda k: prox2[k])
    assert res[ksel][0] == anch_cost, (
        f"case {ci}: cache ksel={ksel} lp cost {res[ksel][0]!r} != anchor "
        f"{anch_cost!r} (data drift)")
    return ksel


def run_exe(ci, k, trace_path=None, force_frame=None):
    c = CASES[ci]
    otp = build_opt_target_pos(c["tp"], c["cons"], c["n"])
    txt = _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"], c["pins"],
                           c["cons"], otp, gnn_hint=None)
    env = dict(os.environ)
    env.update(PROFILES[k])
    if trace_path is not None:
        env["ICCAD_FRAME_EVENT_TRACE"] = str(trace_path)
    if force_frame is not None:
        env["ICCAD_FORCE_FRAME"] = f"{force_frame[0]!r}x{force_frame[1]!r}"
    r = subprocess.run([EXE61], input=txt, capture_output=True, text=True,
                       env=env, timeout=600)
    return _parse_output(r.stdout, c["n"])


# ── trace parsing ─────────────────────────────────────────────────────────────
def parse_trace(path):
    """-> (frames: [(fw,fh)], events: [(src, ax, fw, fh, d, b)])."""
    frames, events = [], []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if tok[0] == "FRM":
                frames.append((float(tok[1]), float(tok[2])))
            elif tok[0] == "EVT":
                kv = dict(t.split("=") for t in tok[1:])
                events.append((kv["src"], kv["ax"], float(kv["fw"]),
                               float(kv["fh"]), float(kv["d"]), int(kv["b"])))
    return frames, events


def fkey(w, h):
    """Frame identity key, mirrors the C++ llround*1e6 dedup."""
    return (round(w * 1e6), round(h * 1e6))


def synth_events(frames, events):
    """Phase B: per base frame min-dW / min-dH event frames, deduped vs the
    full existing frame list and among themselves.
    -> (kept: [dict], dropped_dup: int, raw_evt: int)"""
    exist = {fkey(w, h) for (w, h) in frames}
    by_base = {}
    for (src, ax, fw, fh, d, b) in events:
        if d <= DMIN:
            continue
        k = fkey(fw, fh)
        cur = by_base.setdefault(k, {})
        if ax not in cur or d < cur[ax][0]:
            cur[ax] = (d, src, b, fw, fh)
    kept, dropped = [], 0
    for k, axes in by_base.items():
        for ax, (d, src, b, fw, fh) in sorted(axes.items()):
            W2 = fw + d if ax == "W" else fw
            H2 = fh + d if ax == "H" else fh
            ek = fkey(W2, H2)
            if ek in exist:
                dropped += 1
                continue
            exist.add(ek)
            kept.append(dict(base_fw=fw, base_fh=fh, ax=ax, d=d, src=src,
                             blk=b, W=W2, H=H2))
    return kept, dropped, len(events)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None,
                    help="comma list override (default: spec targets)")
    args = ap.parse_args()
    targets = ([int(x) for x in args.cases.split(",")] if args.cases
               else list(TARGETS))

    aj = json.load(open(ANCHOR_JSON))
    ANCH = {t["test_id"]: t for t in aj["test_results"]}
    anchor_total = aj["total_score"]
    print(f"[anchor] {ANCHOR_JSON.name} total={anchor_total:.10f}", flush=True)
    lj = json.load(open(L1_JSON))
    L1 = {t["test_id"]: t["cost"] for t in lj["test_results"]}
    l1_total = lj["total_score"]
    print(f"[live]   {L1_JSON.name} total={l1_total:.10f}", flush=True)

    l3 = pickle.load(open(L3_CACHE, "rb"))          # READ-ONLY
    sig_expect = repr((repr(PROFILES),
                       hashlib.md5(open(EXE_SHIPPED, "rb").read()).hexdigest()))
    assert l3.get("sig") == sig_expect, \
        "m53_l3_cache signature != current pool/exe -> data drift, abort"
    db = l3["db"]

    msig = repr((repr(PROFILES),
                 hashlib.md5(open(EXE61, "rb").read()).hexdigest()))
    m61 = {}
    if M61_CACHE.exists():
        try:
            _c = pickle.load(open(M61_CACHE, "rb"))
            if _c.get("sig") == msig:
                m61 = _c["db"]
            else:
                print("[cache] m61 signature mismatch -> reset", flush=True)
        except Exception:
            print("[cache] m61 unreadable -> reset", flush=True)

    def save():
        tmp = M61_CACHE.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"sig": msig, "db": m61}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, M61_CACHE)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    report_cases, t0 = [], time.perf_counter()
    for ci in targets:
        c = CASES[ci]
        ksel = winner_host(ci, db, ANCH[ci]["cost"])
        runP = [tuple(p) for p in db[("run", ci, ksel)]]

        # Phase A ── byte-gate + trace
        gk = ("gate", ci, ksel)
        if gk not in m61:
            p_off = [tuple(p) for p in run_exe(ci, ksel)]
            assert p_off == runP, f"case {ci}: m61 exe (envs off) != cache run"
            m61[gk] = True
            save()
        trace_path = SCRATCH / f"m61_trace_{ci}.txt"
        tk = ("traceok", ci, ksel)
        if tk not in m61 or not trace_path.exists():
            p_on = [tuple(p) for p in run_exe(ci, ksel, trace_path=trace_path)]
            assert p_on == runP, f"case {ci}: trace run perturbed stdout"
            m61[tk] = True
            save()
        frames, events = parse_trace(trace_path)
        assert frames, f"case {ci}: no FRM records in trace"

        # Phase B ── event-frame synthesis
        kept, dropped_dup, raw = synth_events(frames, events)
        print(f"case {ci:3d} n={c['n']:3d} k={ksel:2d} frames={len(frames)} "
              f"raw_evt={raw} event_frames={len(kept)} dup_dropped="
              f"{dropped_dup} ({time.perf_counter() - t0:.0f}s)", flush=True)

        # baselines: L1 pool live outcome (primary), host's own pre-LP (info)
        bk = ("basecost", ci, ksel)
        if bk not in m61:
            mb = cost_eval(ci, runP)
            m61[bk] = (mb.cost, bool(mb.is_feasible))
            save()
        host_cost, host_feas = m61[bk]
        base_l1 = L1[ci]
        anch_cost = ANCH[ci]["cost"]

        # Phase C ── FORCE_FRAME each event frame
        A_hat = 1.035 * max(sum(max(0.0, float(c["at"][i]))
                                for i in range(c["n"])), 1e-9)
        pm84 = {k: db[("pm", ci, k)] for k in range(84)}
        rows = []
        for ev in kept:
            fkq = ("force", ci, ksel, fkey(ev["W"], ev["H"]))
            if fkq not in m61:
                m61[fkq] = [tuple(p) for p in
                            run_exe(ci, ksel, force_frame=(ev["W"], ev["H"]))]
                save()
            P = [tuple(p) for p in m61[fkq]]
            new_topo = P != runP
            m = cost_eval(ci, P)
            pmn = oc._proxy_metrics(P, c["at"], c["b2b"], c["p2b"], c["pins"],
                                    c["cons"], c["n"])
            hmin = min(min(v[1] for v in pm84.values()), pmn["hpwl"]) or 1.0
            prox = {k: (v[0] / A_hat + RH * v[1] / hmin) * math.exp(2 * v[2])
                    for k, v in pm84.items()}
            prox_new = ((pmn["area"] / A_hat + RH * pmn["hpwl"] / hmin)
                        * math.exp(2 * pmn["vrel"]))
            prox_best = min(prox.values())
            selected = prox_new < prox_best
            d_l1 = base_l1 - m.cost
            pct_l1 = 100.0 * (d_l1 * W[ci] / TOTW) / l1_total
            pct_anch = 100.0 * ((anch_cost - m.cost) * W[ci] / TOTW) / anchor_total
            rows.append(dict(
                ev=ev, new_topology=new_topo, feasible=bool(m.is_feasible),
                cost=m.cost, hpwl=m.hpwl_total, area_gap=m.area_gap,
                vrel=m.violations_relative, prox_new=prox_new,
                prox_pool_best=prox_best, proxy_selected=bool(selected),
                d_l1=d_l1, pct_l1=pct_l1, pct_anchor=pct_anch))
            print(f"    EVT {ev['ax']}+{ev['d']:.6g} (src={ev['src']} "
                  f"b={ev['blk']} base={ev['base_fw']:.4g}x"
                  f"{ev['base_fh']:.4g}) -> feas={m.is_feasible} "
                  f"new_topo={new_topo} cost={m.cost:.6f} "
                  f"(L1 {base_l1:.6f} host {host_cost:.6f} "
                  f"anchor {anch_cost:.6f}) "
                  f"d_l1={d_l1:+.6f} pct_l1={pct_l1:+.4f}% "
                  f"proxy_sel={selected}", flush=True)

        report_cases.append(dict(
            ci=ci, n=c["n"], ksel=ksel, n_frames=len(frames), raw_events=raw,
            n_event_frames=len(kept), dup_dropped=dropped_dup,
            base_cost_l1=base_l1, host_pre_lp_cost=host_cost,
            host_feasible=host_feas, anchor_cost=anch_cost, events=rows))

    # Phase D ── kill gate
    all_rows = [(rc, r) for rc in report_cases for r in rc["events"]]
    n_events = len(all_rows)
    n_new_feas = sum(1 for _, r in all_rows if r["feasible"]
                     and r["new_topology"])
    ok = [(rc, r) for rc, r in all_rows
          if r["feasible"] and r["new_topology"]]
    best_o = max(ok, key=lambda t: t[1]["pct_l1"], default=None)
    best_oracle = best_o[1]["pct_l1"] if best_o else 0.0
    sel = [(rc, r) for rc, r in ok if r["proxy_selected"]]
    best_r = max(sel, key=lambda t: t[1]["pct_l1"], default=None)
    best_real = best_r[1]["pct_l1"] if best_r else 0.0
    if n_events == 0:
        why = "all events duplicate existing frames (or no events at all)"
        verdict = "RED"
    elif n_new_feas == 0:
        why = "no new feasible topology from any event frame"
        verdict = "RED"
    elif best_real >= BAR_PCT:
        why = (f"realizable (proxy-selected) best single-case weighted gain "
               f"{best_real:+.4f}% >= {BAR_PCT}%")
        verdict = "GREEN"
    elif best_oracle >= BAR_PCT:
        why = (f"oracle best {best_oracle:+.4f}% >= {BAR_PCT}% but proxy "
               f"never selects it (realizable best {best_real:+.4f}%)")
        verdict = "YELLOW-ORACLE-ONLY"
    else:
        why = (f"best single-case weighted gain {best_oracle:+.4f}% "
               f"(oracle) < {BAR_PCT}% bar")
        verdict = "RED"
    print(f"\n== M61 verdict: {verdict} ({why}) ==", flush=True)
    print(f"== events={n_events} new-feasible-topologies={n_new_feas} "
          f"best_oracle_case={best_o[0]['ci'] if best_o else None} "
          f"best_realizable_case={best_r[0]['ci'] if best_r else None} ==",
          flush=True)

    out = dict(verdict=verdict, why=why, bar_pct=BAR_PCT,
               n_event_frames=n_events, n_new_feasible=n_new_feas,
               best_pct_oracle=best_oracle, best_pct_realizable=best_real,
               anchor_total=anchor_total, l1_total=l1_total,
               cases=report_cases,
               timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(_DIR / "m61_events.json", "w") as f:
        json.dump(out, f, indent=1)
    print("[dump] m61_events.json", flush=True)


if __name__ == "__main__":
    main()
