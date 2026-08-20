"""L140 - run the L135 soft-violation audit on HELD-OUT cases.

WHY. Handoff 2026-08-19 section 4: the same package scores vrel 0.0140 in set and
0.0967 on OOS s1 -- 6.9x -- and on OOS the violation axis is worth MORE than hpwl
(17.58% vs 12.20% if driven to zero). Every audit this project has run was on the
in-set 100, i.e. on the one distribution where violations barely exist. This runs
the identical audit where the violations actually are.

TWO STEPS, because the positions do not exist yet: the in-set audit reads a
results json produced by the official evaluator, and no such artefact exists for
the OOS corpus.

  run    solve the OOS sample with the SHIPPED optimizer and dump positions +
         the official per-case metrics (the same loader m77/m67/l133/l137 use,
         so this is the same 240 cases as every historical OOS number)
  audit  replay l135_soft_audit.audit_case over that dump

The audit code itself is IMPORTED from l135_soft_audit, never copied, so the
in-set and OOS columns cannot drift apart.

  <python> -u l140_oos_soft_audit.py run   --sample s1 --cores 48 --out l140_oos_s1_c48.json
  <python> -u l140_oos_soft_audit.py audit l140_oos_s1_c48.json --sample s1

Knobs: like l137_oos_ab, ICCAD_* are captured BEFORE importing m77_oos_probe
(which strips them at import time) and restored afterwards, so ICCAD_HINT_MODE et
al. reach both this process and the C++ children.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

_KNOBS = {k: v for k, v in os.environ.items() if k.startswith("ICCAD_")}

import torch                                                        # noqa: E402


def _load_specs(sample, limit):
    import m77_oos_probe as M
    specs = M._specs(sample)
    return specs[:limit] if limit else specs


def cmd_run(a):
    # specs FIRST: importing m77_oos_probe strips every ICCAD_* from the
    # environment (its line ~78), so the knobs have to be restored after it,
    # exactly as l137_oos_ab does -- otherwise both --cores and ICCAD_HINT_MODE
    # are silently dropped and the run measures shipped defaults.
    specs = _load_specs(a.sample, a.limit)
    os.environ.update(_KNOBS)
    if a.cores:
        os.environ["ICCAD_ADAPTIVE_CORES"] = str(a.cores)
    import m67_oos_probe as m67
    import optimizer_constructive as oc
    from proxy_analysis import build_opt_target_pos
    from iccad2026_evaluate import evaluate_solution

    # --bin: m77_oos_probe imports oc with ICCAD_* already stripped, so oc._BIN
    # is bound to the shipping exe before our restore runs and
    # ICCAD_CONSTRUCTIVE_BIN would be SILENTLY ignored on the normal path (which
    # uses the module-level _BIN, not the env). Set it directly.
    if a.bin:
        oc._BIN = Path(a.bin if os.path.isabs(a.bin) else str(_DIR / a.bin))
        print(f"[l140] binary override -> {oc._BIN.name}", flush=True)

    print(f"[l140] {len(specs)} cases, sample {a.sample}, "
          f"ADAPTIVE_CORES={os.environ.get('ICCAD_ADAPTIVE_CORES', 'auto')}, "
          f"HINT_MODE={os.environ.get('ICCAD_HINT_MODE', '0')}, "
          f"binary={oc._BIN.name if oc._BIN else '?'}", flush=True)
    opt = oc.MyOptimizer(verbose=False)

    by_file = defaultdict(list)
    order = {}
    for i, (ck, fk, lay_id, n) in enumerate(specs):
        by_file[fk].append((ck, lay_id, n))
        order[ck] = i

    rows = []
    t0 = time.time()
    for fk, items in by_file.items():
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in items:
            lay = m67._load_case(d, lay_id)
            assert lay["n"] == n, f"n mismatch {lay['n']} != {n} on {ck}"
            lay["base"], _dev = m67._baseline_official(lay)
            tt = torch.tensor([[float(v) for v in q] for q in lay["tp"]])
            otp = build_opt_target_pos(tt[:n], lay["cons"], n)
            t1 = time.perf_counter()
            P = opt.solve(n, lay["at"], lay["b2b"], lay["p2b"], lay["pins"],
                          lay["cons"], otp)
            dt = time.perf_counter() - t1
            m = evaluate_solution({"positions": [list(p) for p in P],
                                   "runtime": 1.0},
                                  lay["base"], lay["cons"], lay["b2b"],
                                  lay["p2b"], lay["pins"], lay["at"],
                                  target_positions=tt[:n], median_runtime=1.0)
            rows.append(dict(
                test_id=order[ck], key=ck, n=n,
                positions=[list(map(float, p)) for p in P],
                runtime_seconds=dt,
                feasible=bool(m.is_feasible), cost=float(m.cost),
                hpwl_gap=float(m.hpwl_gap), area_gap=float(m.area_gap),
                vrel=float(m.violations_relative),
                v_bnd=int(m.boundary_violations),
                v_grp=int(m.grouping_violations),
                v_mib=int(m.mib_violations),
                nsoft=int(m.max_possible_violations)))
            if a.verbose:
                print(f"  {ck:<34} n={n:>3}  vrel={m.violations_relative:.4f}"
                      f"  b/g/m={m.boundary_violations}/"
                      f"{m.grouping_violations}/{m.mib_violations}"
                      f"  {dt:.2f}s", flush=True)

    rows.sort(key=lambda r: r["test_id"])
    _summary(rows, f"{a.sample} run")
    print(f"[l140] wall {time.time() - t0:.0f}s")
    if a.out:
        json.dump(dict(submission_name="L140", sample=a.sample,
                       cores=a.cores or 0,
                       hint_mode=os.environ.get("ICCAD_HINT_MODE", "0"),
                       test_results=rows), open(a.out, "w"))
        print(f"[l140] wrote {a.out}")
    return 0


def _summary(rows, title):
    """Weighted aggregates, and the split of vrel into its three components.
    Weighting is the official exp(n/12) so this is comparable to every other
    OOS number in the project."""
    ws = sum(math.exp(r["n"] / 12.0) for r in rows)

    def wavg(k):
        return sum(math.exp(r["n"] / 12.0) * r[k] for r in rows) / ws

    print(f"\n=== L140 {title}: {len(rows)} cases ===")
    print(f"  feasible          {sum(1 for r in rows if r['feasible'])}"
          f"/{len(rows)}")
    for k in ("cost", "hpwl_gap", "area_gap", "vrel"):
        print(f"  weighted {k:<10}       {wavg(k):.6f}")
    for k, lab in (("v_bnd", "boundary"), ("v_grp", "grouping"),
                   ("v_mib", "MIB")):
        share = sum(math.exp(r["n"] / 12.0) * r[k] / max(r["nsoft"], 1)
                    for r in rows) / ws
        raw = sum(r[k] for r in rows)
        print(f"  vrel from {lab:<10}      {share:.6f}   "
              f"({100 * share / max(wavg('vrel'), 1e-12):5.1f}% of vrel, "
              f"{raw} raw)")
    hit = sum(1 for r in rows if r["v_bnd"] + r["v_grp"] + r["v_mib"] > 0)
    print(f"  cases with >=1 violation   {hit}/{len(rows)}")
    return wavg


def _price(rows, floors):
    """What each violation family is WORTH, on the official weighted total.

    cost = (1 + 0.5*(hpwl_gap + area_gap)) * exp(2*vrel), RF omitted -- the Beta
    result put us on the RF floor where the derivative is exactly zero (handoff
    section 2), so a violation-only change does not move it.

    The MIB-to-floor row is the one that matters: MIB-to-zero is unreachable
    whenever a group's target areas span more than 1.01/0.99 (identical shapes
    imply identical areas, and +-1% area is a HARD constraint), so the greedy
    bucket floor is the honest ceiling for that family."""
    def total(fn):
        num = den = 0.0
        for r in rows:
            w = math.exp(r["n"] / 12.0)
            v = fn(r) / max(r["nsoft"], 1)
            num += w * (1 + 0.5 * (r["hpwl_gap"] + r["area_gap"])) \
                * math.exp(2 * v)
            den += w
        return num / den

    base = total(lambda r: r["v_bnd"] + r["v_grp"] + r["v_mib"])
    print(f"\n=== what each family is worth (weighted total {base:.6f}) ===")
    scen = [
        ("boundary -> 0", lambda r: r["v_grp"] + r["v_mib"]),
        ("grouping -> 0", lambda r: r["v_bnd"] + r["v_mib"]),
        ("MIB      -> 0", lambda r: r["v_bnd"] + r["v_grp"]),
        ("MIB      -> provable floor",
         lambda r: r["v_bnd"] + r["v_grp"] + floors.get(r["key"], r["v_mib"])),
        ("all soft -> 0", lambda r: 0.0),
    ]
    for lab, fn in scen:
        t = total(fn)
        print(f"  {lab:<28} {t:.6f}   {100 * (base - t) / base:+7.4f}%")
    return base


def cmd_audit(a):
    import m67_oos_probe as m67
    import l135_soft_audit as L135

    blob = json.load(open(a.results))
    sample = a.sample or blob.get("sample", "s1")
    rows = blob["test_results"]
    by_key = {r["key"]: r for r in rows}
    _summary(rows, f"{sample} ({Path(a.results).name})")

    specs = _load_specs(sample, 0)
    by_file = defaultdict(list)
    for ck, fk, lay_id, n in specs:
        if ck in by_key:
            by_file[fk].append((ck, lay_id, n))

    G, B, M = [], [], []
    floors = {}
    tot_g = tot_b = tot_m = floor_m = 0
    for fk, items in by_file.items():
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in items:
            lay = m67._load_case(d, lay_id)
            r = by_key[ck]
            g, b, m, tg, tb, tm, fm = L135.audit_case(
                r["test_id"], r["positions"], lay["cons"], lay["at"], n)
            G += g
            B += b
            M += m
            tot_g += tg
            tot_b += tb
            tot_m += tm
            floor_m += fm
            floors[ck] = fm
            # cross-check against the official counts recorded by `run`
            if (tg, tb, tm) != (r["v_grp"], r["v_bnd"], r["v_mib"]):
                print(f"  !! {ck} audit {tg}/{tb}/{tm} != official "
                      f"{r['v_grp']}/{r['v_bnd']}/{r['v_mib']} (g/b/m)")

    L135.report(f"{Path(a.results).name} [{sample}]", G, B, M,
                tot_g, tot_b, tot_m, floor_m, a.show)
    _price(rows, floors)
    if a.dump:
        json.dump(dict(grp=G, bnd=B, mib=M), open(a.dump, "w"))
        print(f"[l140] wrote {a.dump}")
    return 0


def _nsoft(cons, n):
    """iccad2026_evaluate.py:459-471, verbatim."""
    tot = int((cons[:n, 4] != 0).sum().item())
    for col in (2, 3):
        c = cons[:n, col]
        for g in range(1, (int(c.max().item()) if c.numel() else 0) + 1):
            tot += max(0, int((c == g).sum().item()) - 1)
    return tot


def cmd_inset(a):
    """The same audit + pricing on the in-set 100, so the two columns come out
    of one code path. Reads an official results json (no solving)."""
    from iccad2026_evaluate import ContestEvaluator
    import l135_soft_audit as L135

    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    res = {r["test_id"]: r for r in json.load(open(a.results))["test_results"]}

    G, B, M = [], [], []
    rows, floors = [], {}
    tot_g = tot_b = tot_m = floor_m = 0
    for idx in sorted(res):
        r = res[idx]
        P = r.get("positions")
        if not P:
            continue
        at, _b2b, _p2b, _pins, cons = ev.dataset[idx]["input"]
        n = int((at != -1).sum().item())
        g, b, m, tg, tb, tm, fm = L135.audit_case(idx, P, cons, at, n)
        G += g
        B += b
        M += m
        tot_g += tg
        tot_b += tb
        tot_m += tm
        floor_m += fm
        floors[idx] = fm
        ns = _nsoft(cons, n)
        rows.append(dict(key=idx, n=n, nsoft=ns, feasible=r["is_feasible"],
                         cost=r["cost"], hpwl_gap=r["hpwl_gap"],
                         area_gap=r["area_gap"],
                         vrel=r["violations_relative"],
                         v_bnd=tb, v_grp=tg, v_mib=tm))
        got = (tb + tg + tm) / max(ns, 1)
        if abs(got - r["violations_relative"]) > 1e-9:
            print(f"  !! case {idx}: audit vrel {got:.6f} != official "
                  f"{r['violations_relative']:.6f}")

    _summary(rows, f"in-set ({Path(a.results).name})")
    L135.report(a.results, G, B, M, tot_g, tot_b, tot_m, floor_m, a.show)
    _price(rows, floors)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--sample", default="s1")
    r.add_argument("--cores", type=int, default=48)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", default="")
    r.add_argument("--bin", default="",
                   help="probe binary to use instead of the shipping exe")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(fn=cmd_run)

    q = sub.add_parser("audit")
    q.add_argument("results")
    q.add_argument("--sample", default="")
    q.add_argument("--show", type=int, default=25)
    q.add_argument("--dump", default="")
    q.set_defaults(fn=cmd_audit)

    i = sub.add_parser("inset")
    i.add_argument("results")
    i.add_argument("--show", type=int, default=25)
    i.set_defaults(fn=cmd_inset)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
