"""L144 - are the held-out BOUNDARY violations even SATISFIABLE?

L140: boundary is the only violation family on OOS with headroom (+2.44%..+4.49%).
L141: 99.6% of side-misses are on sides with spare TOTAL capacity.
L143: the edge is fragmented at the moment of placement.
L144 trace: a compliant candidate existed 78.3% of the time and was taken 100%
            of those times -- so the whole gap is the 21.7% where none exists.

This probe asks the question underneath all of those, with no solver in the loop:
for every (case, side) on held-out data, do the blocks REQUIRED to touch that side
physically fit along it?

Blocks sitting on one side all share that side's coordinate, so their intervals
along the edge are pairwise disjoint. 1-D interval packing with free positions is
feasible iff  sum(extent_i) <= capacity, so the whole question is a sum.

Three regimes, each a strict RELAXATION of the previous one (so each can only
over-state feasibility -- every "INFEASIBLE" verdict below is a proof, every
"feasible" verdict is only "not ruled out by edge capacity"):

  (i)   achieved shapes, achieved bbox extent.
        capacity = H for L/R, W for T/B on the frame the placer actually built.

  (ii)  achieved shapes, frame free to re-proportion.
        ASSUMPTION, stated: the frame may be any W'xH' with W'*H' = A_bb, the
        ACHIEVED bbox area. Justification: (a) the layout must still hold the
        blocks, and A_bb is the only packing density this placer has actually
        demonstrated on this case -- a smaller frame area assumes a density it
        never reached; (b) a LARGER frame is not free, area_gap is a first-class
        cost term (cost = (1 + alpha*(hpwl_gap + area_gap)) * exp(beta*V_rel)),
        so growing area to buy a boundary fix has to be priced, not assumed.
        The frame must still contain every block: W' >= wmax, H' >= hmax.
        per-side  : side L/R passes iff D <= A_bb / wmax  (H' raised as far as
                    the widest block allows).
        joint     : all four sides at once iff
                    max(D_L,D_R,hmax) * max(D_T,D_B,wmax) <= A_bb.
        Also reports the frame-area multiplier the joint test would need.

  (iii) free aspect inside the +-1% area constraint, achieved bbox extent.
        A movable, non-fixed-shape block may pick the along-edge extent that
        minimises its footprint on that edge:
            e_min = sqrt(0.99 * area_target_i / R)
        (area at its legal floor 0.99*a_i, aspect at the bound R with the long
        side pointing INTO the frame). Capped by the achieved extent, since the
        placer can always reproduce what it already did. Fixed-shape and
        preplaced blocks keep their extent -- their (w,h) is a HARD constraint
        (iccad2026_evaluate.check_dimension_hard_constraints, tol 1e-4).
        R = 2, 2.5, 4, 8. R=2.5 is not arbitrary: constructive.cpp:121-122 ships
        LR_ASPECT 2.50 / TB_ASPECT 0.40 (= 1/2.5) for exactly this purpose.

  (iv)  (ii) + (iii) together, the loosest bound in the file.

Preplaced blocks are handled as the HARD constraint they are: a preplaced block
required to touch a side FORCES that side's bbox coordinate to its own, and the
side is provably over-constrained when two required preplaced blocks disagree,
or when some other preplaced block already sits beyond the forced coordinate.
That is L135's finding generalised to held-out data.

READ-ONLY. Writes nothing, runs no binary.

  <python> -u l144_feas_probe.py l140_oos_s1_c48.json --sample s1
  <python> -u l144_feas_probe.py l140_oos_s2_c48.json --sample s2 --show 20
"""
import argparse
import collections
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# m77_oos_probe deletes every ICCAD_* at import time anyway; do it here too so
# the deletion is visible rather than a side effect of an import.
for _k in [k for k in os.environ if k.startswith("ICCAD_")]:
    del os.environ[_k]

import torch                                                        # noqa: E402

import m67_oos_probe as m67                                         # noqa: E402
import m77_oos_probe as m77                                         # noqa: E402

EPS = 1e-6
AREA_TOL = 0.01          # iccad2026_evaluate.AREA_TOLERANCE
BITS = ((1, "L"), (2, "R"), (4, "T"), (8, "B"))
CORNERS = ((1, 8, "LB"), (1, 4, "LT"), (2, 8, "RB"), (2, 4, "RT"))
ASPECTS = (2.0, 2.5, 4.0, 8.0)


# --------------------------------------------------------------------------- #
# corpus                                                                       #
# --------------------------------------------------------------------------- #
def cases(results, sample):
    """Yield a dict per case: achieved positions + the full input spec."""
    blob = json.load(open(results))
    rows = {r["key"]: r for r in blob["test_results"]}
    specs = m77._specs(sample or blob.get("sample", "s1"), verbose=False)
    byf = collections.defaultdict(list)
    for ck, fk, lay_id, n in specs:
        if ck in rows:
            byf[fk].append((ck, lay_id, n))
    for fk in sorted(byf):
        d = torch.load(m67._path_of(fk))
        for ck, lay_id, n in byf[fk]:
            lay = m67._load_case(d, lay_id)
            r = rows[ck]
            yield dict(idx=r["test_id"], key=ck, n=n,
                       P=[list(map(float, p)) for p in r["positions"][:n]],
                       cons=lay["cons"], at=lay["at"], tp=lay["tp"],
                       v_bnd=int(r["v_bnd"]), v_grp=int(r["v_grp"]),
                       v_mib=int(r["v_mib"]), cost=float(r["cost"]))


# --------------------------------------------------------------------------- #
# per-case geometry                                                            #
# --------------------------------------------------------------------------- #
def analyse(c, aspects=ASPECTS):
    n, P, cons = c["n"], c["P"], c["cons"]
    code = [int(cons[i][4]) for i in range(n)]
    fixed = [float(cons[i][0]) != 0 for i in range(n)]
    prepl = [float(cons[i][1]) != 0 for i in range(n)]
    at = [float(c["at"][i]) for i in range(n)]
    tp = c["tp"]

    x0 = min(p[0] for p in P)
    y0 = min(p[1] for p in P)
    x1 = max(p[0] + p[2] for p in P)
    y1 = max(p[1] + p[3] for p in P)
    W, H = x1 - x0, y1 - y0
    A_bb = W * H
    wmax = max(p[2] for p in P)
    hmax = max(p[3] for p in P)

    # --- preplaced forcing / over-constraint (item 4) ----------------------- #
    pp = [i for i in range(n) if prepl[i]]
    pre = {}                    # side -> dict(state, coord)
    for bit, side in BITS:
        rq = [i for i in pp if code[i] & bit]
        if not rq:
            pre[side] = dict(state="free", coord=None, nreq=0)
            continue
        if side == "L":
            val = lambda i: tp[i][0]
            dom = lambda v: min(tp[j][0] for j in pp) < v - EPS
        elif side == "R":
            val = lambda i: tp[i][0] + tp[i][2]
            dom = lambda v: max(tp[j][0] + tp[j][2] for j in pp) > v + EPS
        elif side == "B":
            val = lambda i: tp[i][1]
            dom = lambda v: min(tp[j][1] for j in pp) < v - EPS
        else:
            val = lambda i: tp[i][1] + tp[i][3]
            dom = lambda v: max(tp[j][1] + tp[j][3] for j in pp) > v + EPS
        vs = sorted({round(val(i), 9) for i in rq})
        if len(vs) > 1:
            pre[side] = dict(state="conflict", coord=None, nreq=len(rq))
        elif dom(vs[0]):
            pre[side] = dict(state="dominated", coord=vs[0], nreq=len(rq))
        else:
            pre[side] = dict(state="forced", coord=vs[0], nreq=len(rq))

    # --- corner conflicts --------------------------------------------------- #
    corner_bad = []
    for b1, b2, nm in CORNERS:
        k = [i for i in range(n) if (code[i] & b1) and (code[i] & b2)]
        if len(k) > 1:
            corner_bad.append((nm, len(k)))

    # --- per-side capacity tests ------------------------------------------- #
    sides = {}
    D = {}
    Dhard = {}
    Dfree = {a: {} for a in aspects}
    for bit, side in BITS:
        rq = [i for i in range(n) if code[i] & bit]
        vert = side in "LR"
        cap_i = H if vert else W
        ext = {i: (P[i][3] if vert else P[i][2]) for i in rq}
        D[side] = sum(ext.values())
        for R in aspects:
            tot = 0.0
            for i in rq:
                e = ext[i]
                if not (fixed[i] or prepl[i]) and at[i] > 0:
                    e = min(e, math.sqrt((1.0 - AREA_TOL) * at[i] / R))
                tot += e
            Dfree[R][side] = tot
        # rules-only demand: the contest bounds AREA (+-1%) but never ASPECT, so
        # a movable soft block's along-edge extent has no positive lower bound.
        # Only fixed-shape and preplaced blocks contribute a demand that no
        # legal solution can shrink.
        Dhard[side] = sum(ext[i] for i in rq if fixed[i] or prepl[i])
        touch = {1: lambda p: abs(p[0] - x0) < EPS,
                 2: lambda p: abs(p[0] + p[2] - x1) < EPS,
                 4: lambda p: abs(p[1] + p[3] - y1) < EPS,
                 8: lambda p: abs(p[1] - y0) < EPS}[bit]
        miss = [i for i in rq if not touch(P[i])]
        # max-cardinality subset with sum <= cap  ->  greedy smallest-first
        room, kept = cap_i, 0
        for i in sorted(rq, key=lambda j: ext[j]):
            if ext[i] <= room + EPS:
                room -= ext[i]
                kept += 1
        sides[side] = dict(bit=bit, req=rq, nreq=len(rq), ext=ext,
                           cap_i=cap_i, miss=len(miss),
                           forced_i=len(rq) - kept if rq else 0,
                           pre=pre[side])

    # frame extent forced on an axis by preplaced boundary blocks on BOTH ends.
    # This is the ONLY upper bound on frame extent that the rules themselves
    # impose, and it is the capacity a fully COMPLIANT layout would have to live
    # inside -- not the achieved capacity, which the placer is free to (and does)
    # grow by breaking one of the pinning constraints. See case 85.
    axis_forced = {}
    if pre["L"]["state"] == "forced" and pre["R"]["state"] == "forced":
        axis_forced["W"] = pre["R"]["coord"] - pre["L"]["coord"]
    if pre["B"]["state"] == "forced" and pre["T"]["state"] == "forced":
        axis_forced["H"] = pre["T"]["coord"] - pre["B"]["coord"]
    cap_pin = {"L": axis_forced.get("H"), "R": axis_forced.get("H"),
               "T": axis_forced.get("W"), "B": axis_forced.get("W")}

    cap_ii = {"L": A_bb / max(wmax, 1e-12), "R": A_bb / max(wmax, 1e-12),
              "T": A_bb / max(hmax, 1e-12), "B": A_bb / max(hmax, 1e-12)}
    need_H = max(D["L"], D["R"], hmax)
    need_W = max(D["T"], D["B"], wmax)
    joint_ii = need_H * need_W / max(A_bb, 1e-12)

    for bit, side in BITS:
        s = sides[side]
        if not s["nreq"]:
            continue
        s["i"] = D[side] <= s["cap_i"] + EPS
        s["ii"] = D[side] <= cap_ii[side] + EPS
        s["cap_ii"] = cap_ii[side]
        s["ratio_i"] = D[side] / max(s["cap_i"], 1e-12)
        s["ratio_ii"] = D[side] / max(cap_ii[side], 1e-12)
        for R in aspects:
            s[f"iii_{R}"] = Dfree[R][side] <= s["cap_i"] + EPS
            s[f"iv_{R}"] = Dfree[R][side] <= cap_ii[side] + EPS
            s[f"ratio_iii_{R}"] = Dfree[R][side] / max(s["cap_i"], 1e-12)
        # --- regime (0): the pinned frame a compliant layout must live in ---
        cp = cap_pin[side]
        s["cap_pin"] = cp
        s["pinned"] = cp is not None
        if cp is None:
            s["pin_hard"] = s["pin_ach"] = True
            s["pin_a25"] = True
            s["forced_pin"] = 0
        else:
            s["pin_hard"] = Dhard[side] <= cp + EPS      # rules-only proof
            s["pin_ach"] = D[side] <= cp + EPS           # achieved shapes
            s["pin_a25"] = Dfree[2.5][side] <= cp + EPS  # shipped aspect
            se = s["ext"]
            room, kept = cp, 0
            for i in sorted(s["req"], key=lambda j: se[j]):
                if se[i] <= room + EPS:
                    room -= se[i]
                    kept += 1
            s["forced_pin"] = len(s["req"]) - kept
        s["ratio_pin"] = (D[side] / cp) if cp else 0.0
        s["ratio_pin_hard"] = (Dhard[side] / cp) if cp else 0.0

    # provably-forced boundary violations from PREPLACED CONFLICTS alone: with
    # k preplaced blocks required on one side sitting at distinct coordinates,
    # at most the largest same-coordinate group can comply.
    forced_pre = 0
    for bit, side in BITS:
        if pre[side]["state"] != "conflict":
            continue
        rq = [i for i in pp if code[i] & bit]
        f = {1: lambda i: tp[i][0], 2: lambda i: tp[i][0] + tp[i][2],
             4: lambda i: tp[i][1] + tp[i][3], 8: lambda i: tp[i][1]}[bit]
        cnt = collections.Counter(round(f(i), 9) for i in rq)
        forced_pre += len(rq) - max(cnt.values())

    return dict(sides=sides, D=D, Dhard=Dhard, Dfree=Dfree, W=W, H=H, A_bb=A_bb,
                forced_pre=forced_pre,
                wmax=wmax, hmax=hmax, joint_ii=joint_ii, pre=pre,
                corner_bad=corner_bad, axis_forced=axis_forced,
                n_pre=len(pp), n_bnd=sum(1 for i in range(n) if code[i]))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="l140_oos_s1_c48.json")
    ap.add_argument("--sample", default="")
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()

    regimes = ["i", "ii"] + [f"iii_{R}" for R in ASPECTS] + \
              [f"iv_{R}" for R in ASPECTS]
    side_tot = 0
    side_pass = collections.Counter()
    case_bad = {g: set() for g in regimes}
    case_tot = set()
    forced_max = collections.Counter()      # case -> max_s forced_i
    forced_sum = collections.Counter()
    per_case = {}
    joint_ii_fail = set()
    joint_mult = []
    pre_state = collections.Counter()
    pre_axis_forced = collections.Counter()
    pre_axis_tight = []
    corner_cases = []
    ratios_i = []
    W_of = {}

    for c in cases(a.results, a.sample):
        r = analyse(c)
        idx = c["idx"]
        per_case[idx] = (c, r)
        W_of[idx] = math.exp(c["n"] / 12.0)
        if r["n_bnd"] == 0:
            continue
        case_tot.add(idx)
        fmax = fsum = 0
        for _b, side in BITS:
            s = r["sides"][side]
            if not s["nreq"]:
                continue
            side_tot += 1
            ratios_i.append(s["ratio_i"])
            for g in regimes:
                if s[g]:
                    side_pass[g] += 1
                else:
                    case_bad[g].add(idx)
            fmax = max(fmax, s["forced_i"])
            fsum += s["forced_i"]
            pre_state[s["pre"]["state"]] += 1
        forced_max[idx] = fmax
        forced_sum[idx] = fsum
        joint_mult.append(r["joint_ii"])
        if r["joint_ii"] > 1 + 1e-9:
            joint_ii_fail.add(idx)
        for ax, ext in r["axis_forced"].items():
            pre_axis_forced[ax] += 1
            perp = ("T", "B") if ax == "W" else ("L", "R")
            for side in perp:
                if r["sides"][side]["nreq"]:
                    pre_axis_tight.append((idx, ax, side,
                                           r["D"][side], ext,
                                           r["D"][side] / max(ext, 1e-12)))
        if r["corner_bad"]:
            corner_cases.append((idx, r["corner_bad"]))

    name = Path(a.results).name
    NC = len(case_tot)
    print(f"\n{'=' * 78}")
    print(f"L144 boundary FEASIBILITY probe   {name}   "
          f"sample={a.sample or 'auto'}")
    print(f"{'=' * 78}")
    print(f"cases with >=1 boundary-constrained block : {NC} / {len(per_case)}")
    print(f"constrained (case,side) pairs             : {side_tot}")
    tot_vbnd = sum(per_case[i][0]['v_bnd'] for i in case_tot)
    vcases = {i for i in case_tot if per_case[i][0]["v_bnd"] > 0}
    print(f"cases with v_bnd > 0                      : {len(vcases)}"
          f"   (total v_bnd = {tot_vbnd})")

    # ---------------- regime table ---------------- #
    print(f"\n--- A. edge-capacity feasibility by regime "
          f"(necessary conditions only) ---\n")
    print(f"{'regime':<28} {'sides OK':>12} {'sides INFEAS':>13} "
          f"{'cases INFEAS':>13}")
    lbl = {"i": "(i)   achieved / achieved",
           "ii": "(ii)  achieved / free frame"}
    for R in ASPECTS:
        lbl[f"iii_{R}"] = f"(iii) aspect {R:<4g} / achieved"
        lbl[f"iv_{R}"] = f"(iv)  aspect {R:<4g} / free frame"
    for g in regimes:
        ok = side_pass[g]
        print(f"{lbl[g]:<28} {ok:>12} {side_tot - ok:>13} "
              f"{len(case_bad[g]):>13}")
    print(f"\n(ii) JOINT test  max(D_L,D_R,hmax)*max(D_T,D_B,wmax) <= A_bb : "
          f"{NC - len(joint_ii_fail)}/{NC} cases pass")
    jm = sorted(joint_mult)
    print(f"     frame-area multiplier needed: median {st.median(jm):.4f}"
          f"   p90 {jm[int(0.9 * len(jm))]:.4f}   max {max(jm):.4f}")
    ri = sorted(ratios_i)
    print(f"\ndemand/capacity, regime (i), all {side_tot} sides: "
          f"median {st.median(ri):.3f}  p90 {ri[int(0.9 * len(ri))]:.3f}  "
          f"p99 {ri[int(0.99 * len(ri))]:.3f}  max {max(ri):.3f}")

    # ---------------- headline cross-tab ---------------- #
    print(f"\n--- B. HEADLINE: actual v_bnd vs regime-(i) infeasibility ---\n")

    def xtab(bad, title):
        a11 = len(vcases & bad)
        a10 = len(vcases - bad)
        a01 = len(bad - vcases)
        a00 = NC - a11 - a10 - a01
        vb_in = sum(per_case[i][0]["v_bnd"] for i in (vcases & bad))
        vb_out = sum(per_case[i][0]["v_bnd"] for i in (vcases - bad))
        wt_in = sum(W_of[i] for i in (vcases & bad))
        wt_out = sum(W_of[i] for i in (vcases - bad))
        print(f"  {title}")
        print(f"    {'':<22}{'edge INFEASIBLE':>17}{'edge feasible':>15}")
        print(f"    {'v_bnd > 0':<22}{a11:>17}{a10:>15}")
        print(f"    {'v_bnd = 0':<22}{a01:>17}{a00:>15}")
        print(f"    v_bnd mass                {vb_in:>13} "
              f"{vb_out:>14}   ({100 * vb_out / max(tot_vbnd, 1):.1f}% of all "
              f"v_bnd is on cases with NO capacity obstruction)")
        print(f"    exp(n/12) weight          {wt_in:>13.1f} "
              f"{wt_out:>14.1f}   "
              f"({100 * wt_out / max(wt_in + wt_out, 1e-9):.1f}% of the "
              f"violating weight)\n")

    xtab(case_bad["i"], "regime (i)  achieved shapes, achieved frame")
    xtab(case_bad["ii"], "regime (ii) achieved shapes, free frame")
    xtab(case_bad["iii_2.5"], "regime (iii) aspect 2.5 (= shipped LR/TB), "
                              "achieved frame")

    lb_max = sum(forced_max.values())
    lb_sum = sum(forced_sum.values())
    print(f"  capacity-FORCED violations, regime (i):")
    print(f"    lower bound  sum_case max_side forced   = {lb_max:>5}"
          f"   ({100 * lb_max / max(tot_vbnd, 1):.2f}% of v_bnd {tot_vbnd})")
    print(f"    upper-ish    sum_case sum_side forced   = {lb_sum:>5}"
          f"   ({100 * lb_sum / max(tot_vbnd, 1):.2f}% of v_bnd {tot_vbnd})")

    # ---------------- preplaced ---------------- #
    print(f"\n--- C. preplaced over-constraint (L135 generalised) ---\n")
    print(f"  constrained sides whose bbox coordinate is set by a preplaced "
          f"required block:")
    print(f"    free (no preplaced block requires this side) : "
          f"{pre_state['free']:>5}")
    print(f"    FORCED  (coordinate pinned, satisfiable)     : "
          f"{pre_state['forced']:>5}")
    print(f"    OVER-CONSTRAINED, two required preplaced disagree : "
          f"{pre_state['conflict']:>5}")
    print(f"    OVER-CONSTRAINED, another preplaced sits beyond   : "
          f"{pre_state['dominated']:>5}")
    ovc = pre_state["conflict"] + pre_state["dominated"]
    print(f"    -> provably impossible sides from preplaced ALONE : "
          f"{ovc} / {side_tot}")
    print(f"\n  axis extent fully pinned by preplaced on BOTH ends: "
          f"W {pre_axis_forced['W']}   H {pre_axis_forced['H']} (of {NC} cases)")
    if pre_axis_tight:
        over = [t for t in pre_axis_tight if t[5] > 1 + 1e-9]
        print(f"    perpendicular sides tested against the pinned extent: "
              f"{len(pre_axis_tight)}   over capacity: {len(over)}")
        for t in sorted(pre_axis_tight, key=lambda t: -t[5])[:6]:
            print(f"      case {t[0]:>4} axis {t[1]} side {t[2]}  "
                  f"demand {t[3]:>10.3f}  pinned extent {t[4]:>10.3f}  "
                  f"ratio {t[5]:.3f}")
    print(f"\n  corner conflicts (>=2 blocks required at the SAME corner): "
          f"{len(corner_cases)} cases")
    for idx, cb in corner_cases[:6]:
        print(f"      case {idx:>4} {cb}")

    # ---------------- regime (0): pinned frame ---------------- #
    print(f"\n--- D. regime (0): capacity of the frame a COMPLIANT layout is "
          f"pinned to ---\n")
    print("  When preplaced boundary blocks pin BOTH ends of an axis, the")
    print("  perpendicular capacity is EXACT and is not the achieved one: the")
    print("  placer can only exceed it by breaking one of the pins (case 85).")
    npin = sum(1 for idx in case_tot for _b, sd in BITS
               if per_case[idx][1]["sides"][sd]["nreq"]
               and per_case[idx][1]["sides"][sd]["pinned"])
    f_ach = f_a25 = f_hard = 0
    bad_pin = []
    forced_pin_tot = forced_pre_tot = 0
    for idx in sorted(case_tot):
        c, r = per_case[idx]
        forced_pre_tot += r["forced_pre"]
        worst = 0
        for _b, sd in BITS:
            s = r["sides"][sd]
            if not (s["nreq"] and s["pinned"]):
                continue
            f_ach += 0 if s["pin_ach"] else 1
            f_a25 += 0 if s["pin_a25"] else 1
            f_hard += 0 if s["pin_hard"] else 1
            if not s["pin_ach"]:
                bad_pin.append((idx, c["n"], sd, s["nreq"], s["cap_pin"],
                                r["D"][sd], r["Dhard"][sd],
                                r["Dfree"][2.5][sd], s["forced_pin"],
                                c["v_bnd"]))
                worst = max(worst, s["forced_pin"])
        forced_pin_tot += worst
    print(f"\n  constrained sides sitting on a pinned axis : {npin} / {side_tot}")
    print(f"    demand > pinned capacity, achieved shapes      : {f_ach}")
    print(f"    demand > pinned capacity, shipped aspect 2.5   : {f_a25}")
    print(f"    demand > pinned capacity, RULES ONLY           : {f_hard}")
    print("      (rules-only counts just fixed-shape/preplaced blocks: the")
    print("       contest bounds area to +-1% but never bounds ASPECT, so a")
    print("       movable soft block's edge extent has no positive floor)")
    if bad_pin:
        print(f"\n{'case':>5} {'n':>4} {'sd':>3} {'req':>4} {'cap_pin':>9} "
              f"{'D_ach':>9} {'D_a2.5':>8} {'D_hard':>8} {'frcd':>5} "
              f"{'v_bnd':>6}")
        for t in sorted(bad_pin, key=lambda t: -(t[5] / t[4])):
            print(f"{t[0]:>5} {t[1]:>4} {t[2]:>3} {t[3]:>4} {t[4]:>9.3f} "
                  f"{t[5]:>9.3f} {t[7]:>8.3f} {t[6]:>8.3f} {t[8]:>5} "
                  f"{t[9]:>6}")
    print(f"\n  PROVABLY-FORCED boundary violations on held-out s1:")
    print(f"    from preplaced coordinate conflicts (rules only) : "
          f"{forced_pre_tot:>4}")
    print(f"    from pinned-axis capacity, achieved shapes       : "
          f"{forced_pin_tot:>4}")
    print(f"    from regime (i) achieved-frame capacity          : "
          f"{lb_max:>4}")
    print(f"    ---------------------------------------------------")
    print(f"    union upper estimate                             : "
          f"{forced_pre_tot + forced_pin_tot + lb_max:>4}"
          f"   of v_bnd {tot_vbnd}   "
          f"({100 * (forced_pre_tot + forced_pin_tot + lb_max) / max(tot_vbnd, 1):.2f}%)")

    # ---------------- worst sides ---------------- #
    print(f"\n--- E. the {a.show} worst constrained sides, regime (i) ---\n")
    rows = []
    for idx in case_tot:
        c, r = per_case[idx]
        for _b, side in BITS:
            s = r["sides"][side]
            if s["nreq"]:
                rows.append((s["ratio_i"], idx, c["n"], side, s["nreq"],
                             s["miss"], s["forced_i"], s["ratio_ii"],
                             s[f"ratio_iii_2.5"], c["v_bnd"],
                             s["pre"]["state"]))
    rows.sort(reverse=True)
    print(f"{'case':>5} {'n':>4} {'sd':>3} {'req':>4} {'miss':>5} {'frcd':>5} "
          f"{'D/cap(i)':>9} {'D/cap(ii)':>10} {'D/cap a2.5':>11} "
          f"{'v_bnd':>6} {'preplaced':>11}")
    for t in rows[:a.show]:
        print(f"{t[1]:>5} {t[2]:>4} {t[3]:>3} {t[4]:>4} {t[5]:>5} {t[6]:>5} "
              f"{t[0]:>9.3f} {t[7]:>10.3f} {t[8]:>11.3f} {t[9]:>6} "
              f"{t[10]:>11}")

    # ---------------- miss vs fit ---------------- #
    print(f"\n--- F. D/capacity split by whether the side actually MISSED ---\n")
    hit = [t[0] for t in rows if t[5] == 0]
    mis = [t[0] for t in rows if t[5] > 0]
    for nm, v in (("sides with 0 misses", hit), ("sides WITH misses", mis)):
        v = sorted(v)
        print(f"  {nm:<22} {len(v):>5}   median {st.median(v):.3f}   "
              f"p90 {v[int(0.9 * len(v))]:.3f}   max {max(v):.3f}   "
              f"over capacity {sum(1 for x in v if x > 1 + 1e-9)}")
    print(f"  side-misses total: {sum(t[5] for t in rows)}  "
          f"(a corner block can miss on two sides, so this exceeds v_bnd)")

    # ---------------- preplaced conflict detail ---------------- #
    print(f"\n--- G. the preplaced-over-constrained sides, one line each ---\n")
    print(f"{'case':>5} {'n':>4} {'sd':>3} {'state':>10} {'npre_req':>9} "
          f"{'coords required to coincide':<34} {'v_bnd':>6}")
    for idx in sorted(case_tot):
        c, r = per_case[idx]
        n, cons, tp = c["n"], c["cons"], c["tp"]
        for bit, side in BITS:
            p = r["sides"][side]["pre"]
            if p["state"] not in ("conflict", "dominated"):
                continue
            rq = [i for i in range(n)
                  if float(cons[i][1]) != 0 and int(cons[i][4]) & bit]
            f = {1: lambda i: tp[i][0], 2: lambda i: tp[i][0] + tp[i][2],
                 4: lambda i: tp[i][1] + tp[i][3], 8: lambda i: tp[i][1]}[bit]
            vs = ", ".join(f"blk{i}={f(i):g}" for i in rq)
            print(f"{idx:>5} {c['n']:>4} {side:>3} {p['state']:>10} "
                  f"{p['nreq']:>9} {vs:<34} {c['v_bnd']:>6}")

    # ---------------- shape-lever ceiling ---------------- #
    print(f"\n--- H. what the aspect lever buys on the sides that DO fail ---\n")
    fail_i = [t for t in rows if t[0] > 1 + 1e-9]
    print(f"  sides failing regime (i): {len(fail_i)}")
    for R in ASPECTS:
        still = 0
        for idx in case_tot:
            c, r = per_case[idx]
            for _b, side in BITS:
                s = r["sides"][side]
                if s["nreq"] and not s["i"] and not s[f"iii_{R}"]:
                    still += 1
        print(f"    aspect {R:<4g}: still infeasible {still:>4} / "
              f"{len(fail_i)}   ({100 * (len(fail_i) - still) / max(len(fail_i), 1):.1f}% rescued)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
