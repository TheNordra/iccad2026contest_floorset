"""L199 verdict -- in-set gates for the L196 tree (flat depth map + per-case LP gate).

Seven arms, eight checks. Three of them assert things that were IMPOSSIBLE to
assert before L196, and two invert checks that L172/L177 ran in the other
direction:

  G1  DETERMINISM     det1 == det2 on cost AND positions, 100/100.
  G2  L147 KILL       ICCAD_SHAPE_LP_L147=0 reproduces the committed pre-L147
                      band bit-for-bit.
  G3  THE GATE FIRED  the SET of block counts the LP actually ran on must equal
                      _L196_LPGATE's 1-set exactly -- not merely have the right
                      size. ICCAD_LP_GATE=0 must widen it to all 100, and
                      ICCAD_SHAPE_LP=0 must empty it. A table that silently
                      kept old values passes determinism and the kill switch
                      while changing nothing; the stats line count is the only
                      thing that can tell them apart.
  G4  THE MAP IS FLAT inverted from L172's G4. _L157_DEPTH is all 1s, so
                      ICCAD_SHAPE_LP_DEPTH2=0 must be a NO-OP: k1 identical to
                      det1 on cost and positions, and no case may spend >1 pass.
  G5  FEASIBILITY     every arm 100/100 feasible, no errors.
  G6  GATE VALUE      det1 vs gateoff -- what the gate gives up in quality to
                      buy back the wall. Informational: the sign is expected
                      negative, the trade is priced in seconds elsewhere.
  G7  LP VALUE        det1 vs lpoff -- the total in-set quality of the gated LP.
                      Must be positive or the whole mechanism is pointless.
  G8  HB PREDICTOR    hboff vs det1 -- L171 in THIS configuration. Its sign
                      already flipped once when the map changed underneath it.

  <python> l199_verdict.py
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

DIR = Path(__file__).parent
PREFIX = "L199"
ANCHOR = None          # optional: a prefix every arm must reproduce bit-for-bit


def load(tag):
    f = DIR / "results_{}_{}.json".format(PREFIX, tag)
    if not f.exists():
        return None
    return {r["test_id"]: r for r in json.load(open(f))["test_results"]}


def stats(tag):
    """[(n, kept, tier, passes_spent)] from ICCAD_SHAPE_LP_STATS.

    A MISSING file is ambiguous and the two readings are opposite: either the
    LP never executed (which for the lpoff arm is the correct outcome and the
    proof we want) or the arm never ran at all. Disambiguated by the results
    JSON -- if the arm produced results and no stats file, the LP genuinely
    never ran, which is 0 executions and not "not run". The first version of
    this file returned None here and reported G3 NOT RUN on a passing package,
    which is the same false-negative the gate exists to prevent, pointed the
    other way."""
    f = DIR / "{}_{}_stats.txt".format(PREFIX.lower(), tag)
    if not f.exists():
        return [] if (DIR / "results_{}_{}.json".format(PREFIX, tag)).exists() \
            else None
    out = []
    for line in f.read_text().split("\n"):
        p = line.split()
        if len(p) >= 4:
            out.append(tuple(int(x) for x in p[:4]))
    return out


def wq(d):
    """The project's weighted cost: exp(n/12) weighting, as every L1xx table."""
    num = sum(math.exp(r["block_count"] / 12.0) * r["cost"] for r in d.values())
    den = sum(math.exp(r["block_count"] / 12.0) for r in d.values())
    return num / den


def delta(a, b, label, note=""):
    """quality of b relative to a, in percent (positive = b is better)."""
    ids = sorted(set(a) & set(b))
    A = {i: a[i] for i in ids}
    B = {i: b[i] for i in ids}
    q = 100 * (wq(A) - wq(B)) / wq(A)
    mv = sum(1 for i in ids if B[i]["cost"] != A[i]["cost"])
    ws = sum(1 for i in ids if B[i]["cost"] > A[i]["cost"] + 1e-12)
    bt = sum(1 for i in ids if B[i]["cost"] < A[i]["cost"] - 1e-12)
    print("{}   {:+.4f}%   {} moved ({} better / {} worse) of {}"
          .format(label, q, mv, bt, ws, len(ids)))
    if note:
        print("                 " + note)
    return q


def identical(a, b, label):
    ids = sorted(set(a) & set(b))
    c = sum(1 for i in ids if a[i]["cost"] == b[i]["cost"])
    p = sum(1 for i in ids if a[i].get("positions") == b[i].get("positions"))
    ok = len(ids) > 0 and c == len(ids) and p == len(ids)
    print("{}  cost {}/{}  positions {}/{}   {}"
          .format(label, c, len(ids), p, len(ids), "PASS" if ok else "FAIL"))
    return ok


def main():
    global PREFIX, ANCHOR
    if len(sys.argv) > 1:
        PREFIX = sys.argv[1]
    if len(sys.argv) > 2:
        ANCHOR = sys.argv[2]
    src = (DIR / "optimizer_constructive.py").read_text(encoding="utf-8")
    DEPTH = eval(re.search(r"^_L157_DEPTH = \{.*?^\}", src, re.S | re.M)
                 .group(0).split("=", 1)[1])
    GATE = eval(re.search(r"^_L196_LPGATE = \{.*?^\}", src, re.S | re.M)
                .group(0).split("=", 1)[1])
    want_on = {n for n, v in GATE.items() if v}

    arms = {t: load(t) for t in ("det1", "det2", "gateoff", "k1",
                                 "l147off", "hboff", "lpoff", "pooldropoff")}
    fails = []

    def need(*tags):
        return [t for t in tags if not arms.get(t)]

    print("=" * 72)
    print("L199 -- in-set gates, L196 tree "
          "(depth {}, LP gate {} on / {} off)"
          .format(dict(sorted(Counter(DEPTH.values()).items())),
                  len(want_on), len(GATE) - len(want_on)))
    print("=" * 72)

    # ---- G1 determinism -----------------------------------------------------
    m = need("det1", "det2")
    if m:
        fails.append("G1(not run)")
        print("G1 determinism   NOT RUN (missing {})".format(",".join(m)))
    elif not identical(arms["det1"], arms["det2"], "G1 determinism  "):
        fails.append("G1")

    # ---- G2 L147 kill switch ------------------------------------------------
    # results_L165_l147off.json was produced BEFORE the LP gate existed, when
    # ICCAD_SHAPE_LP_L147=0 still ran the LP on all 100 cases. Under L196 it
    # runs on 63, so a flat 100/100 bit-compare against that anchor reads FAIL
    # on a package where nothing is wrong -- a stale anchor, not a broken
    # escape hatch. Repointed, not relaxed, into three parts that together say
    # strictly MORE than the original did:
    #
    #   G2a  on the 63 block counts where the LP still runs, l147off must
    #        reproduce the anchor bit-for-bit. The hatch itself is unchanged.
    #   G2b  on the 37 where it does not, l147off must equal the no-LP arm
    #        bit-for-bit -- positive proof that "the LP was skipped" means the
    #        untouched layout and not some third thing.
    #   G2c  DECISIVE: l147off WITH the gate killed must reproduce the anchor
    #        100/100. With L147 off the depth is k=1 in both trees
    #        (_shape_lp_depth gates depth>=2 on tangent_on), so that arm is
    #        exactly the L165 configuration and has no excuse to differ.
    # L216: results_L165_l147off.json was produced with the full 51-profile
    # pool. Since L211/L213 the shipped pool drops 8 per block count, so the
    # portfolio winner can change for reasons that have nothing to do with the
    # L147 hatch -- G2a/G2c read FAIL on a package where the hatch is fine, and
    # the 12 cases they differ on are EXACTLY the 12 the drop moves (measured,
    # both set differences empty). Prefer the arm in the anchor's own pool
    # configuration when it exists; fall back to the old arm otherwise.
    ref = DIR / "results_L165_l147off.json"
    both = load("l147off_gateoff_nodrop") or load("l147off_gateoff")
    if need("l147off") or not ref.exists():
        fails.append("G2(not run)")
        print("G2 L147 kill     NOT RUN")
    else:
        R = {r["test_id"]: r for r in json.load(open(ref))["test_results"]}
        off = arms["l147off"]
        ids = sorted(set(off) & set(R))
        # G2a is also pool-dependent; when the nodrop arm is present, G2c is the
        # decisive one and G2a is reported for information only.
        _has_nodrop = load("l147off_gateoff_nodrop") is not None
        on_ids = [i for i in ids if GATE.get(off[i]["block_count"], 1)]
        sk_ids = [i for i in ids if not GATE.get(off[i]["block_count"], 1)]
        a_same = sum(1 for i in on_ids if off[i]["cost"] == R[i]["cost"])
        ok_a = len(on_ids) > 0 and a_same == len(on_ids)
        print("G2a L147 hatch   {}/{} identical to results_L165_l147off.json on "
              "the block counts the gate KEEPS   {}"
              .format(a_same, len(on_ids),
                      ("INFO (pool-dependent; G2c decides)" if _has_nodrop
                       else ("PASS" if ok_a else "FAIL"))))

        if arms.get("lpoff"):
            no = arms["lpoff"]
            b_same = sum(1 for i in sk_ids
                         if off[i]["cost"] == no[i]["cost"]
                         and off[i].get("positions") == no[i].get("positions"))
            ok_b = len(sk_ids) > 0 and b_same == len(sk_ids)
            print("G2b skipped==no-LP {}/{} identical to the lpoff arm on the "
                  "block counts the gate DROPS   {}"
                  .format(b_same, len(sk_ids), "PASS" if ok_b else "FAIL"))
        else:
            ok_b = False
            print("G2b skipped==no-LP NOT RUN (no lpoff arm)")

        if both:
            ids2 = sorted(set(both) & set(R))
            c_same = sum(1 for i in ids2 if both[i]["cost"] == R[i]["cost"])
            ok_c = len(ids2) > 0 and c_same == len(ids2)
            print("G2c L147+gateoff {}/{} identical to results_L165_l147off.json"
                  "   {}".format(c_same, len(ids2), "PASS" if ok_c else "FAIL"))
        else:
            ok_c = None
            print("G2c L147+gateoff NOT RUN -- the decisive arm "
                  "(ICCAD_SHAPE_LP_L147=0 ICCAD_LP_GATE=0) is still pending")

        ok = (ok_c if _has_nodrop else ok_a) and ok_b and (ok_c is not False)
        fails += [] if ok else ["G2"]
        if ok_c is None:
            print("                 G2 is PROVISIONAL until G2c lands.")

    # ---- G3 the gate actually fired ----------------------------------------
    s_on, s_off, s_no = stats("det1"), stats("gateoff"), stats("lpoff")
    if s_on is None or s_off is None or s_no is None:
        fails.append("G3(not run)")
        print("G3 gate fired    NOT RUN (missing a stats file)")
    else:
        got_on = {n for n, _k, _t, _p in s_on}
        got_off = {n for n, _k, _t, _p in s_off}
        ok_on = got_on == want_on
        ok_off = got_off == set(GATE)
        ok_no = len(s_no) == 0
        ok = ok_on and ok_off and ok_no
        fails += [] if ok else ["G3"]
        print("G3 gate fired    default   LP ran on {:3d} block counts   "
              "set == table: {}".format(len(got_on), "YES" if ok_on else "NO"))
        print("                 LP_GATE=0  LP ran on {:3d} block counts   "
              "== all 100: {}".format(len(got_off), "YES" if ok_off else "NO"))
        print("                 SHAPE_LP=0 LP ran on {:3d} block counts   "
              "== none:    {}".format(len(s_no), "YES" if ok_no else "NO"))
        if not ok_on:
            print("                 !! ran-but-shouldnt {}   shouldve-but-didnt {}"
                  .format(sorted(got_on - want_on)[:10],
                          sorted(want_on - got_on)[:10]))
        print("                 {}".format("PASS" if ok else "FAIL"))
        print("                 n>100 with the LP on: {} (s=1.0 keeps only 2)"
              .format(sum(1 for n in got_on if n > 100)))

    # ---- G4 the depth map is flat ------------------------------------------
    if need("det1", "k1") or s_on is None:
        fails.append("G4(not run)")
        print("G4 map is flat   NOT RUN")
    else:
        ok_id = identical(arms["det1"], arms["k1"], "G4 map is flat  ")
        hist = dict(sorted(Counter(p for _n, _k, _t, p in s_on).items()))
        ok_h = set(hist) <= {1}
        ok = ok_id and ok_h
        fails += [] if ok else ["G4"]
        print("                 passes spent {}   (all-1s map => must be "
              "{{1: N}})   {}".format(hist, "PASS" if ok_h else "FAIL"))
        print("                 DEPTH2=0 is a no-op here BY CONSTRUCTION -- "
              "that is the assertion, not a null result.")

    # ---- G5 feasibility -----------------------------------------------------
    bad = []
    for t, d in arms.items():
        if not d:
            continue
        nf = sum(1 for r in d.values() if not r.get("is_feasible", True))
        er = sum(1 for r in d.values() if r.get("error"))
        if nf or er:
            bad.append("{}({} infeasible, {} errors)".format(t, nf, er))
    ran = [t for t, d in arms.items() if d]
    if len(ran) < 7:
        fails.append("G5(not run)")
        print("G5 feasibility   NOT RUN -- only {} of 7 arms present: {}"
              .format(len(ran), ",".join(sorted(ran))))
    else:
        ok = not bad
        fails += [] if ok else ["G5"]
        print("G5 feasibility   {}   {}".format(
            "100/100 in all 7 arms" if ok else "; ".join(bad),
            "PASS" if ok else "FAIL"))

    # ---- G6/G7/G8 quality ---------------------------------------------------
    if arms.get("gateoff") and arms.get("det1"):
        delta(arms["gateoff"], arms["det1"], "G6 gate cost    ",
              "vs the ungated LP. Negative is EXPECTED: the gate trades "
              "quality for wall.")
    else:
        fails.append("G6(not run)")
        print("G6 gate cost     NOT RUN")

    if arms.get("lpoff") and arms.get("det1"):
        q = delta(arms["lpoff"], arms["det1"], "G7 LP value     ",
                  "the whole gated LP, vs no LP at all. Must be > 0.")
        if q <= 0:
            fails.append("G7")
            print("                 !! the gated LP is not worth its own "
                  "existence in set")
    else:
        fails.append("G7(not run)")
        print("G7 LP value      NOT RUN")

    if arms.get("hboff") and arms.get("det1"):
        delta(arms["hboff"], arms["det1"], "G8 hb predictor ",
              "L171 on THIS configuration. It read -0.0512% on L172's map.")
    else:
        fails.append("G8(not run)")
        print("G8 hb predictor  NOT RUN")

    # ---- G10: the L211/L213 pool drop actually fired -----------------------
    # Two halves, and the second is the one that catches a table that silently
    # kept its old values: the KILL SWITCH arm must reproduce the pre-drop
    # anchor bit-for-bit, and the default must differ from it on exactly the
    # cases the drop was measured to move. A table that never loaded would pass
    # the first half and fail the second; a table that dropped everything would
    # pass the second and fail the first.
    POOLDROP_ANCHOR = "results_L209_det1.json"
    POOLDROP_MOVED = 12                      # measured in set at k=8
    ref2 = DIR / POOLDROP_ANCHOR
    if arms.get("pooldropoff") and arms.get("det1") and ref2.exists():
        R2 = {r["test_id"]: r for r in json.load(open(ref2))["test_results"]}
        off2 = arms["pooldropoff"]
        ids = sorted(set(off2) & set(R2))
        same = sum(1 for i in ids if off2[i]["cost"] == R2[i]["cost"]
                   and off2[i].get("positions") == R2[i].get("positions"))
        ok_a = len(ids) > 0 and same == len(ids)
        d1 = arms["det1"]
        ids2 = sorted(set(d1) & set(R2))
        mv = sum(1 for i in ids2 if d1[i]["cost"] != R2[i]["cost"])
        ok_b = mv == POOLDROP_MOVED
        try:
            src2 = src
            tbl = eval(re.search(r"^_L211_POOLDROP = \{.*?^\}", src2, re.S | re.M)
                       .group(0).split("=", 1)[1])
            ok_c = len(tbl) == 100 and all(len(v) == 8 for v in tbl.values())
        except Exception:
            tbl, ok_c = {}, False
        ok = ok_a and ok_b and ok_c
        fails += [] if ok else ["G10"]
        print("G10 pool drop    kill switch {}/{} identical to {}   {}"
              .format(same, len(ids), POOLDROP_ANCHOR, "PASS" if ok_a else "FAIL"))
        print("                 default moves {} cases vs it (measured {})   {}"
              .format(mv, POOLDROP_MOVED, "PASS" if ok_b else "FAIL"))
        print("                 table: {} block counts x {} profiles   {}"
              .format(len(tbl), len(next(iter(tbl.values()))) if tbl else 0,
                      "PASS" if ok_c else "FAIL"))
    else:
        fails.append("G10(not run)")
        print("G10 pool drop    NOT RUN")

    # ---- G9: every arm must reproduce the anchor prefix bit-for-bit --------
    # Used when the change under test is supposed to move WALL and nothing else
    # (route A off, L205). A result difference here would mean the mechanism was
    # never result-neutral, which is a much bigger finding than the wall saving.
    if ANCHOR:
        bad, seen = [], 0
        for t, d in sorted(arms.items()):
            if not d:
                continue
            f = DIR / "results_{}_{}.json".format(ANCHOR, t)
            if not f.exists():
                continue
            R = {r["test_id"]: r for r in json.load(open(f))["test_results"]}
            ids = sorted(set(d) & set(R))
            c = sum(1 for i in ids if d[i]["cost"] == R[i]["cost"])
            pp = sum(1 for i in ids if d[i].get("positions") == R[i].get("positions"))
            seen += 1
            if c != len(ids) or pp != len(ids):
                bad.append("{} (cost {}/{}, pos {}/{})"
                           .format(t, c, len(ids), pp, len(ids)))
        if not seen:
            fails.append("G9(not run)")
            print("G9 vs {}      NOT RUN -- no anchor arms found".format(ANCHOR))
        else:
            ok = not bad
            fails += [] if ok else ["G9"]
            print("G9 vs {}      {} arm(s) compared, all bit-identical on cost "
                  "AND positions   {}".format(ANCHOR, seen,
                                              "PASS" if ok else "FAIL"))
            if bad:
                print("                 !! " + "; ".join(bad))

    # ---- cross-session bonus: det1 vs the previous session's default run ----
    prev = DIR / "_l198_gateon.json"
    if prev.exists() and arms.get("det1"):
        P = {r["test_id"]: r for r in json.load(open(prev))["test_results"]}
        ids = sorted(set(P) & set(arms["det1"]))
        c = sum(1 for i in ids if P[i]["cost"] == arms["det1"][i]["cost"])
        print("-" * 72)
        print("   cross-session: det1 vs _l198_gateon.json  cost {}/{}  {}"
              .format(c, len(ids), "identical" if c == len(ids) else "DIFFERS"))

    print("=" * 72)
    # A gate that reports PASS because nothing ran is the same silent no-op
    # this line of work exists to prevent, so "not run" counts as a failure.
    print("VERDICT: {}".format("ALL PASS" if not fails
                               else "FAIL " + ",".join(sorted(set(fails)))))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
