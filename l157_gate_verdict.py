"""OFFLINE (never shipped): adjudicate the L157 selective-depth implementation.

ROUND 1 found two things. One was my error, one was real.

  * G1 failed against results_L147_r15g.json -- and that anchor is STALE. The
    same arm against results_L154_catchoff.json is 100/100 cost AND positions,
    and the two references differ from EACH OTHER on 50/100. catchoff is also
    what L157's offline pricing used as its quality baseline, so it is the
    anchor. Two files that both look like "the L147 arm at k=1" are not
    interchangeable; pick the one the pricing used.

  * G3c failed for real: the gate bought the second pass on 0/100 cases. The
    gate is stated in ABSOLUTE seconds (t_case + dt <= 0.3046 * M_hat(n)),
    which is correct on the grader -- R = t/M is measured on whatever box runs
    the case -- but this box runs ~9x slower per case (490.7s against beta's
    52.07s over the same 100 cases). Measured t_case/budget: min 3.0x, p50
    5.6x, max 20.7x, 0/100 inside. So the mechanism cannot be exercised here
    at S=1, and ICCAD_SHAPE_LP_DEPTH_S exists to make it exercisable.

Usage:  <python> -u l157_gate_verdict.py
"""
import json
import math
import pathlib
import sys

D = pathlib.Path(__file__).resolve().parent
W = lambda n: math.exp(n / 12.0)                                     # noqa: E731
K1 = "results_L154_catchoff.json"        # the anchor the pricing used


def load(name):
    f = D / name
    return {r["test_id"]: r for r in json.load(open(f))["test_results"]} if f.exists() else None


def stats(tag):
    p = D / f"l157_{tag}_stats.txt"
    if not p.exists():
        return None
    out = []
    for ln in p.read_text().splitlines():
        f = ln.split()
        if len(f) >= 4:
            out.append((int(f[0]), int(f[1]), int(f[2]), int(f[3])))
    return out


def biteq(a, b):
    ids = sorted(set(a) & set(b))
    c = sum(1 for i in ids if a[i]["cost"] == b[i]["cost"])
    p = sum(1 for i in ids if a[i]["positions"] == b[i]["positions"])
    return c == len(ids) and p == len(ids), f"cost {c}/{len(ids)}  positions {p}/{len(ids)}"


def wcost(q):
    return sum(W(r["block_count"]) * r["cost"] for r in q.values())


def main():
    ok_all = True

    def gate(name, ok, msg):
        nonlocal ok_all
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {msg}")

    print(__doc__.split("Usage:")[0].rstrip())
    print("=" * 74)
    ref = load(K1)
    arms = {t: load(f"results_L157_{t}.json")
            for t in ("depthoff", "k2", "gated", "k1b", "notan", "gateS")}

    print("\n--- G1/G2 the two references the implementation must not move ---")
    if arms["depthoff"]:
        gate("G1  round-1 kill switch == L154 catchoff", *biteq(arms["depthoff"], ref))
    if arms["k1b"]:
        gate("G1b round-2 kill switch == L154 catchoff (after the edits)",
             *biteq(arms["k1b"], ref))
    if arms["k2"]:
        gate("G2  ungated k=2 == L148 lp2", *biteq(arms["k2"], load("results_L148_lp2.json")))

    print("\n--- G3 the coupling: no ICCAD_* set must not run the unpriced arm ---")
    s = stats("notan")
    if s:
        h = {}
        for r in s:
            h[r[3]] = h.get(r[3], 0) + 1
        gate("G3 tangent OFF spends exactly 1 pass", set(h) == {1},
             f"{h} -- depth 2 was only ever priced WITH the tangent rows")

    print("\n--- G4 the mechanism, exercised at grader-like speed (S=7.75) ---")
    s = stats("gateS")
    if s and arms["gateS"]:
        h = {}
        for r in s:
            h[r[3]] = h.get(r[3], 0) + 1
        two = h.get(2, 0)
        gate("G4a the gate discriminates", {1, 2} <= set(h),
             f"{two}/{len(s)} cases bought the second pass ({100*two/len(s):.0f}%; "
             f"priced fraction 75%)")
        b = wcost(ref)
        q = 100 * (b - wcost(arms["gateS"])) / b
        qk = 100 * (b - wcost(arms["k2"])) / b if arms["k2"] else float("nan")
        gate("G4b quality positive and bounded by ungated k=2",
             0 < q <= qk + 1e-9, f"gated {q:+.4f}%  ungated {qk:+.4f}%  "
             f"({100*q/qk:.0f}% of the ceiling on {100*two/len(s):.0f}% of the cases)")
        moved = [i for i in arms["gateS"] if arms["gateS"][i]["cost"] != ref[i]["cost"]]
        worse = [i for i in moved if arms["gateS"][i]["cost"] > ref[i]["cost"]]
        gate("G4c no case got worse", not worse,
             f"{len(moved)}/{len(ref)} moved, {len(worse)} worse")
        bad = [i for i, r in arms["gateS"].items() if not r.get("is_feasible", True)]
        gate("G4d feasible", not bad, f"{len(arms['gateS'])-len(bad)}/{len(arms['gateS'])}")

    print("\n--- G5 what is NOT established here ---")
    print("      The grader's per-case ordering. Our slowdown is n-dependent")
    print("      (3.0x at the heavy end, 20.7x at the light end), so no single S")
    print("      reproduces which cases the grader would pick. G4 shows the")
    print("      mechanism is correct and monotone; the SELECTION still rests on")
    print("      beta's RF = 0.70004 (at floor) as evidence that the grader sits")
    print("      inside the budget. That is inference, not measurement.")

    print("\n" + "=" * 74)
    print("VERDICT: " + ("ALL RUN GATES PASS" if ok_all else "AT LEAST ONE GATE FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
