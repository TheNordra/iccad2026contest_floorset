# L342 — §5(a) re-run against RF-SAFE, and why it does not bind

**Verdict: §5(a) is real, was never addressed, and does NOT bind RF-SAFE — because
the instrument it was written on over-estimates our grader runtime against ground
truth, in exactly the direction that makes every widening look unaffordable.**

## The charge

`HANDOFF_2026-08-27.md` §5(a):

> **(a) Do not re-widen the LP gate on the back of the LP speedup.** … Scored
> properly — both sides at the same (rb, f) — **every widening candidate is
> negative at rb = 0.82**, inside the honest interval.

RF-SAFE widens 71 → 83. `l311`/`l312` never mention `rb`. Nobody ran L230's
instrument against RF-SAFE. `HANDOFF_2026-08-29.md`'s "answered every open item in
its §5" refers to a *different* document and says shipping was untouched, so §5(a)
stood unrepealed.

Worse, `L230_REPORT.md` §3 named **{90, 107, 114, 120}** as "what turns it negative
above rb = 0.80". RF-SAFE ungates **107, 114 and 120** — three of those four (D
already had 90 on). And L313 independently found **n=114** to be the single case
that loses 0.2255 pp on Linux.

## Running it (`l342_rb_rfsafe.py`)

Control first: the script re-derives `rb = 0.7682`, exactly L230's published value,
so the instrument reproduces.

    table                                on    rb=0.72  rb=0.7682   rb=0.80   rb=0.82
    live _L196_LPGATE (= D)              71     +4.904     +4.509    +4.095    +3.832
    l228_gate_new.txt                    71     +4.920     +4.329    +3.789    +3.447
    RF-SAFE (uploaded)                   83     +4.798     +4.165    +3.596    +3.238

    delta vs D
    l228                                        +0.017     -0.180    -0.307    -0.385
    RF-SAFE                                     -0.105     -0.345    -0.499    -0.594

**On L230's instrument RF-SAFE is negative everywhere in [0.72, 0.82], including at
the measured `rb`** — worse than the l228 table §5(a) was written to reject. Two
instruments, opposite signs, and the gap is larger than the decision margin.

## Which instrument is right — a test neither of them owns

Both use the same `TH = 0.304551` and the same republished 2026-08-23 medians
(verified identical to 6 dp), and their `dt` estimates agree (n=114: L230 0.388 s
vs L312 0.394 s). **The entire disagreement is in `slack = TH·med − t_ship`, i.e.
in the estimate of our own runtime on the grader.**

That quantity has ground truth. **Beta ran with no LP at all and its graded wall was
measured at 52.07 s.** Any LP-off estimate must come in at or below that.

    L230  sum POOL, rb-scaled, LP OFF      54.90 s   +5.4 %   <- EXCEEDS the measured wall
    L230  sum POOL, unscaled,  LP OFF      60.44 s  +16.1 %
    L312  baseline (beta measured x 0.868) 45.19 s  -13.2 %   <- the shipped package is faster
                                                                 than beta; L234/L237 made it so

L230's LP-off estimate is **larger than the measured LP-free wall**. It cannot be
right, and it errs in the direction that shrinks every slack.

The consequence is visible per case. On L230's numbers **all twelve** of RF-SAFE's
counts overrun their slack and four (79, 81, 94, 95) have *negative* slack — the
case is already past the RF floor before any LP is added:

    cases with NEGATIVE slack, before any LP
      L230 model   16 / 100
      L312 model    2 / 100

**Beta's measured cost-weighted RF was `0.7000400598775689` — 0.006 % above the 0.70
floor.** Sixteen cases sitting past the floor is not compatible with a number that
tight. Two is.

## What this does and does not settle

**Settles:** §5(a)'s verdict on widening rests on a runtime model contradicted by the
one piece of real grader data we have. RF-SAFE's baseline is that data (52.07 s,
matching the recorded beta wall exactly) scaled by a measured speedup. The
prohibition does not bind, and now there is a measured reason rather than an
argument. The {107, 114, 120} overlap is explained the same way — L230 flagged them
using the pessimistic slack.

**Does not settle — two honest caveats:**

1. **This validates the slack half, not `F`.** L230's per-case box→grader factor `k`
   has p50 0.4427, i.e. an implied speed factor of **2.26**. L312 uses a single
   global **F = 2.38–3.17**. Even its conservative end is slightly more optimistic
   than L230's median measurement, and a single constant cannot carry the 1.8×
   per-case spread (k p10 0.329 → p90 0.582) that L230 does measure. RF-SAFE's NET
   is positive down to f_eff ≈ 1.77, so this does not flip the sign, but the `F`
   axis is the weaker half of the case.
2. **n=114 remains fragile for an unrelated reason.** L313's finding is on the
   *quality* axis — the Linux LP lands on a worse vertex of the same degenerate
   program, costing 0.2255 pp. Nothing here touches that. It is a coincidence of
   attention, not of mechanism, that both L230 and L313 named the same case.

## The process gap, which stands

L294–L313 moved to a better-anchored pricer and simply proceeded. **Nobody wrote
down that §5(a) was thereby superseded.** The reviewer was right to demand a reason;
the reason existed and was not recorded. This file is the record.

    l342_rb_rfsafe.py     the re-run, control row included
