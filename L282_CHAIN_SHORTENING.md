# L282 — shortening the critical chain. The box was never what it was costing us.

L281 closed every topology edit that **lengthens** the critical chain and left
exactly one thing open (L281 §10.1 item 2): the dual move — take a unit **off**
the chain so the chain gets *shorter*, letting the LP shrink the bbox and pay
down `area_gap`. Unlike everything L281 measured, this move is feasible by
construction as far as the box is concerned: a shorter chain fits.

**RED, and it fails twice for two unrelated reasons.**

    9 heavy-band cases, 413 chain-shortening relocations, official strict scorer

      LP-infeasible                             374 / 413 = 90.6 %
      of the 39 that solved, cost got worse      31 / 39
      union-oracle vs the polished control       **+0.0057 %**   (1 of 9 cases)

1. **The journey, not the destination.** 90.6 % never solve — and 7 of 30
   sampled failures become feasible only when the bbox is allowed to **grow**
   20 %, for a move whose entire purpose is to shrink it.
2. **The exchange rate is 2.74 : 1 against.** When the shrink *is* realisable,
   buying one unit of `area_gap` costs **2.74** units of `hpwl_gap`, and the
   score prices them identically. The LP, which minimises that exact objective,
   simply declines the shrink in 28 of 39 cases.

⇒ The critical chain really does pin the bounding box (L281 §5), and **the
bounding box is not what the score is charging us for.** Slack was the wrong
thing to want.

---

## 1. Gate 0 — the redundancy is real, so this was worth building

No LP, all 100 graded cases (`l282_chain_gate.py`). Deleting a unit from the
binding axis's constraint graph entirely is the best case for relocating it, so
`row − chain(without u)` upper-bounds what one relocation can shorten:

    chain shortening available from the best single unit
      p50 0.82 % of the row   p90 5.64 %   max 14.57 %
      cases where any shortening is possible : 64/100
      binding floor is the CHAIN             : 93/100   (frozen span in only 7)

      area prize, weighted exp(n/12)
        optimistic  (the other axis absorbs the unit for free)   **+0.6282 %**
        pessimistic (it lands on the other axis's critical path)  −3.1957 %
        heavy band n>=101 only, optimistic                        +0.5901 %

**+0.63 % is above the 0.30 % bar**, and unlike L281 the frozen blocks are *not*
the wall — the chain is, in 93/100 cases. So the redundancy exists and it is the
chain that is holding it. The bracket is wide because the sign depends entirely
on where the unit lands, which no bound can answer. Hence the probe.

⚠️ The redundancy is **not** uniformly distributed: cases 85 and 88 — the two
L281 scanned exhaustively — have a prize of **exactly −0.0000 %**, i.e. no single
unit removal shortens their chain at all. The first pilot ran on them by
inheritance from L281 and found nothing, which was a property of the case
selection, not of the mechanism. The real run targets the 11 heavy cases that
have redundancy (case 93 +3.00 %, 80 +2.35 %, 89 +2.29 %, …).

## 2. The move

Identical machinery to L281 — `force_rel` semantics, the coherence certificate,
the LP, the official strict scorer, and a control given **the same LP and the
same polish budget** — with two changes:

* candidate units are those **on the binding critical chain**, ranked by how
  much removing them shortens it, not by wire prize;
* targets are ranked by the **predicted new bbox area**, computed exactly from
  the new H and V constraint graphs, not by wire.

A candidate is only kept if both new chains still fit the current box and the
predicted area strictly decreases, and only if it is **binding** (L281 §6 —
a relation change that the current placement already satisfies moves nothing).

## 3. Kill 1 — 90.6 % never solve, and the box is not why

    status of 413 chain-shortening relocations

      lp_status_2 (infeasible)   374   90.6 %
      feasible but worse          37    9.0 %
      MOVER                        2    0.5 %

These moves *shrink* the box, so "oversized" cannot be the cause — the
certificate already guarantees both new chains fit. Running L281's diagnostic on
the 30 with the largest predicted shrink (`l282_why_infeasible.py`):

      feasible once boundary equalities are dropped   :  0
      feasible once the bbox may GROW 20 %            :  1
      feasible only with both                         :  6
      still infeasible with both                      : 23

**0/30 on the boundary arm is the third independent reproduction** of M64's most
informative single result (M64 0/15, L281 0/30, L282 0/30) — on three different
move semantics and two different anchors. Boundary equalities have never once
been the cause of anything on this axis.

🔑 **And 7 of 30 solve only when the box may GROW.** A move designed to make the
box smaller becoming feasible only when the box is allowed to get bigger is a
clean statement of the mechanism: the unit cannot *get* to its destination.
Every other pair keeps its anchor disjunct, so the displacement LP has no route
through the layout — M64_REPORT §4's "fixed-disjunct chains", now confirmed for a
move that was supposed to be immune to the geometry.

## 4. Kill 2 — the trade is 2.74 : 1 against, and the LP knows it

This is the more interesting one, because it applies to the 9.4 % that *do*
solve. Over the 39 feasible solutions:

      predicted bbox shrink                       p50  4.014 %
      d_area_gap (negative = area improved)       p50 +0.00000
      d_hpwl_gap (positive = wire got worse)      p50 +0.01917
      d_vrel                                      p50 +0.00000

      solutions where area actually improved      11 / 39
      solutions where wire got worse              31 / 39

      wire paid per unit of area bought, where area did improve (n = 11)
        median **2.74**, inter-quartile range 2.12 - 5.41

The candidates were selected for a median predicted shrink of **4 %**, and the
median realised `area_gap` change is **exactly zero** — the LP *declined to take
the shrink* in 28 of 39 cases. It is not failing to find it; it is minimising
`0.5·(hpwl_gap + area_gap)` and correctly refusing a trade that costs 2.74 units
of wire per unit of area.

🔑 **This is L268's exchange rate again, worse.** L268 measured the packing-time
version — big-first commitment order reached the highest utilisation ever
recorded (81.3 → 85.2 %) and was still never worth it, because it spent **1.2×**
wire per unit of area. Post-hoc the same trade costs **2.74×**. Two independent
mechanisms, opposite ends of the pipeline, same verdict: on this score
**density is not for sale at a price worth paying.**

## 5. Result

    case   base        best        gain
      81   1.217953    1.216692    +0.1035 %
      85   1.186103    1.186103    +0.0000 %
      86   1.202468    1.202468    +0.0000 %
      91   1.168660    1.168660    +0.0000 %
      99   1.265344    1.266667    −0.1045 %
      87   1.168547    1.170070    −0.1303 %
      93   1.203117    1.205768    −0.2203 %
      89   1.323919    1.329616    −0.4303 %
      80   1.289770    1.296922    −0.5545 %

    union-oracle vs the polished control : **+0.0057 %**   (1 of 9 cases gains)

Note that the *best available* relocation is a **loss** on five of the nine
cases. Even taking the maximum over every candidate on each case, most cases
have nothing that beats leaving the layout alone.

## 6. What this closes, and what it does not

**Closes: post-hoc chain shortening.** Together with L281 this closes the axis in
both directions — edits that lengthen the chain are infeasible (L281), and edits
that shorten it are either unreachable (§3) or a losing trade (§4). Three move
semantics — M64's pair flip, L281's relocation, L282's chain shortening — die on
the same fixed-disjunct interlock.

**Does NOT close: the packing-time version.** §3's obstacle — the other ~4900
pairs frozen at their anchor relations — **does not exist during construction**.
A greedy that avoids building a long chain in the first place is not tested by
anything here, and L281 §10.1 was right to call it a packing-time idea.

⚠️ **But §4 does bear on it, and it is the discouraging half.** The 2.74 : 1
exchange rate is a property of the geometry and the scoring formula, not of the
move. A packing-time rule that produced a shorter chain would still be buying
area, and L268 measured that trade at 1.2 : 1 against *at packing time* with the
largest utilisation gain in the project's history. So the packing-time version
is untested, but its prize is bounded by an exchange rate that has now been
measured as unfavourable twice, independently, at both ends of the pipeline.

⇒ Before building a chain-aware packer, the cheap thing to measure is whether a
shorter chain can be obtained **without** compressing — i.e. whether the wire
penalty in §4 is intrinsic to density or an artefact of squeezing an already
committed layout. That is a one-flag experiment on `constructive.cpp`, not an
LP probe.

## 7. Honest limits

1. **9 cases, 413 solves.** Gate 0 is all 100; the LP arm is the 8 heavy cases
   with measurable chain redundancy plus case 85 from the pilot. Cases with no
   redundancy were deliberately excluded — they cannot benefit by construction.
2. **Single-unit moves only.** Shortening a chain may require moving two or more
   units together. That was not tried, and Gate 0's bound is single-unit.
3. The exchange rate in §4 is measured on the 11 solutions where area actually
   improved. That is a small sample and it is conditioned on feasibility.
4. Same offline research LP as L281; nothing here is shippable as it stands.
5. Both corpora: not re-run on OOS, for L281 §8's reason — the mechanism needs
   chain slack, in-set has less of it than OOS (86 % vs 23 % saturated), so an
   OOS measurement would read better and fail to transfer. The candidate is
   negative on the graded corpus, so OOS cannot rescue it.

## 8. Files

```
l282_chain_gate.py       Gate 0: chain redundancy + frozen span, no LP, 100 cases
l282_chain_probe.py      the probe: chain-targeted candidates, area-ranked
l282_why_infeasible.py   boundary equalities / bbox growth / neither
l282_cache.pkl           413 LP solves
l282_gate.log  l282_probe_heavy.log  l282_report.log
```

`constructive.cpp`, `optimizer_constructive.py` and `build_submission.D/` were
not touched.
