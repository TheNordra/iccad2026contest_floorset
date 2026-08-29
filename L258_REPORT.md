# L258 — the acceptance rule is solved, the gain does not transfer, the line is closed

**Verdict on task (b): the acceptance question turned out to be an identity, not a
research problem — and solving it does not help, because the profiles that carry
the gain transfer at 0 %.** The L252→L258 arc ends here.

Nothing shipped.

---

## 1. The acceptance rule was never the hard part

L256 accepts a shrink when `csc_of` improves, and `csc_of`'s exchange rate between
area and hpwl is a hardcoded `hw = (N>=116)?0.12:0.06` (`constructive.cpp:1385`)
with no relation to the true cost's `A₀/H₀`. So a better rule looked like the lever.

Priced offline from `l257_cache.pkl` (base and L256 true cost, 40 cases × 51
profiles), three acceptance modes, each then scored by the shipped proxy:

| | weighted true cost | vs base |
|---|---|---|
| base | 1.511619 | — |
| **PROXY accept** (keep the better-by-shipped-proxy of each pair) | **1.506617** | **−0.3309 %** |
| **ORACLE accept** (keep the better-by-TRUE-cost — needs labels) | **1.506617** | **−0.3309 %** |
| TWIN (keep both, L257) | 1.506617 | −0.3309 % |

🔑 **All three are identical, and PROXY ≡ TWIN is an identity, not a measurement:**
`argmin` over a union equals `argmin` over the per-pair `argmin`s. Two consequences:

* **The best possible acceptance rule buys exactly what the twin buys** —
  −0.3309 %, no more. There is no hidden prize in a smarter rule.
* **It costs no extra profile slots.** L257 priced 8 *extra* profiles; the same
  gain is available by having each gated profile emit both layouts and letting the
  wrapper's proxy arbitrate. That reframing is the one real gain of task (b).

Among the 915 profile-layouts L256 changed, a perfect rule would have kept 548
(60 %) — so `csc_of` is wrong ~40 % of the time, exactly as suspected. Fixing it
is worth the difference between the overlay and the twin, and nothing beyond.

## 2. The wall constraint, and the first positive NET in this arc

The grader's profile phase is **max-bound** at n=120 (the research handoff's own
`d_max 3.204 s > sum/48 2.501 s`), so running the shrink on profile *p* costs
nothing unless *p*'s shrunk time exceeds the pool's max-setter. Measured, 51
profiles × 3 heavy cases, min-of-2, same-batch ratios:

    pool max-setter = prof 100 at 0.986s  (NOT one of the 8 that carry the gain)

    86  0.960 -> 1.141  x1.188  RAISES the wall     0   0.871 -> 0.973  affordable
    101 0.929 -> 1.064  x1.146  RAISES              87  0.813 -> 0.865  affordable
    20  0.889 -> 1.018  x1.145  RAISES              28  0.357 -> 0.427  affordable
    94  0.919 -> 0.994  x1.082  RAISES              95  0.363 -> 0.384  affordable

| gated set | quality | max-bound NET | sum-bound NET |
|---|---|---|---|
| all 8 greedy parents | +0.331 pp | −2.04 pp | +0.03 pp |
| **affordable {0, 28, 87, 95}** | **+0.211 pp** | **+0.211 pp** | **+0.115 pp** |
| affordable, K=3 {87, 28, 95} | +0.209 pp | +0.209 pp | **+0.153 pp** |

That is the first positive NET anywhere in L252–L258.

## 3. And it is zero out of sample

The gated set was chosen greedily on the same 40 cases it was scored on. M76
measured in-sample source-set selection transferring at ≈5 %. Split-half, free
from the cache:

    A->B   chose [95]         train +0.042 pp  ->  TEST +0.000 pp   transfer 0%
    B->A   chose [87, 28, 0]  train +0.372 pp  ->  TEST +0.000 pp   transfer 0%

🚨 **Transfer is exactly zero in both directions.** The profiles that carry the
gain on one half carry nothing on the other, and the two halves' test-optimal
values (+0.372 pp vs +0.042 pp) differ by 9×, so the benefit is not even evenly
distributed across cases.

This is the L257 diagnostic arriving from the other side: only **17 of 2040**
twins beat the best original, concentrated in **9 of 40** cases. Nine idiosyncratic
cases, each won by a different profile — split them and nothing overlaps. It is
M56/M76's "winner is case-idiosyncratic" for the third time in this ledger.

⇒ **Expected OOS value of the affordable gated configuration ≈ 0**, and even the
in-sample +0.211 pp is below this project's own **0.30 % OOS ship bar**.

## 4. What the whole arc adds up to

    L256 global overlay   NET +0.505%      proxy churn; mechanism invisible
    L256 isolated         -0.2506%         real, small, 154/105, ~3.0 sigma
    L257 twin             -0.3309%         oracle-tight, zero selection loss
    L257 twin NET         -0.15 pp at K=1  priced out by profile slots
    L258 best rule        -0.3309%         an identity: cannot exceed the twin
    L258 affordable gate  +0.211 pp NET    first positive... in-sample only
    L258 transfer         0%               and it is noise

**The mechanism is real and the deployment is not.** Every route from +0.25 % of
isolated mechanism to a shippable number is now measured and closed: as an
overlay it drowns in `hmin` churn, as twins it cannot pay for slots, and gated to
the profiles it can afford it is fitting nine cases.

## 5. Honest limits

* 40 cases, s1, heavy band. The transfer test is split-half within s1, not a true
  OOS sample — but it is the *optimistic* version (same distribution, same
  corpus) and it already reads 0 %.
* Timing is 3 cases at n=120 on this box; the max-setter's identity on the grader
  is assumed, not verified. That assumption only matters for §2, which §3 makes
  moot.
* The wall→NET conversion is L248's (0.151 pp per 1 % of heavy-band wall).
* The "emit both layouts from one process" deployment was never implemented —
  §3 removed the reason to.

## 6. Files

```
l258_accept.py     the three acceptance modes + the perfect-rule bound
l258_maxsetter.py  51-profile timing, max-setter identification -> l258_times.pkl
l258_gate.py       the affordable set, its NET, and the split-half transfer test
```
