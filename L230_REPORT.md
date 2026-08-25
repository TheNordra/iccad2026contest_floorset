# L230 — the §4 gate rebuild is real but 8× smaller than claimed, and the handoff's own table is the one variant that is not robust

Follow-on from `HANDOFF_2026-08-26.md` §4, which stages a rebuilt `_L196_LPGATE`
(63 → 71 block counts on) at **+0.483 pp** and ships the table in
`l228_gate_new.txt`. Nothing in the shipping tree was touched by this session.

---

## 0. TL;DR

| table | on | NET @ measured ratio | Δ vs shipped | Δ across the plausible ratio range |
|---|---|---|---|---|
| shipped `_L196_LPGATE` | 63 | **+4.271 %** | — | — |
| `l228_gate_new.txt` (handoff §4) | 71 | +4.329 % | **+0.058 pp** | +0.47 … **−0.37 pp** |
| time gate s = 1.20 recomputed | 68 | +4.405 % | +0.134 pp | +0.42 … −0.16 pp |
| **shipped ∪ {102, 103, 106, 119}** | **67** | **+4.531 %** | **+0.260 pp** | **+0.47 … +0.04 pp** |

* The handoff's **+4.265 % baseline reproduces at +4.271 %** from an
  independent measurement of the pool ratio. The rank-2 claim stands.
* The **+0.483 pp for the rebuild does not.** It is **+0.058 pp**, and it is the
  only candidate that goes **negative** at the pessimistic end of the ratio
  interval.
* What survives is a **strictly additive** four-block-count change, positive
  everywhere in the interval and, being additive, incapable of losing quality.

## 1. Why the §4 number could not have been right

`l228_gate_new.txt` has no deriving script — the derivation was inline. Its
inputs cannot have been consistent, because of where they come from:

* `l203_marginal_gate.py` builds `POOL[n]` from `_l181_cur.json` (full pool,
  LP off) and the box→grader factor `k = t_beta(n)/w_m73(n)` from
  `_l181_m73.json`. **Both are from the 2026-08-24 batch.**
* The 2026-08-25 arms run **17.5 % slower than that batch in the n ≤ 100 control
  band** — a band in which nothing changed.

Dividing an Aug-25 pool time by an Aug-24 `k` therefore puts ~17 % of
systematic error inside a threshold test. §4 reads out as if the pool ratio were
≈ 0.72; the measured ratio is **0.768**, and at 0.72 the l228 table does score
+0.467 pp (§3). **The claim is the right number for the wrong pool.**

## 2. How this measurement is built instead

`l230_calib.sh` — four arms × three repeats, one batch, one box:

| arm | configuration |
|---|---|
| A | REFINE = 2 (shipped), `ICCAD_SHAPE_LP=0` |
| B | `ICCAD_L223_REFINE_HEAVY=4`, `ICCAD_SHAPE_LP=0` |
| C | REFINE = 2, `ICCAD_LP_GATE=0` (LP everywhere) |
| D | `ICCAD_L223_REFINE_HEAVY=4`, `ICCAD_LP_GATE=0` |

Everything enters as a **ratio measured inside one batch**, so the machine
factor cancels and no cross-day term survives:

    POOL_new[n] = POOL_old[n] × A/B          DT_new[n] = DT_old[n] × (C−A)/(D−B)

### 2.1 The ratios are BAND-level, and that is not a convenience

    rho = pool  REFINE2/REFINE4    n<=100 (control) 1.027    n>100 0.789
    sig = dt_LP REFINE2/REFINE4    n<=100 (control) 1.000    n>100 1.044

    dispersion   rho  21-100   p10 0.959  p50 1.027  p90 1.122     <- noise floor
                 rho 101-121   p10 0.736  p50 0.789  p90 1.254
                 sig  21-100   p10 0.086  p50 1.000  p90 9.269     <- pure noise
                 sig 101-121   p10 0.297  p50 1.108  p90 2.490

REFINE is a **band constant** — every case above n = 100 gets the same change —
so the effect has no per-n structure to estimate, and a per-n ratio only injects
this box's ~17 % per-case noise into a threshold test that has 0.12 pp of rank
margin. The control band's dispersion (0.959–1.122) *is* that noise, measured.

🚨 **`sig` is worse than noisy: it is a quotient of DIFFERENCES.** `C−A` and
`D−B` are differences of two ~1 s numbers, so four block counts came out with
`D−B ≤ 0` (the LP-everywhere arm timed *faster* than LP-off) and the per-n ratio
went to **1.9 × 10⁴**. The first run of this analysis scored the shipped gate at
**−59 %** because of exactly that. Band level or nothing.

De-biased band estimates: **pool × 0.7682** on n > 100, **dt_LP × 1.0445**
(16 well-conditioned block counts) — i.e. the LP's own cost is unchanged by
REFINE, which is what a post-pool stage should do.

### 2.2 Where the budget is now

| band | share of weight | pool_old | pool_new | median | slack |
|---|---|---|---|---|---|
| 21–60 | 0.6 % | 8.4 | 8.4 | 41.2 | 1.63× |
| 61–100 | 18.2 % | 28.1 | 28.1 | 99.4 | **1.11×** |
| 101–120 | **81.1 %** | 23.9 | **18.4** | 75.5 | **1.28×** |

`slack = 0.3046·M/pool` — how much more wall a case can spend at zero RF cost.
REFINE = 2 bought the heavy band from 0.97× to 1.28×, and **that 0.28 is the
entire budget the gate rebuild is spending.**

## 2.3 Chain of custody: the ratio is only valid if arm B *is* the old baseline

`POOL_new = POOL_old x (A/B)` assumes arm B — today's tree with
`ICCAD_L223_REFINE_HEAVY=4`, LP off — is the same configuration as
`_l181_cur.json`, the Aug-24 LP-off run `POOL_old` was built from. Between them
sit L205 (route A default off), L211/L213 (the pool drop, default off) and L223
(the band plus its kill switch). Checked rather than assumed:

    _l181_cur.json  vs  results_L230_B1.json   100/100 identical on cost

Bit-for-bit, every block count. So the two arms differ only in *when* they ran,
which is exactly what the ratio is there to cancel.

⚠️ One approximation remains and is not measurable from here: `qual_pern` reads
the OOS LP-on/LP-off arms (`l192_*_full`, `l194_*_fulloff`), which were run on
the pre-REFINE tree. In set the per-case LP gain is unchanged by REFINE — the
band medians agree to three decimals and only the two cases REFINE moves at all
(113, 115) differ — so the transfer is sound in set; out of sample REFINE=2
moves more cases than it does in set, and nobody has re-run the LP arms there.
The direction is unknown and the size is bounded by the in-set agreement.

## 3. The decision table, and why the shape of the answer is "add, never drop"

NET, both OOS samples, both directions, at four values of the pool ratio. The
measured value is 0.7682; ±1 s.e. on a 20-case band median is ≈ ±0.04, so
**[0.72, 0.82] is the honest interval.**

| candidate | on | rb=0.72 | **rb=0.7682** | rb=0.80 | rb=0.82 |
|---|---|---|---|---|---|
| shipped | 63 | +4.453 % | **+4.271 %** | +3.993 % | +3.814 % |
| l228 handoff | 71 | +4.920 % | +4.329 % | +3.789 % | +3.447 % |
| s = 1.15 (4 adds, 5 drops) | 62 | +4.923 % | +4.529 % | +4.115 % | +3.852 % |
| s = 1.20 (5 adds, 0 drops) | 68 | +4.870 % | +4.405 % | +3.945 % | +3.654 % |
| **shipped ∪ s1.15 adds** | **67** | **+4.925 %** | **+4.531 %** | **+4.117 %** | **+3.854 %** |

Δ versus shipped:

| candidate | rb=0.72 | rb=0.7682 | rb=0.80 | rb=0.82 |
|---|---|---|---|---|
| l228 handoff | +0.467 | +0.058 | **−0.204** | **−0.367** |
| s = 1.20 | +0.417 | +0.134 | **−0.048** | **−0.160** |
| **shipped ∪ {102,103,106,119}** | **+0.472** | **+0.260** | **+0.124** | **+0.040** |

Three things fall out:

1. **The drops are worth nothing.** `s=1.15` drops {48, 59, 68, 71, 73} and
   scores +4.529 %; the same table with those five put back scores +4.531 %.
   The adds are the entire effect. Since G7 measures the gated LP at
   **62 better / 0 worse**, an added block count cannot lose quality — it can
   only cost wall — so a purely additive table has a strictly better risk
   profile and is not paying for it.
2. **The l228 table over-adds.** The four block counts it has beyond the robust
   set — {90, 107, 114, 120} — are what turns it negative above rb = 0.80.
3. **`s` is no longer 1.2.** At the new pool times the one-parameter family
   peaks at 1.15, and its optimum is reached by adds alone.

## 4. What this says about the rank claim

The **baseline** moves with rb too: +4.45 % → +3.81 % across [0.72, 0.82]. So

> rank 2 is held with **~0.4 pp of margin** (after the additive change) against
> a **±0.3 pp** modelling uncertainty in the pool ratio alone.

Before this change the margin over the rank-2 team's 0.888187 was
**0.12 pp** — smaller than the uncertainty in the number that establishes it.
That, and not the +0.26 pp, is the reason to take the change.

## 5. Side result: a uniform REFINE band is not the right object — but nearly

`l231_headroom.py`, on the L219 sweep's per-profile durations (100 cases × 51
profiles), asks how much of the uniform cut a **top-k-only** cut would deliver:

```
   k        wall ratio   recovered share of the full cut
   1           0.9869          6%
   8           0.8887         55%
  20           0.8087         95%
  30           0.7981        100%
  51           0.7981        100%
```

The wall is a max, so the tail cannot move it — but the top is a **plateau**
(L203: the argmax agrees across three runs on 6/100 block counts), and it takes
**~30 of 51** profiles to collect the whole saving. So the uniform band is
wasteful, but only by the last 21 profiles: **41 % of the pool is paying quality
for zero wall.** Restoring those 21 to a higher REFINE is a free-quality lever
whose size is unmeasured; it is the cheapest remaining item that is not a
constant re-price.

## 6. Files

```
l230_calib.sh          the four-arm × three-repeat batch
l230_gate.py           the derivation; writes l230_pool_new.json
l230_pool_new.json     POOL/DT on the post-REFINE tree, de-biased
l231_headroom.py       per-profile wall headroom (§5)
```
