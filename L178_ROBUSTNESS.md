> # 🚨 VOID — read this first (added 2026-08-24, later the same day)
>
> **Every runtime projection in this report is withdrawn.** They were built from
> walls measured on this box, and this box cannot measure this configuration.
>
> Route A (`_route_a_default`) is ON by default at **>=40 DETECTED cores**, and
> every arm here ran with `ICCAD_ADAPTIVE_CORES=48`, so route A was live in all
> of them. It converts each profile into frame tasks on a global queue -- built
> for 48 REAL cores, where L110/L111 projected wall **-32.2%**. This box has
> **16 physical** cores, so there it is pure oversubscription:
>
> ```
>            route A ON   route A OFF   M73        cost
>   n= 60      3.589s        1.233s     1.277s     bit-identical (1.210020)
>   n=120      4.011s        2.757s     2.780s     bit-identical (1.267585)
> ```
>
> With route A off the current wrapper matches M73 to within noise. So the
> "1.89x unattributed pool regression", the 3.95x cur/M73 ratio, the `a + b/C`
> fits, "graded 1.10947", and the whole rank ladder are artefacts. Route A's
> cost/benefit **inverts** between 16 and 48 real cores, which no
> parallel-efficiency fit can represent.
>
> What survives: the pool-size and env comparisons (`l180_diff.py`: at n=120 and
> n=60 the current wrapper selects the SAME 35 indices with IDENTICAL effective
> env as M73), and everything in `L172_REPORT.md`, which is arithmetic on the
> published medians and committed OOS arms rather than a timing measurement.
>
> Also retracted from an earlier draft: a claim that the shipped binaries were
> "pre-L108" because `strings` found no `ICCAD_FRAME_REPORT` in them. `strings`
> finds **zero** `ICCAD_` variables in either binary, so the tool was wrong, not
> the binaries -- `grep -a` finds `ICCAD_FRAME_REPORT` and
> `ICCAD_FORCE_FRAME_IDX` in both.
>
> Clean pool-only numbers, route A off: `l181_poolonly.ps1`.

# Is "the current tree scores worse than beta" robust to the core model?

`l178_verdict.py` projects the shipped tree at graded **1.10947** against beta's
**0.92659**, using `k = 0.935` to carry the measured 16-real-core wall ratio to
the grader's 48. The obvious objection: that extrapolation could be wrong, and
a 48-core grader might absorb the bigger pool almost entirely.

Bound it from the other end. In the `wall(C) = a + b/C` fits the wall tends to
`a` as C grows, so the MOST favourable possible ratio -- infinitely many cores,
the pool term free -- is `a_cur / a_M73`:

```
n=120    2.178 / 1.280 = 1.70x
n=114    2.536 / 1.260 = 2.01x
```

So the ratio cannot fall below ~1.7-2.0x however parallel the grader is,
because `a` contains the serial `_proxy_metrics` tail and that scales with
PROFILE COUNT (M47: ~71 ms each, and parallelising them was 4x worse -- GIL).

Pricing the optimistic end on the 2026-08-23 medians: an aggregate 1.85x wall
is ~52.07 -> 96 s, which `l172_budget.py` prices at about **-10.5%** of score
(it reads -4.54% at 1.5x and -13.01% at 2.0x). The quality gain is **+7.80%**.

```
                        wall ratio    RF cost    quality    net vs beta
fit at C = 48              3.69x       ~ -25%     +7.80%      much worse
infinite cores             1.85x       ~ -10.5%   +7.80%      still worse
```

**Both ends of the band are worse than the beta package.** The conclusion does
not depend on the core model; the core model only decides HOW much worse.

The direction of the remaining bias is also known and unfavourable: fitting
`b/C` linearly assumes perfect parallel efficiency, and real speedup saturates,
so the C=48 row understates the wall rather than overstating it.
