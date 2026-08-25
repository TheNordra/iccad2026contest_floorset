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

# L173 — the pool's wall grew since M73, and nobody had ever measured it

**Date** 2026-08-24 · **Status** measured; nothing on the shipped path changed

---

## 0. Why anyone looked

`L172_REPORT.md` re-priced the runtime budget on the 2026-08-23 medians at
**14.72 grader-seconds for the whole corpus**. That budget is spent against
`t_beta`, our graded per-case runtime for the package the organisers actually
ran (M73, 52.07 s total).

Every runtime verdict in this ledger prices a mechanism as a delta *against the
tree it was measured on*. Nothing had ever compared the tree against **M73**.
`HANDOFF_2026-08-24.md` states the assumption outright — "beta's 52.07s
contains no LP" — and then adds LP seconds to it. That is only valid if the
pool did not move.

## 1. The pool moved

`_pool_indices` at n>100 with `ICCAD_ADAPTIVE_CORES=48`:

```
M73 (git 7f38893)   45 profiles defined,  35 run
current tree       138 profiles defined,  51 run
```

and per-profile cost rose too, so the wall ratio exceeds the count ratio.

## 2. The premise both high-core tiers were shipped on has expired

`l173_attrib.sh`, Windows, forced 48c, LP off, four heaviest in-set cases:

| arm | profiles | n=120 | n=117 | n=114 | n=111 |
|---|---|---|---|---|---|
| `ICCAD_ADAPTIVE_POOL=0` (full) | ~138 | 12.652 | 11.473 | 10.895 | 9.947 |
| shipped default | 51 | 8.433 | 8.514 | 7.970 | 5.077 |
| `ICCAD_M80_TIER=0` | 43 | 6.497 | 7.628 | 7.129 | 3.658 |
| `+ ICCAD_M67F_TIER5=0` | 21 | 4.755 | 4.226 | 4.441 | 3.163 |

The comment above `_M80_EXTRA` states the premise M80 **and** tier-5 both ride:

> At 48 cores 100/100 cases are max-setter bound, so K extra profiles that are
> each cheaper than the incumbent max cost ~nothing.

Under max-setter binding, dropping 8 profiles that are *not* the max-setter is a
no-op. It cost 23% of the wall at n=120. The wall is **linear in profile
count**, i.e. sum-bound, not max-bound.

`l125_dt_cache.pkl` (per-profile dt, current 51-profile pool, n=80..99) says why,
and also refutes the obvious suspect: per-profile dt is only **1.0–1.6 s** and
the max-setter is usually a *base* profile (idx 2/17/18/19/20/21/23/24/27), not
one of M80's (idx 86+). M80's profiles are not slow. But

```
c* = sum(dt) / max(dt)  on the CURRENT pool:  29.7 – 36.4
```

against the **19.3 median / 22.5 max** that `_M67F_CORES_MIN = 40` was derived
from. The pool grew and `c*` grew straight through the threshold the gate was
justified on. CLAUDE.md also records that detected cores are an **upper bound**
on effective ones.

## 3. Most of the wall is core-INDEPENDENT

`l173_cores.ps1` confines the whole case — python and every `constructive.exe`
child inherit the affinity — and fits `wall(C) = a + b/C`:

```
current tree   case 99 (n=120)   4c 38.42s   8c 20.27s   16c 11.25s
                  wall(C) = 2.178 + 144.9/C      a = 2.178 s
               case 93 (n=114)   4c 30.38s   8c 16.21s   16c 9.62s
                  wall(C) = 2.536 + 111.1/C      a = 2.536 s
M73 proxy      case 99           4c 16.85s   8c  8.80s   16c  5.30s
               case 93           4c 12.30s   8c  6.80s   16c  4.01s
```

At 48 cores **42–52 % of the current tree's wall is core-independent**. `a` lands
close to the independent estimate from M47's rule (51 profiles × 71 ms = 3.6 s of
serial `_proxy_metrics`); two unrelated measurements agreeing is the reason to
believe there is a real serial floor.

`HANDOFF_2026-08-24.md` already carries that rule —

> every added profile costs ~71 ms of **serial** `_proxy_metrics` on the main
> thread, so N profiles cost at least `N * 0.071 / f` grader-seconds
> **regardless of core count** — parallelising the proxies was 4x WORSE (GIL).

— and used it to kill a 32-profile *probe* tier. **It was never charged against
the 51 profiles already shipping.**

### 32 logical cores are not 32 cores

The 32c point is *slower* than 16c (n=120: 11.314 vs 11.250; n=114: 10.116 vs
9.622). This box is 16 physical + SMT, so 32c carries no throughput information
and every fit above uses 4/8/16 only.

## 4. 🚨 The 6.2x WSL headline is NOT the transferable number

The first measurement compared the l166 WSL lanes against
`results_L160_m73_local.json` and read **6.22x** (LP off). It is real on that
box, but WSL runs the **identical configuration** 3.0–4.1x slower than Windows:

```
   n     WSL 32c    Win 32c   WSL/Win
 120    27.485s     8.433s     3.26x
 117    29.886s     8.514s     3.51x
 114    23.979s     7.970s     3.01x
 111    20.552s     5.077s     4.05x
```

and `f = 3.17` was calibrated on M73's **WSL** run. Dividing Windows walls by a
WSL-calibrated `f` produced "our wall 222.6 s = 4.27x beta, 99/100 off the
floor, graded 1.458". **That number is an artefact of mixing two boxes and is
withdrawn** — the same cross-run-differencing error the ledger already records
three times. It was caught only because the WSL and Windows walls for the same
configuration disagreed by 3x.

On Windows, same box, same flags, the ratio is far smaller: **2.12x at n=120 and
2.40x at n=114, at 16 real cores.**

## 5. The calibration-free projection

`f` cancels if both walls are measured on one box:

```
t_current_grader(n)  =  t_beta(n) * [ w_cur(n) / w_M73(n) ] * k
```

with `t_beta(n)` the grader's own measurement of M73, and `k` carrying the ratio
from this box's 16 real cores to the grader's 48 (from the `a + b/C` fits above).
`l173_pair.ps1` runs both arms over all 100 cases; `l173_final.py` does the
arithmetic and prices it on the 2026-08-23 medians.

> **PENDING** — `_l173p_cur.json` / `_l173p_m73.json` still running.

## 6. The lever this exposes

`_proxy_metrics` recomputes, for all 51 profiles on one case, work that depends
only on `(constraints, n)` and not on `positions`:

```python
bound_l = constraints[:n, 4].tolist()      # three torch -> list conversions,
clust_l = constraints[:n, 3].tolist()      #   identical on every call
mib_l   = constraints[:n, 2].tolist()
nsoft, ngrp, nmib
idx = [i for i in range(n) if int(clust_l[i]) == g]    # O(n) per group,
                                                       # per profile
```

`l174_hoisted.py` computes those once per case and replaces the `O(n * ngrp)`
group scan with one `O(n)` bucketing pass, preserving list order and every quirk
of the original — including that an empty MIB group contributes `-1` to `vm`,
which the shipped code does and which must be kept, because the proxy is the
live selector's argmin and any difference can pick a different candidate.

`l174_proxy_bench.py` asserts bit-identical output on captured real inputs
before reporting timing. **Not yet run** — it must not share the box with a
timing measurement.

## 7. What is NOT established

- **That any tier should be turned off.** M80's OOS NET was +1.786% / +1.909%
  and tier-5's +2.289%. What is established is that their wall cost was priced
  against a premise (max-setter binding) that no longer holds, and against
  `m67e_rf48.py`'s kappa, which L159 records as wrong per case by up to 8x.
  They need **re-pricing, not removal**.
- **The size of the transfer**, until §5 finishes.
- **Anything about the depth map or `requirements.txt`** — see `L172_REPORT.md`.
  Those are staged and their in-set gates have still not run: the first attempt
  was killed by a second agent running `l171_gates.sh` on the same tree one
  minute after mine started.

---

## 8. Addendum — the box became exclusive at 11:45

Everything in §1-§7 above was measured while a second agent ran 100-case
evaluations in the same tree. `_quarantine/l173_contended/README.txt` records
which arms that destroyed. The affinity-confined core scans survive at ~5%
(the same 4c/n=120 point reads 36.902s clean and 38.416s contended); the
unconfined paired full runs do not and are quarantined.

`l176_wall.ps1` re-runs the three arms (cur / m73 / nom80) on an exclusive box,
and `l176_analyze.py` does the calibration-free projection and re-prices M80's K.

### Why the ratio does not wash out on a 48-core grader

The obvious objection to §5 is that this box has 16 physical cores, so BOTH
arms are sum-bound here, whereas on 48 cores both might be max-setter bound --
and the max-setter is a base profile present in both pools, so the ratio would
collapse to ~1.

The `a + b/C` fits answer that directly. As C grows the wall tends to `a`, and

```
a_cur (n=120) = 2.178 s      a_M73 (n=120) = 1.280 s     ratio 1.70
                                       at C = 48:        ratio 2.02
```

`a` is the part that does not shrink with cores, and it contains the serial
`_proxy_metrics` tail, which scales with PROFILE COUNT (M47: ~71 ms each, and
parallelising them was 4x worse). 51 profiles carry more of it than 35. So the
ratio persists at any core count -- that is the whole reason this is a real
cost rather than a small-box artefact.

⚠️ One bias, and it runs against the reassuring direction: fitting `b/C`
linearly assumes perfect parallel efficiency, and real speedup saturates. So
extrapolating from C <= 16 to C = 48 **understates** the 48-core wall. The
projection is optimistic, not pessimistic.
