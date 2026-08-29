# L302–L313 — `f` is pinned: **2.4 – 2.8 overall, 1.6 – 2.1 on the heavy band**, not 3.17

`f` converts a locally measured second into a grader second. Every RF bill this
project has printed depends on it, and the last four verdicts — LP k=2, the depth
frontier, `ICCAD_LP_GATE=0`, and my own rank-1 claim — all turned on which value
was used. The tree carries `F = 3.17` (`l172_depthmap.py:39`); the parallel
session's f-free method implied 2.37–2.44; nobody had measured the quantity `f`
actually is.

**Measured here: `f = 2.38 … 2.84` over the 100 cases, and `1.62 … 2.13` on the
heavy band, which is where the LP's bill lands.** 3.17 is above the whole
bracket.

    consequences, priced with the measured f and with the LP measured directly
    (`ICCAD_LP_TIMING`), against rank 1 = 0.858632

      arm          grader s   on floor      RF     best projected total   vs rank 1
      shipped      45.1-46.1  100/100    -0.23 %      0.868706 - 0.876568   -1.16 %
      LP k=2       48.2-49.9   96-99     -0.17..+0.04 0.866417 - 0.875386   -0.90 .. -1.11 %
      gate0        50.1-52.4   90-91     +1.32..+2.39 0.860840 - 0.872272   -0.26 .. -1.30 %
      gate0 + k=2  56.3-60.0   78-82     +3.17..+4.89 0.869054 - 0.883887   -1.20 .. -2.81 %

    break-even, expressed as a multiplier on the MEASURED f
      LP k=2        0.65 - 0.86 x     -> GREEN with margin
      gate0         0.69 - 0.89 x     -> GREEN, thin under the conservative baseline
      gate0 + k=2   0.87 - 1.13 x     -> STRADDLES; RED at the conservative end

🚨 **This retires my own headline from `L295_L300_RANK1_TARGET.md` §4.** At
f = 3.17 the combined arm read 0.8580/0.8584, just past rank 1. At the measured f
it is **1.2 – 2.8 % short**, and the depth on top of the ungated LP is the part
that stops paying. **Nothing on the table beats rank 1 any more; `gate0` alone
gets to within 0.26 %.**

---

## 1. What f is, and why the old number is a different quantity

`F = 3.17` was built at L157 §5h as `2.71 × 1.17`, where 2.71 = 141.07 s (a
reconstructed beta package, WSL) ÷ 52.07 s (the same package, on the grader).
Both sides of that ratio are **parallel walls**:

* the pool is 43 profile subprocesses. **43 ≤ 48, so on the grader they run in
  ONE wave and the pool's wall is exactly the slowest single profile.**
* this box has **16 physical cores** (Ryzen 9 8940HX, 32 logical). With c\* =
  Σdt/max dt at p50 **21.3** (max 30.2 — CLAUDE.md's "max 22.5" is stale), the
  local run is sum-bound: its wall carries a factor the grader never pays.

Measured directly: local parallel wall / local single-threaded work = **1.53×**.
That factor is the core count, and it is most of the old constant.

The shape LP is `scipy.optimize.linprog(method="highs")` — dual simplex,
**single-threaded**, confirmed here by `cpu/wall = 1.01–1.04` from
`ICCAD_LP_TIMING`. It runs after the pool, in the main process, on one core. It
collects **none** of the grader's core-count advantage, so the constant that
prices it must be a single-thread ratio.

## 2. The measurement

Decompose the grader's own published per-case wall:

    t_grader(n) = ( pool_wall + serial ) / f
      pool_wall = max( M , C )
        M = slowest profile subprocess          measured uncontended here
        C = 43 x _proxy_metrics                 main thread, one at a time
                                                (the M47 comment at :2726 is explicit)
        S = _serialize_input                    timed in situ, before the pool

    f_lo = sum( max(M,C) + S ) / 52.07     C fully hidden behind M
    f_hi = sum(  M + C  + S ) / 52.07      no overlap at all

    measured:  sum M 118.74 s   sum C 23.89 s   sum S 5.27 s
               local parallel wall of the same run 190.0 s
               grader, published                    52.07 s

               f_lo = 2.382     f_hi = 2.840

      band        f_lo   f_hi
      21 - 50     4.16   4.67
      51 - 80     2.66   3.08
      81 -100     2.31   2.73
      101-120     1.62   2.13     <- the band that carries 81 % of the weight

**Gate:** the capture run reproduces L285's beta configuration to the last digit,
`total_score = 1.2598976821946901`, with `ICCAD_REFINE_ITERS=4` on the heavy band
and 43 profiles — i.e. what was measured is the configuration the grader ran.

### 2.1 Both candidate biases were measured, and both are negligible

| bias | direction | measured |
|---|---|---|
| today's `constructive.exe` vs the graded M73 binary (`7f38893`, rebuilt) | unknown | **0.996** (mean over n=101–120, range 0.956–1.015) |
| `M` is a max over 43 profiles; beta had 35 (the L124 twins postdate it) | inflates f | **1.015** (a twin is the max-setter in 4/20 heavy cases) |

### 2.2 Why f falls with n, honestly

The whole-case parallel ratio falls the same way (4.75 → 3.39), so it is in the
raw data, not in the decomposition. The most likely cause is that at small `n`
the graded instance and our in-set instance at the same `n` differ in difficulty
by more than the compute they share — the corpora are matched on `n` only. Treat
the **band** figure as the operative one for a mechanism whose cost lands in that
band, and the spread 1.6…4.7 as the honest precision limit.

## 3. The LP, measured instead of differenced

`ICCAD_LP_TIMING=1` prints per case `cpu`, `wall` and the per-pass times, so the
LP's cost stops being a difference of two ~130 s walls with 8 % session drift.

      arm          LP total (local)   cases running an LP
      shipped           10.54 s              71
      LP k=2            18.93 s              71
      gate0             23.00 s             100
      gate0 + k=2       38.88 s             100

`gate0`'s dt is **+12.46 s**, not the +15.78 s wall-differencing gave — 27 %
smaller. `cpu/wall` is 1.01–1.04 throughout: single-threaded, as §1 requires.

## 4. Pricing chain, with f used only where f belongs

    grader_ship(n) = grader_beta(n) * poolratio(n) + LP_ship(n) / f(n)
    grader_arm(n)  = grader_ship(n)                + dLP(n)     / f(n)

    poolratio(n) = l285_lp_off(n) / l285_betacfg(n)   both LP-free, same session
                                                      => a same-phase ratio, f cancels

`gate0` and `gate0+k=2` change nothing before the LP, so their pool term **is**
the shipped one and the only place f enters is the LP delta. Results are the
table at the top; `l313_final.py` reproduces it.

## 5. What this changes

1. **`f = 3.17` is too high.** Use 2.4–2.8, or the band value. Every arm priced
   at 3.17 (`l293_frontier.py`, `l294_gate.py` §G4, `l297_rank1_price.py`,
   `l300_sensitivity.py`) is optimistic by the ratio of the two.
2. **The f-free ratio method (`l294_final.py` §b) is close to right** — its
   implied 2.37–2.44 sits at the bottom of the global bracket. It understates the
   heavy band, where the true value is 1.6–2.1.
3. **`gate0` survives, `gate0 + k=2` does not.** The depth on top of the ungated
   LP roughly doubles the LP's cost (23.0 → 38.9 s local) for 0.78 pp of extra
   in-set quality, and at the measured f that is a losing trade.
4. **`LP k=2` alone is the most robust arm on the table**: break-even at 0.65–0.86×
   the measured f, 96–99/100 cases still on the RF floor.
5. **Rank 1 is not reachable with anything currently measured.** The best
   projection is `gate0` at 0.8608, **0.26 % short** of 0.858632.

## 5.1 Applied to the parallel session's break-even table (L296–L298)

That session priced four arms and published a break-even `f` for each. Reading
them against the **heavy band** value — which is the right comparison, because
the LP's added seconds land there — settles all four:

      arm                 break-even f   measured heavy-band f 1.62 - 2.13
      lp2   (k=2)             1.03        GREEN, margin 1.6 - 2.1x
      mix   (their pick)      1.56        GREEN, margin 1.04 - 1.37x  <- thin
      gate0                   1.73        STRADDLES (RED at the low end)
      both  (gate0 + k=2)     2.23        RED
      both4 (gate0 + k=4)     4.49        RED, and over the runtime threshold

Their own note says "1.56 is below every value the project has measured (per-case
min 1.79, 2.71, 3.17)". **That is no longer true**: the heavy-band value measured
here is 1.62–2.13, so `mix`'s margin is 1.04–1.37×, not 1.15–2.0×. It still
clears, and it is still the arm with the most headroom against `f` — which is
exactly why it is the right pick.

⚠️ Their break-even numbers apply a single `f` to the whole `dt` vector while the
measurement here says `f` runs 1.62 (heavy) to 4.16 (light). For these arms the
`dt` is heavy-band dominated so the comparison above is close, but a mechanism
whose cost sits on light cases would be much cheaper than a single `f` suggests.

## 6. Traps this cost

1. 🚨 **`solve(..., target_positions=None)` is a different problem.** My first
   driver called `solve()` from my own loop and omitted the preplaced positions;
   the whole run came out **1.7× faster** and the first f I computed (0.92–1.32)
   was garbage. The evaluator builds `opt_target_pos` from the label for
   `preplaced` (x,y,w,h) and `fixed` (w,h) blocks — that is problem input, not
   leakage. Fix: drive the measurement **through the official evaluator** with a
   spy optimizer (`l306_spy_opt.py`), so the inputs are the deployment inputs by
   construction, and gate on reproducing a known total.
2. 🚨 **Another session started a 43-way parallel evaluation in the middle of my
   43-minute uncontended replay.** The replay walks `n` upward, so the
   contamination landed exactly on the heavy band; Σ M there read 17.7 % high. Fix:
   re-measure and merge elementwise with `min` (contention can only inflate).
   **Two sessions on one box cannot both be timing.**
3. `c*` is up to **30.2**, not CLAUDE.md's 22.5. It still clears 48, so the
   one-wave argument holds, but not by the margin the ledger records.

## 7. Files

```
l306_spy_opt.py     spy optimizer, loaded BY the official evaluator
l306_capture.pkl    per case: every profile's (env, stdin), t_serialize, margs
l307_replay.py      serial uncontended replay          -> l307_serial.pkl
l307b_recheck.py    re-measure a band and merge by min
l307c_twins.py      max over 43 vs max over 35 (the L124 twin bias)
l307d_m73.py        today's binary vs the graded M73 one (rebuilt from 7f38893)
l308_f.py           f, the decomposition and the brackets
l309_lptime.py      parse ICCAD_LP_TIMING
l312_lpruns.sh      ship / k2 / gate0 / gate0+k2 with LP timing
l313_final.py       the chain of §4 and the table at the top
constructive_m73.cpp/.exe   the graded binary, rebuilt (probe only)
```

Nothing shipped. `constructive.cpp` md5 `e2c7b2f4…`, `op_wrapper.py` md5
`1c326784…`, both untouched.
