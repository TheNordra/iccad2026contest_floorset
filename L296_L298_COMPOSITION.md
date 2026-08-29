# L296–L298 — the gate and the depth compose, and the best arm is neither of them alone

The question: does `ICCAD_LP_GATE=0` compose with LP `k=2`? It does, and the
composition is **super-additive in quality** — but the naive "turn both flags on"
is **not** the right form of it. The best arm is a third one the composition
itself exposes.

    in-set 100, official evaluator, ADAPTIVE_CORES=48
    dt measured in ONE contiguous block: ship, mix, both, ship, gate0, ship,
    both4, ship  (four ship runs, wall spread 1.3 %)

      arm        total        quality    dt s   NET@3.17  NET ratio  grader  feas
      ship    1.226325126     0.0000%    0.00    0.0000%   0.0000%   45.2 s   100
      lp2     1.222554152    +0.3075%    7.67   +0.3033%  +0.2988%   49.0 s   100
      gate0   1.199000373    +2.2282%   13.17   +1.5699%  +1.1057%   50.8 s   100
      mix     1.195229398    +2.5357%   22.34   +1.9470%  +1.4975%   53.4 s   100
      both    1.189471885    +3.0052%   28.39   +1.5322%  +0.8274%   56.1 s   100
      both4   1.187314246    +3.1811%   62.22   -2.2206%  -3.1844%   67.9 s   100  OVER

      break-even f :  lp2 1.03   gate0 1.73   mix 1.56   both 2.23   both4 4.49
      ⚠️ measured f is 1.62-2.13 on the heavy band (L308) — see §5.5;
         the NET@3.17 column above is priced ABOVE the measured range

**`mix` wins on both pricings and has the lowest break-even `f` of the three big
arms** — it is simultaneously the highest-scoring and the least sensitive to the
one constant that is still bracketed.

---

## 1. The two knobs are independent, and the mechanism says so bit-for-bit

`_lp_gate_ok(n)` decides **whether** the LP runs on a case (71 → 100 block
counts). `_shape_lp_depth()` decides **how many passes** where it does. That
makes exact predictions, and they are gates rather than hopes:

| prediction | in-set | s1 | s2 |
|---|---|---|---|
| on the 71 gated-ON `n`: `gate0` == `ship` | **71/71** | **166/166** | **166/166** |
| on the 71 gated-ON `n`: `both` == `lp2` | **71/71** | **166/166** | **166/166** |
| on the 29 gated-OFF `n`: `lp2` == `ship` | **29/29** | **74/74** | **74/74** |

(cost **and** positions, in set). So the two act on **disjoint case sets**, the
decomposition has no residual, and **arm-mixing is exact** — the same
justification L172/L196 relied on.

## 2. The decomposition, and the third arm it exposes

    quality                                       time
    depth, 2nd pass on the 71    +0.3075 pp        9.11 s   -> 0.0338 pp/s
    the gate, 1st pass on the 29 +2.2282 pp       13.16 s   -> 0.1693 pp/s
    CROSS TERM, 2nd pass on t.29 +0.4695 pp        7.67 s   -> 0.0612 pp/s
    ------------------------------------------------------
    both                         +3.0052 pp       28.39 s

`both` is **super-additive**: +3.0052 % against the naive sum of +2.5357 %. The
cross term is real — a second pass on the 29 cases the gate newly admits, which
neither arm alone can reach — and it survives on both held-out samples.

**`mix`** is the arm that takes the first two terms and drops the third: *LP
everywhere at k=1, a second pass only on the 71 where L196 judged the case can
afford it.* Predicted exactly by arm-mixing, then **measured end to end**:

    mix == lp2   on the 71 : 71/71   (cost and positions)
    mix == gate0 on the 29 : 29/29
    mix determinism        : 100/100
    arm-mixed prediction 1.195229398  vs MEASURED 1.195229398   diff 0.00e+00

## 3. Why `both` loses to `mix` — and it is not what it first looked like

The cross term is **good value per second in isolation** — 0.0612 pp/s, nearly
**2×** the depth pass it sits next to. It loses at the margin for a different
reason: **its 7.67 s land entirely on the 29 heavy cases, which have the least
RF slack.** The same seconds spent on light cases are nearly free; spent there
they push cases off the 0.7 floor. NET falls 1.947 % → 1.532 %.

⚠️ **A correction to this session's own first reading.** The first pass at this
reported the cross term at **22.52 s** and `both` as **RED (−0.17 %/−0.96 %)**.
Both were artefacts of differencing against `l294_ship*.json`, three hours older
than the arms. `both` is **GREEN**; the ranking `mix > both` survives, for the
reason above rather than the one first given. The error was caught by §1's
same-work identity, not by inspection — see §4.

## 4. 🚨 The gate that caught it: identical work must cost the same

§1 proves `mix` does bit-identical work to `gate0` on the 29 and to `lp2` on the
71. So those halves **must** cost the same seconds, and the difference is a
direct readout of baseline drift:

    against the stale l294 baseline     mix - gate0 (on 29)   -1.27 s
                                        mix - lp2   (on 71)   -1.29 s
                                        both - lp2  (on 71)   +7.93 s
    in the contiguous block             mix - gate0 (on 29)   -0.01 s
                                        mix - lp2   (on 71)   +0.08 s

Consistent on **both** halves is a **bias**, not scatter. The correction is
large: `both`'s dt 47.71 → 28.39 s, `both4`'s 74.65 → 62.22 s.

🔑 **A bit-equality between two arms is not only a correctness check — it is a
free control for wall-clock drift.** Whenever two arms are proven to do the same
work on a subset, the measured cost difference on that subset is pure noise, and
its size is the error bar on every dt in the table. Nothing else in this ledger
supplies that for free.

This is the handoff's own lesson in a new costume: *"the earlier k=4 figure used
the L276-era dt; re-measured back to back it is 27.57 s. Use the fresh number."*

## 5. Quality on all three corpora

Base-denominator convention throughout (`l276_price.quality_pct`), so these are
directly comparable to every priced number here.

| arm | in-set | s1 | s2 | transfer |
|---|---|---|---|---|
| lp2 | +0.3075 % | +0.4819 % | +0.4867 % | >100 % |
| gate0 | +2.2282 % | +2.4055 % | +2.1883 % | 108 % / 98 % |
| **mix** | **+2.5357 %** | **+2.8874 %** | **+2.6750 %** | **114 % / 105 %** |
| both | +3.0052 % | +3.2905 % | +3.0681 % | 110 % / 102 % |

### L299 — `mix` run end to end on both held-out samples

Its s1/s2 figures above were arm-mixed. Now measured, 240 cases per sample,
through the real `l296_mix_optimizer.py`:

| | s1 | s2 |
|---|---|---|
| per case identical to the arm-mixed prediction | **240/240** | **240/240** |
| weighted total, predicted vs measured | 1.427809826 vs 1.427809826, **diff 0** | 1.426058312 vs 1.426058312, **diff 0** |
| quality (base denominator) | **+2.8874 %** | **+2.6750 %** |
| errors / infeasible | **0 / 0** | **0 / 0** |
| vs `ship` | **216 better / 0 worse** | **218 better / 0 worse** |

So the mixing was exact, and §1's disjointness argument is confirmed by
measurement rather than only by construction. Movers also land exactly where the
decomposition says: 216 = 73 (`gate0`) + 143 (`lp2`) on s1, 218 = 74 + 144 on s2.

**Zero regressions across 434 held-out movers**, on top of the in-set 29/29.
All three bands positive in both samples.

⚠️ Two percent conventions are in play in this ledger and differ ~2 % relative:
`l276_price.quality_pct` divides by the **base** total, `l287_transfer.py` by the
**arm** total. Everything above uses the base form.

## 5.5 🚨 RE-PRICED against L308's measured `f` — and `gate0` alone falls out

A parallel session on this tree **measured `f` directly** (`L302_L313_F_PINNED.md`):
**2.38–2.84 overall, and only 1.62–2.13 on the heavy band** — the tree's 3.17 is
above the entire range. That session already read my break-even table against
its heavy-band value, and correctly flagged the remaining approximation: I applied
**one** `f` to a `dt` vector whose seconds span bands where `f` runs 1.62 to 4.16.

Removing that approximation — per-block-count `dt` divided by that band's own `f`,
at both ends of the measured interval:

      arm      quality     dt s |  NET @ f_lo  grader |  NET @ f_hi  grader
      lp2     +0.3075%     7.67 |   **+0.3016%**  50.5s |   +0.3032%   49.5s
      gate0   +2.2282%    13.17 |   **-0.1269%**  52.3s |   +0.9349%   50.9s
      mix     +2.5357%    22.34 |   **+0.3852%**  56.1s |   +1.3436%   53.9s
      both    +3.0052%    28.39 |     -1.6390%  59.2s |   -0.0688%   56.4s
      both4   +3.1811%    62.22 |     -8.4095%  76.4s |   -5.2605%   70.1s  OVER

      bands: 4.16 / 2.66 / 2.31 / 1.62 (f_lo) on n = 21-50 / 51-80 / 81-100 / 101-120

🔑 **`mix` is the only large arm that survives the conservative end**, and it
clears the 0.30 % bar there (+0.385 %). **`gate0` alone goes NEGATIVE at `f_lo`**
— the +0.91…+1.42 % this report and L294 quoted was priced at or above the top of
the measured range.

**Why `mix` beats `gate0` at *both* ends, and by more at the conservative one:**
the seconds `mix` adds on top of `gate0` are the k=2 second pass, and **−0.4 % of
`lp2`'s `dt` lands on the heavy band** — they are spent where `f` is 2.66–4.67,
i.e. where a local second is cheapest. `gate0`'s own seconds are **49 % heavy**,
where `f` is 1.62. So `mix` is not merely "gate0 plus more"; it is gate0 plus the
cheapest seconds on the table.

⚠️ My same-box ratio column (gate0 +1.11 %, mix +1.50 %) lands near `f_hi`.
L308's reading is that the ratio method sits at the low end of the *overall* range
but **overestimates the heavy band** — which is where these arms spend. Treat the
`f_lo` column as the number to defend.

✅ **My `dt` itself survives this**: the contiguous block's four `ship` runs spread
**1.05 s (0.8 %)** and the same-work identity closed to **−0.01 / +0.08 s** (§4), so
the vector being divided is sound. Only the divisor changed.

## 6. Deployment

`mix` needs **two code defaults**, both wrapper tables, no ELF rebuild:

    _L196_LPGATE  ->  all 1s                    (LP runs everywhere)
    _L157_DEPTH   ->  2 on the old 1-set, 1 on the old 0-set

`_depth_ok`'s shipped path is `pass_no <= _L157_DEPTH.get(n, 1)` —
**deterministic, per block count** — so every bit-equality gate keeps working.
`l296_mix_optimizer.py` is that wrapper, generated as a copy; the tree is
untouched and was verified so.

⚠️ **An env var ships nothing** (L158): the grader strips every `ICCAD_*`.

## 6.5 L300 — the Linux lane: **five lanes, all PASS**

`build_submission.MIX/cadc1075.tar.gz` = the shipped package with the two tables
changed and **the same ELF, byte for byte** (so the glibc-2.43 rebuild floor in
the ledger does not arise — that risk exists only when the ELF is rebuilt here).
WSL2 Ubuntu, nproc 32, py 3.14.4 / numpy 2.5.2 / **scipy 1.18.0** vs Windows'
1.15.3.

| lane | what it proves | result |
|---|---|---|
| **1a** ship @ default cores | the bundled ELF runs on Linux and matches Windows | `1.280025696987226`, **\|d\| 2.2e-16**, 100/100 cost **and** positions, 0 ULP warns |
| **1b** mix @ default cores | mix is **inert below the ≥40-core gate** | **bit-identical to 1a**, 100/100 |
| **2** ship @48c, `SHAPE_LP=0` | the Linux pre-LP base | `1.2589744529416786`, **−0.0000 %** vs the Windows LP-off anchor |
| **3** ship @48c | the control, and the budget | `1.2264069637381392` — **exactly the handoff's recorded 48c Linux value**; LP ran on **exactly the 71** gated block counts; **0 regressions** ⇒ budget **0** |
| **4** **mix @48c, judged** | `judge48()` invariants | **PASS**: feasible **100/100**, **0 regressions vs pre-LP at budget 0** (the strictest form), **+2.4056 %** over the shipped band; LP ran on **exactly 100/100** |
| **5** t4 on the mix tar | corrupt ELF must fall through to the package's own g++ | case 50 reproduces the anchor to **0.000e+00** |

Lane 4's LP-liveness line is the Linux counterpart of the in-set G0: `71 -> 100`
block counts, measured from the package's own stats file, not inferred.

### The cross-platform spread, and what it does to the price

30/100 cases differ from Windows by >1e-9 (worst 1.24e-01 on case 55) — the known
L119 scipy/HiGHS divergence on a degenerate LP, which is why this lane is judged
on invariants rather than bit-equality. It costs a little quality:

      Windows 48c   ship 1.226325126 -> mix 1.195229398   +2.5357 %
      LINUX   48c   ship 1.226406964 -> mix 1.196905117   **+2.4056 %**   <- the grader's platform
      Linux realises 95 % of the Windows gain

Re-priced on the Linux quality with the same `dt` and the per-band `f`:

      quality from    quality     NET@f_lo    NET@f_hi   grader
      Windows       +2.5357%     +0.3852%    +1.3436%    56.1 s
      **LINUX**     **+2.4056%** **+0.2551%** **+1.2134%** 56.1 s

⚠️ **Linux × f_lo is the one corner that lands under the 0.30 % bar** (+0.2551 %).
Three of the four corners clear it. The honest statement is **NET +0.26 % …
+1.34 %**, positive throughout, clearing the bar everywhere except the
simultaneous worst case of both remaining uncertainties.

## 6.7 L301 — deepening ONLY on the 71: **RED**, and the depth axis closes

`both4` had shown uniform depth 4 is RED, but it spends those passes on the 29
heavy cases where the measured `f` is 1.62. This shape spends them on the light
and mid ones where `f` is 2.66–4.67 — and §5.5 showed the band is what decides
the price. Two new wrappers, `_L196_LPGATE` all 1s and `_L157_DEPTH` = 3 / 4 on
the old 1-set:

      arm           total       quality  | NET@f_lo  grader | NET@f_hi  grader
      mix  (k=2) 1.195229398   +2.5357%  | +0.0238%   57.1s | +1.1567%   54.7s
      mix3 (k=3) 1.194365588   +2.6061%  | -0.8640%   63.5s | +0.9794%   59.9s
      mix4 (k=4) 1.194116345   +2.6264%  | -1.8717%   66.9s | +0.2889%   62.7s  OVER

      k=2 -> k=3  +0.0704 pp for +13.32 s = **+0.0053 pp/s**
      k=3 -> k=4  +0.0203 pp for  +7.83 s = **+0.0026 pp/s**
      the gate's own 1st pass on the 29   = **+0.1583 pp/s**

**30–60× worse per second than the gate's first pass**, NET monotonically down,
and `mix4` crosses the runtime threshold. **`k=2` is the optimum on the 71 as
well** — so the depth axis is closed in *both* shapes, and what saturates is the
LP itself, not the band its seconds land in.

`mix4` was predicted exactly before it was run: `l293_k4.json` is
`ICCAD_SHAPE_LP_ITERS=4` with the gate still ON, i.e. k=4 on the 71 and the 29
skipped — mix4's depth component. Arm-mixed prediction **1.194116345** vs
measured **1.194116345**, diff **0.00e+00**, with all four bit-equality gates
passing. All arms 100/100 feasible.

### 6.7.1 🚨 The wall-clock instrument had to be replaced, and then calibrated

The first L301 block's **control failed**: `mix` does bit-identical work in it
and in the L298 block (total 1.195229398 in both, every same-work gate passing)
yet its `dt` read **35.36 s vs 22.34 s**, and `mix3` read *cheaper* than `mix`
while doing strictly more passes. The box was loaded (three `ship` runs spread
6.0 %, against 0.8 % in L298).

So the LP is now timed **inside the process** with `ICCAD_LP_TIMING=1`, which
L159 built for exactly this. `cpu/wall` came back **1.008–1.045** — single
threaded, nothing stealing it.

That was still not enough. Applying the same-work identity **to the LP clock**:

      gate0 vs ship , k=1 on the 71 : 10.70s vs 14.33s   **-25.3 %**
      mix   vs lp2  , k=2 on the 71 : 22.07s vs 22.58s     -2.3 %
      both  vs lp2  , k=2 on the 71 : 22.69s vs 22.58s     +0.5 %
      mix   vs gate0, k=1 on the 29 : 15.11s vs 15.18s     -0.5 %

One k=1-on-71 observation was 25 % off while the k=2 triple agreed to 2.8 %. The
fix is the ledger's own rule, applied at the right granularity: the arms are
compositions of **six distinct work units**, each observed by several runs, so
`min-of-N` goes **per unit**, not per arm.

      k1_71  5 obs, spread 34.0 %      k1_29  5 obs, spread 11.6 %
      k2_71  4 obs, spread  3.6 %      k2_29 / k3_71 / k4_71  1 obs each

🔑 **The noisiest unit is `k1_71` — and it is the BASELINE, so its error
propagates into every arm's `dt`.** That is why `mix`'s NET@f_lo has moved three
times in this session (+0.3852 % wall-differenced, −0.2003 % single LP clock,
**+0.0238 %** per-unit min-of-N). The relative verdict never moved; the absolute
level is now limited by `dt` measurement, not by `f`.

⚠️ **Superseded by §6.8**, which removes the cross-run differencing entirely and
puts `mix`'s conservative corner back at **+0.3933 %** (Windows) / **+0.2632 %**
(Linux). The arms' *ordering* — `mix` > `gate0` > `mix3` > `both` > `mix4` — is
stable across all four instruments.

## 6.8 L302 — `dt` pinned, and the noisy unit turns out not to matter

The unit that had been moving every number was `k1_71`: the LP's first pass on
the 71, 34 % spread over five observations, and the **baseline** of every
cross-run difference. Re-measured five more times — and then found to be
**irrelevant**:

    ship pass 1 on the 71, 5 runs : 10.14  10.97  12.79  12.49  11.03
    mix  pass 1 on the 71, 5 runs : 12.07  11.34  10.38  11.80  10.94

🔑 **Every arm runs pass 1 on the 71 exactly as `ship` does** (bit-for-bit,
§1), so the term **cancels**. It only ever entered the numbers because whole-LP
walls were being differenced across runs. Drop the differencing and each arm's
`dt` becomes **self-contained, measurable inside one process in one run**:

    gate0 dt = the LP's wall on the 29           (ship spends 0 there)
    mix   dt = that, plus pass 2+ on the 71      (pass 1 cancels)

which is verbatim what L159 built `_LAST_PASS_DT` for: *"Timing both passes in
the SAME process removes the drift entirely instead of trying to average it
away."*

**Second correction: `min-of-N` is biased downward with N**, so an arm with five
repeats cannot be compared against one with one. Pooled by work unit — licensed
by the same bit-equality that licenses arm-mixing for quality:

    k1_29  9 observations  13.04 s      p2_71   8 observations   8.62 s
    k2_29  1               24.46 s      p23_71  1               20.68 s
                                        p234_71 1               28.63 s

### The final table

      arm        total       dt s  on 29 | NET win lo   win hi | NET LNX lo  LNX hi | grader
      ship    1.226325126    0.00   0.00 |   +0.0000%  +0.0000% |  +0.0000% +0.0000% | 45.2s
      lp2     1.222554152    8.62   0.00 |   +0.3068%  +0.3070% |  +0.2910% +0.2912% | 49.2s
      gate0   1.199000373   13.04  13.04 |   +0.0865%  +1.0456% |  -0.0278% +0.9313% | 51.8s
      **mix** 1.195229398   21.66  13.04 | **+0.3933%** **+1.3526%** | **+0.2632%** **+1.2225%** | 55.8s
      mix3    1.194365588   33.72  13.04 |   -0.0665%  +1.3360% |  -0.2002% +1.2023% | 61.6s
      both    1.189471885   33.09  24.46 |   -2.7775%  -1.0263% |  -2.9317% -1.1805% | 61.5s
      mix4    1.194116345   41.67  13.04 |   -1.0963%  +0.7432% |  -1.2310% +0.6084% | 65.1s OVER

**`mix` is NET +0.26 % … +1.35 %, positive in all four corners** of
{Windows, Linux} × {f_lo, f_hi}; three of the four clear the 0.30 % bar and the
Linux × f_lo corner sits at +0.2632 %. `gate0` alone straddles zero at the
conservative end. `mix3` is still behind `mix` where it matters (the
conservative end) even though the two converge at f_hi.

### What the excursion was worth

`mix`'s NET@f_lo (Windows) across four instruments in one session:

    +0.3852%   wall-differenced, contiguous block   (L298)
    -0.2003%   single LP clock, inflated ship baseline
    +0.0238%   per-unit min-of-N on whole-LP walls
    **+0.3933%**   self-contained, pooled per unit    <- final

It came back to within **0.008 pp** of where the L298 block had it. The L298
measurement was sound; the two excursions were instrument artefacts, and both
were caught by the same-work identity rather than by inspection. **The ordering
of the arms never moved under any of the four.**

## 7. What is NOT measured

1. ~~**`f` remains bracketed.**~~ **Superseded by §5.5**: `f` is now measured, and
   `mix`'s break-even 1.56 sits against a heavy-band 1.62–2.13 — a margin of
   1.04–1.37×, not the 1.15–2.0× claimed here. It still clears, and it is still the
   arm with the most headroom against `f`, which is what makes it the pick.

## 8. Files

```
l296_compose.sh      in-set both/both_r2/both4 + OOS s1,s2
l296_price.py        the decomposition, the additivity test, the G5 mix price
l296_mix_optimizer.py   the mix wrapper (a copy of the tree's, two tables changed)
l297_mix.sh          mix end to end, x2
l298_clean.sh        the contiguous block; l298b_k4.sh  both4 re-measured
l298_price.py        the final table, the same-work dt gate, break-even f
l301_deepen.sh / l301b_lptime.sh / l301c_rest.sh / l301d_minofn.sh
l301_mix{3,4}_optimizer.py       depth 3 / 4 on the 71 only
l301_final.py                    per-work-unit min-of-N pricing
l302_selfcontained.sh / l302_pin.py / l302_final.py   the pinned dt
l300_win32.sh / l300_wsl_mix.sh   the Linux lane (5 lanes)
build_submission.MIX/            the mix package, same ELF
l300_linux_*.json                the Linux results, copied back
l287_transfer.py     + the `both` and `mix` arms.  `mix` swaps the MODULE, not
                     the environment, so set_arm() asserts the module's TABLES
                     (all-1s gate, depth histogram {1:29, 2:71}, _depth_ok on
                     both sides) -- a flag-liveness check is blind to this arm.
l299_mix_oos.sh      mix end to end on s1 and s2; l299_s1.log / l299_s2.log
```

Nothing shipped or modified. `constructive.cpp` md5 `e2c7b2f4…`,
`op_wrapper.py` md5 `1c326784…`, both unchanged.


---

## CORRECTION (2026-08-28, later): the compiler was never broken

Section 6.5 reported `C:\msys64\ucrt64\bin\g++.exe` as unable to compile anything -- "exit 1, zero
output, cc1plus present but not starting, probably a half-finished msys2 upgrade".
That diagnosis was **wrong**.

The compiler works. `cc1plus.exe` lives under `lib/gcc/...` and loads its DLLs from
`ucrt64/bin`; invoking `g++` by ABSOLUTE PATH finds g++'s own DLLs (same directory)
but leaves `cc1plus` unable to load its own, so it dies silently and `g++` returns 1
with no diagnostic. Put the msys `ucrt64/bin` directory on PATH and it compiles:
exit 0, 131,230 bytes.

This is exactly the ledger's [[windows-msys-path-silent-sa-fallback]] -- msys
installed but not on PATH, compile fails, package degrades silently to the Python
SA. I re-derived the symptom and mis-attributed the cause. The CONSEQUENCE I
reported stands unchanged: running the package on Windows without PATH (and without
`ICCAD_CONSTRUCTIVE_BIN`) silently yields the SA fallback's
`Total Score 10.0000 / Feasible 100/100`. Only the cause was wrong.

Bearing on the shipped package: the parallel session's patch dropped the hardcoded
absolute msys path from the compile chain, commenting "a bare g++ resolves to the
same msys binary". WITH ucrt64/bin on PATH that comment is correct; without it,
neither form works, because the failure is cc1plus rather than the driver. Either
way the grader is Linux and uses the bundled ELF, so the removal is cost-free there.
