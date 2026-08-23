# L153 — the Linux verify of the L147 config

Closes `HANDOFF_2026-08-20.md` §5.1, the 🔴 line: *"Linux-verify the L147 config —
nobody has."* Nobody had; now somebody has.

**Verdict: all six gates PASS. The L147 config runs on Linux, is feasible
100/100, and regresses nothing against the pre-LP layout. But the gain is
platform-dependent and has to be reported as a RANGE, not the point estimate the
handover carries — and the reason is a single, nameable case.**

Tree: `l147-tangent-cut` @ `4509645` (L137 base + L147). Package re-staged from
that tree for the run and then restored to the tracked 08-17 stage — `git status`
on `build_submission/cadc1075*` is clean, division of labour respected.

---

## 1. What ran

| | |
|---|---|
| Windows | `C:/Users/.01/anaconda3/envs/floorset/python.exe`, py3.10 / np 2.2.6 / scipy 1.15.3, 32 cores |
| Linux | WSL2 Ubuntu, `~/iccadvenv`, **py3.14.4 / np 2.5.2 / scipy 1.18.0** / torch 2.13.0+cpu / shapely 2.1.2, nproc 32 |
| package | freshly staged, `op_wrapper.py` `eb498f9dd25a54a7493532db4050a5df`, `bin/constructive_linux` `bc9912072cd97b45b47a03adec7170ce` (the post-L137 ELF the handover names) |
| flags | `ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0` |

Six gates:

| gate | where | verdict |
|---|---|---|
| G0 `l113_ship_gate.py --cores 48` + the four flags | Windows | **ALL PASS** — total `1.1966792860111928`, **cost-equal 100/100, positions-equal 100/100** vs `results_L147_on_L137.json`; route A peak 31 / queue 32 (the cores gate really fired); no fallback line |
| LANE 1 `final`, default cores | Linux | **PASS**, `|d| = 0.000e+00`, **0 ULP warns** on 100 cases |
| LANE 2 `final48`, `ICCAD_SHAPE_LP=0` | Linux | **PASS**, total `1.260246745790688` — **bit-identical to Windows** |
| LANE 3 `final48`, shipped band | Linux | **PASS**, kept **100/100**, `1.2276727446271392` |
| LANE 4 `final48`, **the L147 config** | Linux | **PASS**, kept 97/100, `1.201017738792057`, feasible 100/100, **0 regressions vs pre-LP** |
| LANE 5 `t4`, corrupt ELF | Linux | **PASS**, falls through to the package's own g++ and lands on the anchor cost bit-for-bit |

Everything is in `l153_wsl_verify.log`, `l153_shipgate2.log`, and the four
`results_L153_linux_*.json`.

## 2. 🔑 The LP is the ONLY thing that moves across platforms — now proven, not assumed

L119 said the 48-core lane is not bit-reproducible and inferred the LP was the
cause. LANE 2 turns that into a measurement:

    48 cores, LP OFF      windows 1.260246745790688
                          linux   1.260246745790688      0/100 movers, |d| = 0

The C++ pool, the M80 knob-cloud tier and route A's frame queue all transfer
**exactly** — across a different libc, a different libm, py3.10 vs py3.14 and a
different multiprocessing start method. Every cross-platform difference in this
package is downstream of `scipy.optimize.linprog`.

That is also why LANE 2 is worth keeping as a permanent gate: it is the only one
of the four that CAN be bit-judged, so it is the one that would catch a real
platform bug hiding behind "the LP is degenerate".

## 3. 🚨 The gain is a range, and 107% of the spread is case 96

| | in-set 100 @48c | vs its own control |
|---|---|---|
| windows ctrl (shipped band) | 1.227176561424409 | — |
| **windows arm (L147)** | **1.196679286011** | **+2.4852%** |
| linux ctrl (shipped band) | 1.2276727446271392 | — |
| **linux arm (L147)** | **1.201017738792057** | **+2.1712%** |

The mechanism transfers cleanly — **83/100 cases have an identical per-case
gain**, 94 vs 95 cases improve, 6 vs 5 are hurt. The whole difference is in the
degenerate ties, and weighted by the actual score
(`Σ cost·e^{n/12} / Σ e^{n/12}`) it is one case:

    case  96 n=117  w=0.7788  raw +0.074542  ->  weighted +0.004642804   (107.0% of the gap)
    case  67 n= 88  w=0.0695  raw -0.101610  ->  weighted -0.000564640   (-13.0%)
    case  73 n= 94  w=0.1146  raw +0.044641  ->  weighted +0.000408991   (  9.4%)
    ... everything else nets to about -4%

⚠️ **The raw mover list points at the wrong case.** Sorted by |Δcost| the
headline is case 9 (n=30, +0.158) — but n=30 carries weight `e^{-7.5}` = 5.5e-4
and contributes **0.2%** of the gap. This is `HANDOFF_2026-08-20` §4.1 (light-band
screening) in a new location: the exponential weighting has to be applied before
anything is called a mover.

### What actually happened to case 96

It is not "a different optimum". It is a **rejection**:

    case 96  n=117   pre-LP 1.215357 | ctrl W 1.186644  L 1.186644 | arm W 1.140815  L 1.215357
                     arm-vs-preLP:   W -0.074542        L +0.000000

`+0.000000` exactly. On Linux `hard_ok` refused the tangent arm's layout on case
96, `kept=0`, and the case fell all the way back to the **pre-LP** layout —
throwing away not just the tangent increment but the whole shipped LP gain of
0.0287 that the control keeps on the same case.

The kept-rate counter says the same thing independently:

    linux ctrl : 100 cases, kept 100/100, rejected at n=[]
    linux arm  : 100 cases, kept  97/100, rejected at n=[31, 42, 117]
    windows arm:            kept  98/100, rejected at n=[31, 42]

Cases 10 (n=31) and 21 (n=42) are rejected on **both** platforms — they are a
property of the config, not of the platform. Case 96 (n=117) is the only
acceptance that differs, and it is the high-weight one.

**So the honest in-set range is the coin flip on that one case:**

| | if case 96 is kept | if case 96 is rejected |
|---|---|---|
| windows | +2.4852% (measured) | +2.1068% |
| linux | +2.5494% | +2.1712% (measured) |

**Report L147 as +2.11% ~ +2.55% in-set, not +2.49%.** Carrying the same spread
through the handoff's RF cost of −0.9726% and its OOS gains (+2.2416% / +2.1406%)
puts NET at roughly **+0.85% ~ +1.45%** rather than +1.269% / +1.168%. The bar is
0.30%, so the decision does not change at either end — but the number quoted to
the teammate should be the range.

## 4. The measured prize this exposes: catch a rejection with the shipped band

Today a rejected case falls from the tangent arm straight to **pre-LP**. It could
fall to the **shipped band** instead — that layout is already computed for the
same case and is known-good. Upper bound, from the runs on disk:

| | arm as-is | arm + band-catch |
|---|---|---|
| windows (2 rejections, n=31/42) | +2.4852% | +2.4861% (**+0.0009pp**) |
| linux (3 rejections, n=31/42/117) | +2.1712% | **+2.3178% (+0.1466pp)** |

The mean gain is small; the point is **variance**: it bounds the downside of
every future rejection at "no worse than what is already shipped". Cost is one
extra LP solve on the 2–3 cases per 100 that reject.

⚠️ Two guesses in this paragraph were wrong and L154 measured both. The spread
does not collapse to 0.03pp — it halves, 0.3140pp → 0.1683pp. And the retry is
not "inside the noise floor": stacked on L147's own per-case dt, a big-n rescue
costs **RF −0.0560%**, because that case has 0.864s of headroom left and the
retry needs 0.935s.

**IMPLEMENTED AND PRICED 2026-08-21 -> `L154_REPORT.md`.** Measured +0.1498% on
Linux, +0.0009% on Windows, +0.0356% / +0.0532% OOS, 0 cases worse in 680
case-runs, spread 0.3140pp -> 0.1683pp. The upper bound above was right: the
Linux total was predicted at 1.1992182077469893 and measured at
1.1992182077469895. But the RF cost is NOT zero once it is stacked on L147's
own dt -- a big-n rescue costs -0.0560%, NET +0.0841% per event.

## 5. 🚨 The gate that could not run, and why it looked fine

`l113_ship_gate.py --cores 48` **FAILED** on the first attempt with
`total=9.999916545892749` next to `feasible=100/100`.

Cause: the package deliberately ships no Windows binary, so the Windows gate must
compile `constructive.cpp` on site. `C:\msys64\ucrt64\bin\g++.exe` is installed
and is the first candidate in `_ensure_compiled()`, but **its own directory was
not on PATH**, so the exe could not load its DLLs and exited 1 with *completely
empty stderr*. Every case then fell to the pure-Python SA.

    PATH without C:\msys64\ucrt64\bin   ->  g++ exit 1, stderr empty, total 9.9999, "feasible 100/100"
    PATH with    C:\msys64\ucrt64\bin   ->  g++ exit 0, total 1.1966792860111928, ALL PASS

This is the `windows-msys-path-silent-sa-fallback` note hitting a **gate** rather
than a measurement. Two fixes landed in `l113_ship_gate.py`:

* `_BAD_RE` now includes `"[constructive]"`. Every print carrying that tag is a
  degradation notice and there is no benign one; without it G2 surfaced only the
  last line of the cascade and swallowed the two `[constructive] <g++> -O3
  failed:` lines directly above it that named the cause.
* `_cxx_preflight()` probes for a working compiler before the run, prepends the
  msys64 dir if that is what is missing, and says so in one line. On the grader
  this branch never executes — POSIX takes `bin/constructive_linux` and skips the
  compile entirely — so it is purely about keeping the local gate meaningful.

## 6. What changed in the tools

* **`l117_linux_verify.py` rewritten** (see its docstring). Three holes closed:
  `--env K=V` past the `ICCAD_*` strip (without it `final48` would have measured
  the shipped band, compared it to a shipped-band anchor and printed PASS while
  saying nothing about L147 — `HANDOFF_2026-08-20` §4.3 in a second location);
  `--stats` forcing `ICCAD_SHAPE_LP_STATS` and gating on the kept-rate (scipy
  absent, `_LP_IMPORTS_OK` False, or a malformed flag — `_shape_lp` swallows
  `ValueError` and drops the whole tangent dict — all disable the LP with no
  stderr line at all); and `judge48` taking its anchors on the command line with
  a **liveness gate** against a control arm.
* **`judge48`'s invariant is now satisfiable.** The old "no case worse than the
  pre-LP anchor" was unsatisfiable as written because its anchor was four
  milestones stale. Rewritten as *"no case worse than the pre-LP base by more
  than `--budget`"*, with the budget **measured** by `l153_budget.py` from the
  already-shipped band on the same base and the same platform. On Linux that
  budget came out **0.000000000** — the shipped band regresses nothing against
  its own pre-LP base — so the arm was judged at 0 and passed at 0.
* **`l153_anchors.sh`** regenerates the two Windows anchors on the tree under
  test. The pre-existing `results_L147_lpoff.json` / `results_L136_default.json`
  were taken on the **L136** base; reusing them would have priced L137 into this
  verdict (`HANDOFF_2026-08-20` §4.4).
* **`l153_xplat.py`** does the weighted cross-platform decomposition in §3.

## 7. Reproduce

```bash
cd /c/ICCAD_ml/ship_final && bash l153_anchors.sh
```
```bash
cd /c/ICCAD_ml/ship_final && PATH="/c/msys64/ucrt64/bin:$PATH" "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l113_ship_gate.py --cores 48 --anchor results_L147_on_L137.json --env ICCAD_SHAPE_LP_R=1.5 --env ICCAD_SHAPE_LP_G=1.10 --env ICCAD_SHAPE_LP_TOL=0.006 --env ICCAD_SHAPE_LP_PRICE=1.0
```
```bash
wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l153_wsl_verify.sh
```
```bash
cd /c/ICCAD_ml/ship_final && "C:/Users/.01/anaconda3/envs/floorset/python.exe" l153_xplat.py
```

## 8. Housekeeping

* `build_submission/cadc1075*` was re-staged for the run and **restored
  byte-identical** to the tracked 08-17 stage afterwards.
* `FINAL_LINUX_VERIFY_RUNBOOK.md` is stale in three ways: it says this box has no
  WSL (it does), it points at `C:\Users\Nordra\...` paths that do not exist here,
  and its anchors are M74-era. `l117_linux_verify.py` + `l153_wsl_verify.sh`
  supersede it.
* ⚠️ **C: is at 1.7 GB free / 100% used.** Nothing failed because of it, but a
  full disk during the last week before 08-28 would be an ugly way to lose a run.
