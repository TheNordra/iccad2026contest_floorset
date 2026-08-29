# `build_submission.RFSAFE` — the RF-SAFE ungate candidate (L312)

**This is a CANDIDATE. Nothing here has been uploaded.** The uploaded package is
still `build_submission.D` (`op_wrapper.py` md5 `1c326784…`), untouched.

## What it is

The shipped package with **one wrapper table changed and nothing else**:

    _L196_LPGATE  71 -> 83 on
      newly ungated: [38, 40, 56, 76, 79, 81, 94, 95, 107, 108, 114, 120]

Those 12 block counts are exactly the ones whose **measured added grader time
fits inside that case's own slack to the RF floor**, so they are the part of the
LP-gate gain that carries (almost) no RF bill. Selection uses **no quality
information** — it is the L157 shape, so there is nothing to over-fit.

`constructive.cpp` and `bin/constructive_linux` are **byte-identical** to the
shipped package ⇒ **no ELF rebuild, no Linux binary work.**

| | |
|---|---|
| `op_wrapper.py` / `op_src.py` | **`62db6ee4569b31ddc8c546ccf3e7cd0b`** |
| `constructive.cpp` | `e2c7b2f418ef2b70b6bff99f7adfbd37` (== shipped) |
| `bin/constructive_linux` | `bc9912072cd97b45b47a03adec7170ce` (== shipped) |
| `requirements.txt` | `6c59feb458f1f48247373d8b69f401c2` (== shipped) |
| tar | 409,589 bytes, 8 members, **no vendor/** |

## What it scores

    in-set 100, 48c    1.226325126 -> 1.215239132    +0.9040 %
    12 cost movers, 12 better, 0 worse, feasible 100/100
    default cores      1.280025697 == shipped, cost 100/100 AND positions 100/100

RF bill from the **measured** dt of those 12 (6.40 s local added), across the
joint f·m axis — f is machine speed, m is median drift, and only the product is
identifiable (L310):

    f_eff   3.17     2.84     2.61     2.38     2.10     1.90     1.77     1.60
    RF    +0.007%  +0.014%  +0.029%  +0.108%  +0.267%  +0.425%  +0.597%  +0.858%
    NET   +0.897%  +0.890%  +0.875%  +0.796%  +0.637%  +0.479%  +0.307%  +0.046%

L308 measured f = 2.38–2.84; f_eff 2.10 is "the medians shrink again as much as
they did between the two beta publications". **NET stays positive across the
whole range and above the 0.30 pp ship bar down to f_eff ≈ 1.77.** The full
ungate (`ICCAD_LP_GATE=0`) and `mix` both flip negative in that same region.

## Verification actually run (L312)

| lane | result |
|---|---|
| `make_submission.py stage` | **PASS** — binary/source OK, vendor byte-identical to the wheel, hygiene OK |
| no-vendor tar | 8 members, 0 vendor entries, member list == D |
| `l246_compliance.py` | **20 / 20** |
| diff vs D | **4 hunks**, all in `op_wrapper.py`/`op_src.py`: the L240 probe (pre-existing, default OFF), the msys line + its comment (the compliance patch), and `_L196_LPGATE`. Everything else byte-identical |
| Windows 48c | **1.215239132**, feasible 100/100, **12 movers / 12 better / 0 worse**, **0 movers outside the 12 ungated block counts** |
| Windows default cores | **1.280025697**, `\|d\| = 0.000e+00`, cost 100/100 **and positions 100/100** — inert below the ≥40-core gate |
| SA fallback | 0 lines in both runs |

`python l312_verdict.py` → **ALL PASS**.

## OOS — positive on BOTH held-out samples (L275's rule satisfied)

`l312_rfsafe_oos.sh` → `l287_transfer.py --arms ship,gate0,rfsafe`, 240 cases each,
`s1` = floorset_lite worker_0..9, `s2` = worker_10..19 (disjoint). All arms
feasible in every case. `gate0` rides along as the control and **reproduces its
recorded L295 values exactly** (s1 +2.4648 %, s2 +2.2373 %), so the harness is
measuring what it did before.

```
arm           in-set        s1        s2
gate0        2.2282%   2.4648%   2.2373%
rfsafe       0.9040%   1.1122%   1.2265%      <- 32/240 movers on both samples
retained       40.6%     45.1%     54.8%      <- share of the full ungate kept
```

Per band (`rfsafe`): s1 light +0.633 % / mid +0.978 % / **heavy +1.129 %**;
s2 light +0.752 % / mid +0.876 % / **heavy +1.270 %**. The heavy band is the
strongest on both samples, which is where the `exp(n/12)` weight actually is.

🔑 **Transfer is 123 % (s1) and 136 % (s2)** — this arm is *better* out of sample
than in it. That is the expected signature of a selection rule that uses no
quality information: there was nothing to over-fit, so nothing decays. Contrast
the project's usual 46–93 % transfer on quality-selected mechanisms.

## 🚨 Linux realises only 77.4 % of the Windows gain (L313)

The five Linux lanes came back **ALL PASS** (`L313_RFSAFE_LINUX.md`), and with
them the number that matters most:

```
platform   RF-SAFE vs D   movers   WORSE   stray   feasible
WINDOWS      +0.9040%       12       0       0     100/100
LINUX        +0.6994%       12       0       0     100/100     <- the grader's platform
```

`1.2264069637381392 -> 1.2178289924684162`. **Realisation 77.4 %.** For contrast
`mix` took only a 5 % haircut (L300). Verified independently here: the identity
(`op_wrapper` `62db6ee4…`, cpp `e2c7b2f4…`, ELF `bc991207…`, tar 409,589 B) and
the arithmetic both reproduce.

**🔑 0.774 is NOT a transfer coefficient. It is one case.** The per-case
decomposition (reproduced here from `l313_linux_ctrl.json` /
`l313_linux_rfsafe.json`) kills both hypotheses that were offered for it — "L119
bites hardest on the heavy band" and "concentration":

```
   n  case     win pp     lnx pp       lost  realised
 114    93    +0.2744    +0.0489    +0.2255       18%   <- the entire story
 108    87    +0.1990    +0.1990    +0.0000      100%
 107    86    +0.1426    +0.1426    +0.0000      100%
 120    99    +0.1124    +0.1124    -0.0000      100%
  95    74    +0.0612    +0.0612    +0.0000      100%
  94    73    +0.0410    +0.0410    +0.0000      100%
  79    58    +0.0301    +0.0301    +0.0000      100%
  81    60    +0.0241    +0.0241    +0.0000      100%
  76    55    +0.0165    +0.0372    -0.0207      226%   <- BETTER on Linux
  56    35    +0.0016    +0.0016    +0.0000      100%
  40    19    +0.0009    +0.0009    +0.0000      100%
  38    17    +0.0001    +0.0003    -0.0002      231%   <- BETTER on Linux
     TOTAL    +0.9040    +0.6994    +0.2046      77.4%
```

**n=114 alone is 0.2255 pp = 110 % of the net loss**, and two counts come back
*better* on Linux. The heavy hypothesis is dead — 107, 108 and 120 are all heavy
and all transfer. The concentration hypothesis is dead as a *mechanism* — it
would predict degradation spread in proportion to weight; what happened is one
LP vertex. This is the L153 family exactly (there it was case 96, n=117, worth
107 % of the Win/Linux spread).

n=114 improves on **both** platforms — `1.207200 → 1.137836` on Windows,
`1.207200 → 1.194835` on Linux. The Linux LP simply lands on a worse vertex of
the same degenerate program. **Nothing regresses anywhere on either platform.**

⚠️ One correction to the "10 of 12 realise exactly 100 %" reading: that is
4-decimal display precision, not bit-equality. On raw cost, **6 of 12 movers are
bit-identical across platforms**; the other six differ, but only n=114 differs
materially. Corpus-wide, D diverges on 29 cases and RF-SAFE on 35 — running the
LP on 12 more block counts buys 6 more degenerate programs to disagree about.

**Consequence for the numbers below:** 0.774 must not be applied to s1/s2 as if
it were a coefficient. Doing so assumes those corpora contain a proportionate
n=114-like flip, which is a claim about the corpus, not a property of the
mechanism. The observed base rate for a *material* flip is 1 in 12 ungated counts.

## NET on the grader's platform

Two honest bounds rather than one scaled table. The **low** column is the
measured Linux in-set quality (+0.6994 %, i.e. the n=114 flip happens); the
**high** column is the measured Windows quality (+0.9040 %, i.e. it does not).
Both minus the measured RF bill:

```
 f_eff |        in-set lo/hi |            s1 lo/hi |            s2 lo/hi
  3.17 |   +0.692%   +0.897% |   +0.901%   +1.105% |   +1.015%   +1.220%
  2.84 |   +0.685%   +0.890% |   +0.893%   +1.098% |   +1.008%   +1.212%
  2.38 |   +0.591%   +0.796% |   +0.799%   +1.004% |   +0.914%   +1.118%   <- L308 f low
  2.10 |   +0.433%   +0.637% |   +0.641%   +0.845% |   +0.755%   +0.960%   <- medians shrink again
  1.90 |   +0.274%   +0.479% |   +0.482%   +0.687% |   +0.597%   +0.801%
  1.77 |   +0.102%   +0.307% |   +0.311%   +0.515% |   +0.425%   +0.629%
  1.60 |   -0.158%   +0.046% |   +0.050%   +0.254% |   +0.164%   +0.369%
```

(s1/s2 low columns subtract the same **0.2046 pp absolute** loss the in-set
showed, not a 0.774 ratio — the loss is one case, so an absolute offset is the
honest carry-over. Neither OOS corpus has been run on Linux.)

**Worst case across every corpus and the whole reachable f band (2.38–2.84):
+0.591 %.** The 0.30 pp bar holds on the in-set down to f_eff ≈ 2.10 and on both
OOS samples down to ≈ 1.77.

## Not done

* **OOS on Linux.** s1/s2 were measured on Windows only.
* The upload itself. That is the user's decision.

## To reproduce

The tree was **restored** to `13c629f2…` after staging, because the analysis
tools (`l298_selective_ungate.py`, `l311_rfsafe_robust.py`) read `_L196_LPGATE`
out of the live source — leaving the edit applied silently changes what they
measure. The change is saved as `../l312_rfsafe.patch`.

```
cd ship_final
python l312_build_rfsafe_gate.py          # rebuild the subset from min-of-N dt
git apply l312_rfsafe.patch               # or: patch optimizer_constructive.py < l312_rfsafe.patch
PATH="C:\msys64\ucrt64\bin;$PATH"         # or the compile chain dies silently -> SA
python make_submission.py stage
bash l312_gates.sh && python l312_verdict.py
```
