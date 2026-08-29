# L313 — RF-SAFE: the five Linux lanes, all PASS

Run at the request of the parallel session, which built RF-SAFE but has no WSL
distro. Every claim it made was re-verified here rather than taken on trust.

## 0. Package identity — verified independently

| | claimed | measured | |
|---|---|---|---|
| `op_wrapper.py` md5 | `62db6ee4…` | `62db6ee4569b31ddc8c546ccf3e7cd0b` | ✅ |
| `op_wrapper` == `op_src` | — | identical | ✅ |
| `constructive.cpp` | byte-identical to D | `e2c7b2f4…` **== D** | ✅ |
| `bin/constructive_linux` | byte-identical to D | `bc991207…` **== D** | ✅ |
| tar | 409,589 B, 8 members | 409,589 B, 8 members | ✅ |
| `_L196_LPGATE` | 71 → 83 on | **71 → 83** | ✅ |
| newly ungated | `[38,40,56,76,79,81,94,95,107,108,114,120]` | **exactly that, 12 counts, none lost** | ✅ |

⇒ **no ELF rebuild is involved**, confirmed rather than asserted.

## 1. The Windows 48c headline, reproduced from scratch

    D       1.226325126
    RFSAFE  1.215239132     claimed 1.215239132        exact
    quality      +0.9040%   claimed +0.9040%           exact
    movers 12 (claimed 12)  worse 0 (claimed 0)  stray 0 (claimed 0)  feasible 100/100

## 2. The five lanes

WSL2 Ubuntu, nproc 32, py 3.14.4 / numpy 2.5.2 / **scipy 1.18.0** vs Windows' 1.15.3.

| lane | what it proves | result |
|---|---|---|
| **1a** D @ default cores | the bundled ELF runs on Linux and matches Windows | `1.280025696987226`, **\|d\| 2.2e-16**, 100/100 cost and positions, 0 ULP warns |
| **1b** RF-SAFE @ default cores | **inert below the ≥40-core gate** | **bit-identical to 1a** |
| **2** D @48c, `SHAPE_LP=0` | the Linux pre-LP base | `1.2589744529416786`, **−0.0000 %** vs the Windows LP-off anchor |
| **3** D @48c | the control, and the budget | `1.2264069637381392`; LP ran on **exactly the 71**; **0 regressions ⇒ budget 0** |
| **4** **RF-SAFE @48c, judged** | `judge48()` invariants | **PASS**: `1.2178289924684162`, feasible **100/100**, **0 regressions vs pre-LP at budget 0**, **+0.6994 %** over the control; LP ran on **exactly the 83** |
| **5** t4 on the RF-SAFE tar | corrupt ELF must fall through to g++ | case 50 reproduces the anchor to **0.000e+00** |

Lane 4's LP-liveness line is the Linux counterpart of the in-set gate: **71 → 83**
block counts, read from the package's own stats file.

## 3. The check the lane does NOT do, and it matters

`judge48()` compares against the **pre-LP base**, not against D. What decides
"is RF-SAFE better than what we ship" is the per-case comparison against **D**:

      platform   RFSAFE vs D    movers   WORSE   stray   feasible
      WINDOWS      +0.9040%       12       0       0      100/100
      LINUX        +0.6994%       12       0       0      100/100

**12 movers, 0 worse, 0 stray on both platforms.** `stray = 0` is the structural
statement: on each platform separately, RF-SAFE differs from D *only* on the 12
newly-ungated block counts, which is exactly what the gate table says should happen.

## 4. ⚠️ Linux realises only 77 % of the Windows gain

    Windows   1.226325126 -> 1.215239132   +0.9040 %
    LINUX     1.226406964 -> 1.217828992   +0.6994 %     <- the grader's platform
    Linux realises 77.4 % of the Windows gain

That is a **bigger haircut than `mix` took** (95 %, L300 §6.5), and it should be
carried into the pricing: the peer's NET table is built on the Windows +0.9040 %.
Scaling by 0.774, the NET at each `f_eff` becomes roughly:

    f_eff        3.17     2.38     2.10     1.77     1.60
    NET (win)  +0.897%  +0.796%  +0.637%  +0.307%  +0.046%
    NET (lnx)  ~+0.69%  ~+0.61%  ~+0.49%  ~+0.23%  ~+0.03%

so on the grader's platform it clears the 0.30 % bar down to about `f_eff` 1.9
rather than 1.77. Still positive across the whole reachable range, but less margin
than the Windows table shows.

## 5. The large cross-platform movers are pre-existing, not RF-SAFE's

`judge48` reports 12/100 cases differing from Windows, the largest being case 9 at
1.57e-01. That is **not** RF-SAFE:

      case  n     D(win)     D(lnx)   RFSAFE(win) RFSAFE(lnx)
         9  30   1.204401   1.361360    1.204401    1.361360   <- D differs too
        22  43   1.153272   1.214184    1.153272    1.214184   <- D differs too
        55  76   1.379573   1.379573    1.280644    1.156172   <- RF-SAFE mover
        93 114   1.207200   1.207200    1.137836    1.194835   <- RF-SAFE mover
        17  38   1.314570   1.314570    1.296395    1.272600   <- RF-SAFE mover

Cases 9 and 22 carry the *identical* cost under D and under RF-SAFE on each
platform — they are the known L119 scipy/HiGHS divergence on a degenerate LP, which
D already has and which is why this lane is judged on invariants rather than on
bit-equality.

## 6. Status

Everything the parallel session listed as outstanding for RF-SAFE is now done:
package identity, Windows 48c, five Linux lanes, and (by that session) OOS s1/s2.
Nothing was shipped, staged or modified here — `build_submission.D` and
`build_submission.RFSAFE` are both untouched, and the tree wrapper was not edited.

```
l313_win48.sh          the Windows 48c reproduction
l313_wsl_rfsafe.sh     the five lanes (derived from l300_wsl_mix.sh)
l313_win48_rfsafe.json / l313_linux_{ctrl,rfsafe,lpoff}.json
```


---

## 7. L313b — the 22.6 % is ONE CASE, not a systematic haircut

Both my first explanation ("4 heavy counts, L119 bites hardest") and the peer's
("concentration: 80.6 % of the gain sits on four cases, nothing to average over")
were wrong about the mechanism. Decomposing the gain per block count on both
platforms settles it:

      n     case      win pp     lnx pp    lost pp   realised
      114    93      +0.2744    +0.0489    +0.2255       18%
      108    87      +0.1990    +0.1990    +0.0000      100%
      107    86      +0.1426    +0.1426    +0.0000      100%
      120    99      +0.1124    +0.1124    -0.0000      100%
      95     74      +0.0612    +0.0612    +0.0000      100%
      94     73      +0.0410    +0.0410    +0.0000      100%
      79     58      +0.0301    +0.0301    +0.0000      100%
      81     60      +0.0241    +0.0241    +0.0000      100%
      76     55      +0.0165    +0.0372    -0.0207      226%
      56     35      +0.0016    +0.0016    +0.0000      100%
      40     19      +0.0009    +0.0009    +0.0000      100%
      38     17      +0.0001    +0.0003    -0.0002      231%
      TOTAL          +0.9040    +0.6994    +0.2046      77.4%

🔑 **Ten of the twelve counts realise EXACTLY 100 % — bit-identical on both
platforms. One case, n=114, loses 0.2255 pp, which is 110 % of the entire net loss.
Two counts (n=76, n=38) actually do BETTER on Linux.**

So it is neither "the heavy band degrades" (n=107, 108 and 120 are all heavy and all
realise 100 %) nor "concentration with nothing to average over" (the concentration is
real, but 10 of 12 transfer perfectly). It is **a single case where the degenerate LP
lands on a different vertex**, which is the L153 family exactly: *"case 96 (n=117) is
rejected on Linux and kept on Windows, and that one case is 107 % of the
Windows/Linux spread."* Same phenomenon, different case.

### What that changes about the risk

* n=114 is still **better than D on Linux** (+0.0489 pp), just less better. There is
  no regression anywhere — 12 movers, 0 worse, on both platforms.
* The arm is **not systematically degraded** across the platform boundary; it is
  exposed to *whether a case of this kind exists in the graded set*, which is a
  property of the corpus, not of the mechanism.
* ⇒ the honest framing is **"+0.70 % measured on Linux, +0.90 % on Windows, and the
  0.20 pp difference is one case's LP vertex"** — not a 77 % transfer coefficient to
  be applied to other corpora. Applying 0.774 to the s1/s2 numbers assumes those
  samples contain a proportionate n=114-like case, which is an assumption, not a
  measurement.


## 8. ⚠️ Correction to §7: that was display precision, not bit-equality

The parallel session caught it and it is right. "Ten of twelve realise exactly
100 %" was read off pp contributions rounded to four decimals. On **raw cost**:

      D       bit-identical across platforms:  12/12
      RF-SAFE bit-identical across platforms:   6/12   (17, 35, 55, 86, 93, 99 differ)
      corpus-wide divergence:  D 29 cases, RF-SAFE 35 cases

Their one-line mechanism is exact and the arithmetic closes: running the LP on 12
more block counts buys **6 more degenerate programs to disagree about**, and
35 − 29 = 6.

### But the binary count hides the thing that matters

      n     case         |dcost|    pp impact       tier
      76     55       0.124472564      0.0207%   MATERIAL   (Linux BETTER)
      114    93       0.056998549      0.2255%   MATERIAL   (Linux worse)
      38     17       0.023795260      0.0002%   MATERIAL   (Linux BETTER)
      120    99       0.000003863      0.0000%      small
      56     35       0.000000000      0.0000%   ULP-level
      107    86       0.000000000      0.0000%   ULP-level

Six diverge; **three are material, and two of those three favour Linux.** The net
0.2046 pp is n=114 losing 0.2255 minus n=76 and n=38 giving back 0.0209. Two of the
six (56, 107) differ only below 1e-9 — real bit-inequality, zero score impact.

So the base rate depends on which question is asked:
* *does the LP land on a different vertex?* — **6/12**
* *does that change the score at all?* — **4/12**
* *does it change the score materially?* — **3/12**
* *does it cost us?* — **1/12**

### The ledger line, in the peer's sharpened form

> **A ratio computed from an aggregate is not a coefficient until you have
> decomposed it — and check bit-equality on raw values, not on rounded
> contributions.**

Both halves bit one of us in the same exchange: I turned an aggregate into a 0.774
coefficient, and I then read "identical" off a rounded table.
