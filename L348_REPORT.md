# L348 — the hidden corpus is EASIER than validation on geometry. All of its extra difficulty is violations.

Applying L347's method to the next inference down: `l296_project`'s `DQ_SHIP = −0.0497` is
a transfer *assumption* that every graded projection in this session sits on. Half of it is
measurable with no new runs, and measuring it produces three results:

1. **Corpus difficulty, measured: +1.939 %** — and its decomposition is the finding.
   **On geometry the hidden set is −1.826 % *easier* than validation.** All of the +1.94 %
   and more comes from violations, where hidden `vrel` runs **1.6–1.8×** validation's in
   *every* band.
2. **A scalar `DQ` is defensible** — our gain since M73 is uniform across bands (spread
   2.42 pp), and a band-wise `DQ` moves the projection by **+0.0021 pp**. That assumption
   is now checked rather than trusted.
3. **But `DQ_SHIP` is stale**: it encodes D's in-set gain (−5.34 %), not RF-SAFE's
   (**−6.199 %**, measured here). Corrected, the projection is **0.871174, +1.461 % behind
   rank 1 — not the +2.321 % every number this session quoted.**

Tool `l348_transfer.py`, output `l348_out.txt`. No shipping change, no solver runs.

---

## 1. Gates

Every input must reproduce its own recorded total from its per-case rows before use:

```
beta pkg on HIDDEN (raw)   100 cases  1.320664945  vs recorded 1.320664945   PASS
beta pkg on VAL @48c       100 cases  1.295547821  vs recorded 1.295547821   PASS
RF-SAFE on VAL @48c        100 cases  1.215239132  vs recorded 1.215239132   PASS
```

The first two are **the same code on two corpora, both RF-free** — which is what makes
corpus difficulty a measurement rather than an inference.

## 2. 🔑 Corpus difficulty, measured — and it is entirely violations

```
hidden 1.320664945 / validation 1.295547821  =  +1.939 %
```

| band | weight share | val | hidden | harder | val `vrel` | hidden `vrel` |
|---|---|---|---|---|---|---|
| 21–50 | 0.3 % | 1.3670 | 1.3585 | **−0.62 %** | 0.0322 | 0.0519 |
| 51–80 | 3.3 % | 1.3265 | 1.3273 | +0.06 % | 0.0260 | 0.0377 |
| 81–100 | 15.3 % | 1.3452 | 1.3378 | **−0.55 %** | 0.0324 | 0.0520 |
| **101–120** | **81.1 %** | 1.2847 | 1.3170 | **+2.52 %** | 0.0224 | 0.0409 |

**Geometry only, dropping `exp(2·vrel)`: hidden 1.2112 vs validation 1.2338 = −1.826 %.**

⇒ **The hidden set is easier than validation on geometry and harder only on violations.**
Three of four bands are *easier* overall; the entire penalty is the heavy band, and within
it, the violation factor.

This sharpens two ledger entries rather than contradicting them. L296 measured hidden
`vrel` at 3× the in-set aggregate; here it is 1.6–1.8× band by band, and the aggregate
3× comes from the *mix* (the heavy band carries 81 % of the weight and has the largest
`vrel` gap in absolute terms). And `[[l275-arc-priced-on-wrong-corpus]]`'s "OOS heavy band
is 22–24 % harder" is about **OOS s1**, not the graded set — the graded set is only
+2.5 % harder on its heavy band.

### 🚨 The tension this creates

The one axis on which the hidden corpus punishes us is **exactly the axis L342/L345/L347
just closed by measurement**. Our geometry work transfers *better* than in-set numbers
suggest; the violation deficit is real, is where the whole corpus penalty lives, and is
the thing we have now shown we cannot buy at any price the pool offers.

## 3. Is a scalar `DQ` defensible? Yes — now checked

Our gain since M73, measured per band on validation @48c (in-set total 1.295548 → 1.215239
= **−6.199 %**):

| band | weight | M73 | RF-SAFE | gain | corpus diff |
|---|---|---|---|---|---|
| 21–50 | 0.3 % | 1.3670 | 1.2506 | −8.513 % | −0.62 % |
| 51–80 | 3.3 % | 1.3265 | 1.2445 | −6.182 % | +0.06 % |
| 81–100 | 15.3 % | 1.3452 | 1.2551 | −6.697 % | −0.55 % |
| 101–120 | 81.1 % | 1.2847 | 1.2064 | −6.093 % | +2.52 % |

Gain spread **2.42 pp**, and the band carrying 81 % of the weight is within 0.1 pp of the
weighted mean. Substituting a band-wise `DQ` for the scalar moves the projection by
**+0.0021 pp**. The scalar is fine — that is now a measurement, not a hope.

## 4. 🚨 But the scalar is stale, and it moves the headline by 0.86 pp

* `−5.34 %` is **D**'s in-set gain over M73: `1.226325 / 1.295548 − 1 = −5.343 %`.
* RF-SAFE's in-set gain over M73 is **−6.199 %**.
* The coefficient `DQ_SHIP` encodes is **0.931** — matching its own docstring's "93 %". So
  the constant is internally consistent; it is simply built for the wrong arm.

| | `DQ` | projection | gap to rank 1 |
|---|---|---|---|
| `DQ_SHIP` as-is (= the **D** arm) | −4.970 % | 0.878564 | **+2.321 %** |
| same coefficient, **RF-SAFE** gain | **−5.769 %** | **0.871174** | **+1.461 %** |

**Independent cross-check:** `SHIP_DECISION_2026-08-28` puts D+RF-SAFE at 0.86726–0.86994
by a completely different route (per-arm OOS transfer, not a scalar on the graded corpus).
This lands at **0.87117 — the two methods agree to 0.30 pp.** Two independent estimates of
the same quantity is worth more than either.

⇒ **Every graded projection quoted in L343/L345/L347 (+2.321 % behind rank 1) is the D
arm.** For what we would actually ship it is **+1.461 %**.

## 5. The violation prize, recounted once more

Against the corrected gap, with L347's disambiguated `N_soft`:

> **8 violations** close the +1.461 % gap.
> (L343 said 2 against +2.32 %; L347 corrected that to 7; the arm correction makes it 8
> against the smaller, correct gap.)

The number moved three times. Each move came from measuring something the previous version
had assumed — first the `N_soft` law, now the arm the baseline models.

## 6. 🔑 What generalises

L347's lesson was *go and measure the quantity the inference depends on*. Applied twice
now, it has found:

* an inference built on a **lower bound used as a point estimate** (L347), and
* an inference built on a **stale constant for the wrong arm** (here).

Neither was found by re-reading the reasoning. Both were found by locating the *input* and
going to get it. The general form: **when a constant carries a docstring explaining how it
was derived, re-derive it — the docstring records the derivation, not its currency.**

⚠️ What is *not* fixed: the improvement-transfer coefficient (0.931) is still an
assumption. It cannot be measured without running the new code on the hidden set, which we
cannot do. What L348 adds is that the *other* half of `DQ_SHIP` — corpus difficulty — is
now measured, uniform enough for a scalar, and pointed in a direction nobody had checked.

## 7. Reproduce

```bash
cd ship_final
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l348_transfer.py
```

Seconds. Three gates print first; if any says FAIL the rest is void.
