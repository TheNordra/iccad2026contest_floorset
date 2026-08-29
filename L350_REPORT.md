# L350 — the bracket resolves: we are behind rank 1. The aggregate transports; the decomposition does not.

**Verdict: construction (i) is right. RF-SAFE projects to ≈0.871, +1.46 % behind rank 1.**
The optimistic construction (ii) that appeared to put us *ahead* relied on transporting the
score **component by component** between corpora — and that is exactly the operation the
data says is unreliable.

Measured over six mechanisms, each run OFF and ON on two corpora:

| what is transported | mean drift | median | **max \|drift\|** |
|---|---|---|---|
| **`q` (the aggregate quality factor)** | −0.022 % | −0.114 % | **0.490 %** |
| `hpwl_gap` alone | +1.128 % | −0.078 % | **8.017 %** |
| `area_gap` alone | −0.600 % | −0.922 % | **11.341 %** |
| `vrel` alone | −0.401 % | −0.188 % | 1.544 % |

**The aggregate is ~20× more portable than its components.** Construction (i) transports
the aggregate; construction (ii) transports the components. That settles it.

Tool `l350_transport.py`, output `l350_out.txt`. No shipping change.

---

## 1. What the bracket actually reduced to

L349 left two constructions 2 pp apart, straddling rank 1. Before testing anything I
worked out where the 2 pp comes from, and **it is not the transfer coefficient** (0.931 vs
1.0 is worth only ~0.43 pp). It is that the two apply the improvement at different levels:

* **scalar `DQ`** multiplies each case's whole `q = (1+G)·exp(2v)` by `(1+DQ)`
* **transport** shrinks `h`, `a`, `v` *individually* by their measured factors

These differ whenever the corpus's component **mix** differs from validation's — and it
does: the hidden corpus carries `vrel` 0.0425 against validation's 0.0241 (**+76.5 %**),
while RF-SAFE's single largest relative win is precisely violations (**−41.9 %**). So
component-wise transport predicts a *larger* total improvement there (−7.6 %) than the
scalar allows (−6.2 %). That is a genuine mix effect, not an arithmetic slip — which is
why the disagreement needed a measurement rather than an argument.

⇒ The bracket reduces to one question: **does a mechanism's per-component relative effect
survive a corpus change?**

## 2. The test

Six mechanisms were measured OFF and ON on **both** OOS samples. For each mechanism and
each component, compare its relative effect on s1 against the same effect on s2:

```
eff_s1 = x_s1(on)/x_s1(off)      eff_s2 = x_s2(on)/x_s2(off)      drift = eff_s2/eff_s1 - 1
```

Transport assumes the corpus ratio is arm-independent, which is the same as drift = 0.

| mechanism | hpwl | area | vrel | **q** |
|---|---|---|---|---|
| l151 lp-gate | −0.20 % | +1.92 % | +0.45 % | **+0.10 %** |
| l186 twins | −1.13 % | **+9.32 %** | +0.16 % | **+0.49 %** |
| l192 thin-pool | **+8.02 %** | **−11.34 %** | −1.54 % | **−0.20 %** |
| l213 refine-k8 | −0.09 % | −1.61 % | −0.38 % | **−0.20 %** |
| l223 r2/k8r2 | +0.23 % | −1.66 % | −1.09 % | **−0.30 %** |
| l243 devex | −0.07 % | −0.23 % | 0.00 % | **−0.03 %** |

**Every `q` drift is inside ±0.5 %. Component drifts reach 8–11 %.** A mechanism's *total*
effect is portable between corpora; *which axis it lands on* is not.

## 3. Why this is decisive

Construction (i) needs only the aggregate to transport — it multiplies `q`. Measured
stability: **±0.5 %**.
Construction (ii) needs each component to transport independently. Measured stability:
**±8–11 %** on the two geometry channels.

So (ii) is built on the less reliable of the two operations, and its 2 pp advantage came
specifically from re-mixing the components. **Construction (i) stands.**

Three independent routes now agree we are behind:

| route | projection | vs rank 1 |
|---|---|---|
| `SHIP_DECISION` (per-arm OOS transfer) | 0.86726–0.86994 | +1.00 – 1.32 % |
| L348 (graded corpus × corrected scalar `DQ`) | 0.871177 | +1.461 % |
| L350 error bar on that (±0.5 % on `q`) | ≈0.867–0.875 | **+1.0 – 1.9 %** |

**Even the optimistic end of the corrected estimate leaves us behind rank 1 (0.858632).**

## 4. 🔑 The lesson, for the third time this session

* L342: `[Q]` was reported as if it were the score; two omitted factors were each larger
  than the effect.
* L347: `vrel` was exact, and every number derived from splitting it into count × per-unit
  was wrong.
* **L350: the aggregate `q` transports between corpora to ±0.5 %; its decomposition
  transports to ±11 %.**

`[[aggregate-is-not-its-decomposition]]` has now appeared three times in one session, in
three different guises. The operational form worth keeping: **when you must move a
measurement to a new corpus, move the coarsest quantity that answers your question.** Every
extra level of decomposition you transport multiplies the error you carry with it.

## 5. Honest limit

s1 and s2 differ in difficulty by only ~0.13 %, while validation→hidden is +1.94 %. So
this bounds **arm-dependence** well and does **not** extrapolate the drift to the larger
shift. What it establishes without extrapolation is the *relative* reliability of the two
operations on identical data — and that ordering is what chooses between the constructions.

## 6. Reproduce

```bash
cd ship_final
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l350_transport.py
```

Seconds.
