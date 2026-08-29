# L345 — generation gap, on the band we can measure. The band that carries the prize is in no corpus we own.

**Answer, on the measurable band: GENERATION.** The 51-profile pool's own violation floor
sits only **5.9 %** below what we already select; forcing the minimum-violation candidate
costs **+2.14 %**; and the geometry it charges per violation removed is **2.30× the
break-even licence** L343 derived (1 of 12 cases pays).

**But the result cannot vote on the prize.** The graded (= final) heavy band goes down to
`N_soft = 14` and has **6 of 19** violated cases at `N_soft ≤ 33`, where δ\* is **3.9×
larger**. Neither corpus we can run has **a single** heavy case below `N_soft = 43`. This
is an L278 antecedent failure, it is now measured rather than suspected, and it is
**fixable** — the missing corpus is constructible from the 1M training shards by an index
scan with no solver runs.

No shipping change. Offline oracle probe (2026-08-05 ruling). Tool `l345_gapsplit.py`,
output `l345_out.txt`.

---

## 1. The gate — it independently reproduces L250/L251

| | measured | expected |
|---|---|---|
| proxy pick's rank in true cost | **0.06 / 51** | ~0 (M13/M76/M77) |
| selection efficiency | **99.95 %** | ~100 % |
| cost of proxy pick vs oracle | **+0.0124 %** | L250/L251 recorded **+0.0124 %** |
| cases where the proxy pick *is* the oracle | **39 / 40** | L250/L251 recorded **39/40** |

Both headline numbers land on L250/L251's recorded values, on a probe written from
scratch. The pool reconstruction and the scoring path are sound.

## 2. The generation ceiling

Corpus: OOS s1, `n ≥ 101`, **40 cases** (not 80 — `l252_cache.pkl` holds 40 per sample),
all 51 shipped-pool candidates, strict official scoring, weighted `exp(n/12)`.

| | weighted violations |
|---|---|
| the proxy pick (what we ship) | **6.192** |
| the pool's **minimum** over 51 candidates | **5.827** |
| the true-cost oracle | 6.173 |

* a lower-violation candidate exists on **12/40** cases (33.5 % of weight)
* the true-cost oracle has fewer violations on **1/40**
* on **28/40** the proxy pick is *already* the pool minimum

**The whole pool, across 51 profiles, can only reach 5.9 % fewer violations than the one
we pick.** By type: bnd 1.816 → 1.532, grp 0.515 → 0.455, mib 3.861 → 3.841.

## 3. And taking it would cost more than it saves

| | weighted cost |
|---|---|
| proxy pick | 1.511619 |
| forced pool-min-violation pick | 1.543986 — **+2.1412 %** |
| true-cost oracle | 1.511432 — −0.0124 % |

Against L343's licence δ\* = `(1+G)(exp(2/N_soft) − 1)`, on the 12 cases that have a
lower-violation candidate at all:

* **1 of 12 pays for itself** (5.9 % of that weight)
* median `paid / δ*` = **2.30**

⇒ **a generation gap wearing a selection gap's clothes.** It is not that the pool cannot
make fewer violations — it is that the candidates that do charge more geometry than the
trade is worth.

🔑 Worth noting the *size*: 2.30×, not the 30–60× L342 measured for L340's SA. A mechanism
that removed a violation at ~43 % of the pool's current geometry price would break even
here. That is a demanding target but not an absurd one.

## 4. 🚨 The blind spot — and it is the whole prize

`δ*` scales with `1/N_soft`, so the licence depends entirely on which cases you look at.

| corpus | heavy `n ≥ 101` | `N_soft` min / p50 / max | cases with `N_soft ≤ 33` |
|---|---|---|---|
| in-set 100 (validation) | 20 | 43 / 61 / 67 | **0** |
| OOS s1 (`l252_cache`) | 40 | 59 / 68 / 81 | **0** |
| **graded = final hidden** | 19 that carry a violation | **14** / 52 / 65 | **6** |

The graded heavy cases that carry the L296 prize sit at `N_soft` = 14, 18, 26, 28, 29, 33
(n = 111, 116, 105, 119, 109, 117). **Neither runnable corpus contains one instance of
that antecedent.** L278's rule applies verbatim: *a corpus can only vote on a mechanism
whose antecedent it contains.* Section 3's verdict is therefore sound for `N_soft ≥ 59`
and **silent** for `N_soft ≤ 33`.

**What the extrapolation would say, labelled as extrapolation.** δ\* at `N_soft = 18` is
3.9× δ\* at `N_soft = 68`. If `paid` were unchanged, the median ratio would fall from
**2.30 to ≈ 0.6 — profitable.** Two reasons not to believe that yet:

1. `paid` — the geometry the pool charges to shed a violation — is exactly the kind of
   quantity that is not constant across a regime change. Assuming it is, is the failure
   this project has recorded as `[[l275-arc-priced-on-wrong-corpus]]`.
2. L296's own caveat: a reduced fraction `1/14` could be `2/28`, so every recovered
   `N_soft` is a **lower bound** ⇒ every δ\* computed from it is an **upper bound** ⇒ the
   licence is, if anything, overstated.

## 5. The blind spot is fixable, cheaply

`N_soft` = boundary blocks + Σ(MIB−1) + Σ(cluster−1) is computable **from `constraints`
alone** — no packer run, no label. And the training set is large enough to contain the
missing regime: **9000 shards × 112 layouts ≈ 1 M**, spanning n = 24…118 (sampled).

⇒ **Build the missing corpus**: index-scan the shards for `n ≥ 101 ∧ N_soft ≤ 33`, sample
a matched set, run the same 51-candidate audit on it, and re-run §3. That converts the
question from unanswerable to answered. It is a scan plus one audit — no new mechanism, no
training, no `constructive.cpp` change.

(A 3-shard spot check found 0 heavy layouts, so the scan must cover the shard space rather
than a prefix; shards appear to hold one `n` each, so the index is cheap to build and
`m67_oos_probe._index_files` already does exactly this and caches it.)

## 6. What this does to the two probes L343 proposed

* **Probe 1 (this one): done, with a named limit.** Selection is not the gap — it is
  oracle-perfect to within 0.0124 %, reproducing L250/L251. On `N_soft ≥ 59` the pool
  genuinely cannot generate the layout. On the prize band, unmeasured.
* **Probe 2 (`BP_WEIGHT × (exp(2/N_soft) − 1)`) is now *better* motivated, not worse.**
  §3 says re-weighting *selection* cannot help — the candidate does not exist. Probe 2
  changes **generation**, which is precisely the gap this probe located. But it should be
  run on the constructed corpus from §5, not on `l252`, or it will be priced on a band
  where the licence is smallest — the same error `[[l275-arc-priced-on-wrong-corpus]]`
  records.

⚠️ Neither ships by the 08-30 freeze, and neither should be attempted to.

## 7. Reproduce

```bash
cd ship_final
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l345_gapsplit.py --limit 80
```

~10 min (40 cases × 51 candidates, strict scoring). The gate block prints first; if
"selection efficiency" is not ≈99.95 % and "proxy pick IS the oracle" is not 39/40, the
reconstruction is wrong and the rest is void.
