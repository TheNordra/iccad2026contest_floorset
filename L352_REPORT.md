# L352 — mix's best possible outcome is 0.095 % ahead of rank 1, inside a ±0.5 % error bar

**Verdict: the corrected baseline confirms SHIP_DECISION, and adds a φ-independent reason.**
Even in the *most favourable* runtime world — mix fully at the RF floor on every case — mix
projects to **0.857820** against rank 1's **0.858632**. A margin of **0.095 %**, against
L350's measured cross-corpus error bar of **±0.5 %**. And rank 1's 0.858632 is their *beta*
score; they submit a final too.

Tool `l352_mix.py`. No shipping change. RF-SAFE is already uploaded.

---

## 1. Why re-derive it at all

L348 found `l296_project`'s `DQ_SHIP` encodes **D**'s in-set gain (−5.34 %), not RF-SAFE's
(−6.199 %). SHIP_DECISION's **NET table is arm-relative**, so that bias does not touch it —
but its **rank** statements ("mix needs `f_eff` ≥ 3.40 for rank 1") need an *absolute*
baseline, which is exactly what was stale.

## 2. Model, and its self-check

Per-arm `DQ = 0.931 × (that arm's measured in-set gain over M73 @48c)`:

| arm | in-set @48c | gain vs M73 | DQ | local wall |
|---|---|---|---|---|
| D | 1.226325126 | −5.343 % | −4.974 % | 137.9 s |
| RF-SAFE | 1.215239132 | −6.199 % | −5.771 % | 138.3 s |
| mix | 1.195229398 | −7.743 % | −7.209 % | 159.0 s |

Runtime: `grader_rt(arm,i) = beta_rt_i × SHIP_S × (t_local(arm,i)/t_local(D,i))`, then
`rf_i(φ) = max(0.7, (grader_rt_i /(M_i·φ))^0.3)`. φ sweeps the single unobservable
(machine speed × median drift). All three arms use per-case local runtimes from the same
platform, so their ratios are comparable.

🔑 **Self-check: at φ = 1 the model gives RF-SAFE 0.871211 against L348's independently
computed 0.871174 — 3.7e−5 apart.** The runtime re-modelling reproduces the projection it
has to reproduce.

## 3. The ladder

| φ | RF-SAFE | vs rank 1 | mix | vs rank 1 | mix beats rank 1? |
|---|---|---|---|---|---|
| 0.60 | 0.913288 | +6.365 % | 0.941835 | +9.690 % | no |
| 0.80 | 0.878202 | +2.279 % | 0.889453 | +3.590 % | no |
| **1.00** | 0.871211 | +1.465 % | 0.869569 | +1.274 % | no |
| 1.25 | 0.871114 | +1.454 % | 0.860413 | +0.207 % | no |
| 1.50 | 0.871114 | +1.454 % | 0.858991 | +0.042 % | no |
| 2.00 | 0.871114 | +1.454 % | 0.857875 | −0.088 % | **YES** |
| 3.00 | 0.871114 | +1.454 % | **0.857820** | **−0.095 %** | **YES** |

* mix reaches rank 1 at **φ ≥ 1.641**
* mix overtakes RF-SAFE at **φ ≥ 0.968**
* **RF-SAFE saturates at φ ≥ 1.25** — it is already at the RF floor, so a faster grader
  buys it nothing and a slower one costs it little. That is the robustness SHIP_DECISION
  bought.

## 4. 🔑 The φ-independent argument

The table saturates: by φ = 2.5 mix is at the RF floor on every case and cannot improve
further. **So mix's best possible score, over all runtime worlds, is 0.857820.**

```
mix, best possible          0.857820
rank 1 (their BETA score)   0.858632
margin                      0.095 %
L350 measured error bar     ±0.5 %      <- 5x larger than the margin
```

Two things follow without needing to know φ at all:

1. **The maximum prize is inside the noise.** A 0.095 % margin cannot be distinguished from
   zero by any projection this project can make.
2. **The target is stale.** A21 confirms the final uses the same hidden testcases — but
   rank 1 submits a final too, and 0.858632 is their *beta* score. Beating a superseded
   number by 0.095 % is not a rank.

Meanwhile the downside is live: below φ ≈ 0.97 mix is *worse* than RF-SAFE, and at φ = 0.8
it is worse by 1.3 %.

## 5. What this does to SHIP_DECISION

**It confirms it, by a different route and with a stronger argument.** SHIP_DECISION's case
was "mix trades a rank we hold for one we might not get", resting on `f_eff` landing outside
L308's measured 2.38–2.84 band. This adds a statement that does not depend on where `f_eff`
lands: *even at the best possible `f_eff`, the prize is 5× smaller than the error bar.*

⚠️ φ and SHIP_DECISION's `f_eff` are different parameterisations and their numeric
thresholds are not comparable. What is comparable is the conclusion, and the two agree.

## 6. Reproduce

```bash
cd ship_final
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l352_mix.py
```

Seconds. The φ = 1 row must reproduce L348's 0.871174 for RF-SAFE.
