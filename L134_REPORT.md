# L134 — lp_polish has no cheap form, and that closes L129 as a candidate

Handoff 08-16 §7.4. Submission untouched.

**Answer: no.** Not because the LP is inefficient, but because its *minimum unit*
already exceeds the budget. One pass costs 4.86s on the cases that matter against
a 48c incumbent wall of 1.0–1.8s, and the candidate has **zero** value without it.

    max NET@48c over every configuration measured = +0.000%
    achieved by doing no LP at all, which also wins 0 of 78 cases

## 1. What each LP pass buys and costs

41 heavy cases (n≥80, 96.3% of the weight), weighted, at DENSITY=0.40 + abut:

| passes | w.cost | cum. time | max case | cases >1.5s |
|---|---|---|---|---|
| **0** | 2.36048 | 0.000 | **0.72s** | **0/24** |
| 1 | 2.12925 | 1.537 | 4.86s | 9/24 |
| 2 | 2.00363 | 3.134 | 10.30s | 15/24 |
| 3 | 1.94992 | 4.465 | 16.26s | 21/24 |
| 4 | 1.89875 | 5.813 | 23.99s | 23/24 |
| 5 | 1.86362 | 7.288 | 28.92s | 23/24 |
| 6 | 1.83352 | 8.641 | 34.01s | 23/24 |

Two things decide everything:

* 🔑 **`place()` alone fits under the wall on every heavy case** — max 0.72s,
  0/24 over 1.5s. The placer is not the problem.
* 🔑 **The value curve is not front-loaded.** −0.231, −0.126, −0.054, −0.051,
  −0.035, −0.030: still earning at pass 6. So "run fewer passes" is not a free
  lunch, it is a proportional trade.

## 2. Every affordable form, scored with REAL runtime

Not `--dt 0`. Full in-set 100, DENSITY=0.40 + abut, coverage 78 throughout:

| config | solo | beats pool | dRF@48c | **NET@48c** |
|---|---|---|---|---|
| **no LP** | 2.364619 | **0/78** | **+0.000%** (0/78 set wall) | **+0.000%** |
| LP_MAXN=25 | 2.364521 | 0/78 | +0.002% (4/78) | −0.002% |
| LP_MAXN=30 | 2.364178 | 0/78 | +0.006% (8/78) | −0.006% |
| LP_MAXN=40 | 2.362955 | 1/78 | +0.016% (18/78) | −0.016% |
| LP_MAXN=60 | 2.358356 | 2/78 | +0.100% (35/78) | −0.098% |
| LP_BUDGET=0.8s | 2.104502 | 1/78 | +5.616% (61/78) | −5.615% |
| full 6 passes | 1.831823 | 3/78 | +38.342% (76/78) | −38.297% |

**Monotone. Every increment of LP is net-negative, and there is no profitable
band at any size.**

* **Without the LP the candidate wins nothing at all** — 0/78. Its entire value
  is the polish, so "drop the LP" trades −38.297% for exactly 0.000%.
* **The time budget does not work, structurally.** An LP pass cannot be
  interrupted, and the first pass is always admitted (elapsed is 0), so a 0.8s
  budget still pays 4.86s on the worst case. 61/78 set a new wall anyway.
* **A size gate does not work either, and the reason is the interesting one:
  the incumbent wall SCALES WITH CASE SIZE.** A small case's wall is small, so
  polishing only small cases still overruns their own walls — LP_MAXN=25 already
  puts 4 cases over.

## 3. The per-pass levers, measured

| lever | effect | verdict |
|---|---|---|
| `prune_B` | **≤1.17×** faster, cost bit-identical (2.35082 at every value) | exact, real, far too small |
| `sep_trim` | cost 1.83352 → 1.80842, time 8.641 → 9.971 | a QUALITY lever, not a speed one |
| area-only LP (L128) | 2.34× per pass | measured earlier, still not enough |

Even stacking the two real speedups optimistically — 1.17 × 2.34 ≈ 2.7× — puts
one pass at 4.86/2.7 ≈ 1.8s, i.e. exactly *at* the wall, for a configuration
worth 2.129 solo that wins ~1 case in 78.

🚨 **Two corrections to earlier claims in this session's reports:**

1. **`PRUNE_B` was never actually tested when L134 first ran.** `l129` strips all
   `ICCAD_*` at import, so `oc.PRUNE_B` is already `None`; the "shipped" arm set
   it to `None` too and the two arms were identical — which is exactly why the
   costs matched to five decimals. The real measurement is the table above, and
   08-15 §4's "the shipped settings are worth 1.37×" does not reproduce here.
2. **`sep_trim` is not null.** It was reported alongside PRUNE_B as a dead end; it
   is not, it just trades the other way.

The `prune_B` figure is from 8 cases at the light end of the heavy bucket
(0.38s/pass there vs 1.54s weighted across all 41), so the transferable number is
the **ratio**, not the absolute.

## 4. What this closes

**L129 as a portfolio candidate is closed on runtime, not on quality.** The
quality story was genuinely good — L130's alternation is −5.1% solo with all
three deficit terms improving, L133 validated DENSITY OOS — and none of it
matters, because:

* the candidate is worth **0** without the LP, and
* **any** amount of LP costs more wall than it earns, at every size band, and
* the cheapest possible pass is within ~1.5× of the wall *before* counting that
  it needs several passes to be worth anything.

What would change the verdict is not a cheaper LP but a different deployment
shape — the candidate is Python competing against a C++ portfolio inside a
max-setter wall. That is a rebuild, not a knob, and it is post-Final work.

**Still standing, and unaffected by any of this:** the L131 abutment fix,
+0.0758% on the shipped result, officially measured, **zero runtime cost**
(it is a coordinate post-process). It remains the only thing this session
produced with a number above a house bar.

## 5. Reproduce

```bash
L129_EXACT_ABUT=1 L129_DENSITY=0.40 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l134_lp_price.py --minn 80
```
```bash
L129_LP=0 L129_EXACT_ABUT=1 L129_DENSITY=0.40 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l129_global_placer.py run --out results_L134_nolp.json
```
Score with REAL runtime — never `--dt 0` — and set `PYTHONIOENCODING=utf-8` or
the probe crashes (cp950) while printing the warning that matters:
```bash
PYTHONIOENCODING=utf-8 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u m77_ml_candidate_probe.py score results_L134_nolp.json --cores 48
```
