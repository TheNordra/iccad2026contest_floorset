# L139 — pool pruning is closed, structurally and out of sample

Follows L138. Submission untouched; nothing here changes shipped behaviour.

**Verdict: RED, twice over.** A deployable fixed drop set buys only **3.03%** of
wall in set — and that same set removes the SELECTED profile on **12 of 22**
held-out cases.

## 1. The deployable version of L138's 12.95% is 3.03%

L138 found that dropping the 10 slowest profiles PER CASE cuts the weighted wall
12.95% with the winner preserved. That is not deployable: the drop set was chosen
from each case's own timings. A shippable version needs ONE fixed set.

Surveying the whole heavy band (41 in-set cases, n≥80, all 51 pool profiles,
winners computed with the shipped selector and every candidate set RE-SELECTED
rather than subtracted, because `hmin` is the pool minimum):

| fixed drop set | weighted wall | vs base |
|---|---|---|
| {} | 1.2632s | — |
| 14 never-winners | 1.2359s | −2.16% |
| **all 33 never-winners** | **1.2249s** | **−3.03%** |

and it **plateaus from size 21 onward** — dropping more never-winners changes
nothing.

## 2. Why it plateaus: the wall's floor is profiles that earn their place

| | count |
|---|---|
| distinct winners across the band | 18 |
| distinct max-setters | 23 |
| **max-setters that ALSO win somewhere** | **9** |
| max-setters that never win anywhere | 14 |

🔑 **Nine profiles both set the wall and win elsewhere.** Dropping the 14 that
never win just promotes those nine, and they cannot be dropped without losing the
cases they win. That is the floor, and it is structural: pruning is exhausted,
not under-tuned. M41/M42 already took the outliers; what remains is a flat
distribution whose top is load-bearing.

## 3. 🚨 And the fixed set does not survive out of sample at all

The risk a fixed drop set carries is precisely "a dropped profile wins on a case
we did not fit on". Measured directly — same harness, 22 held-out OOS cases from
the s1 sample, n≥80 — rather than inferred:

| | cases | distinct winners | **winner removed by the drop set** |
|---|---|---|---|
| in-set (fitted here) | 41 | 18 | **0/41** |
| **OOS s1 (held out)** | 22 | 15 | **12/22** |

The 0/41 is *by construction* — the greedy stopped the moment any winner moved.
On held-out cases the same set deletes the selected profile on **55%** of them.

    in-set winners  [3, 6, 9, 13, 16, 17, 20, 21, 22, 25, 86, 87, 88, 89, 90, 91, 92, 93]
    OOS winners     [0, 1, 2, 3, 5, 6, 7, 11, 14, 16, 18, 19, 20, 25, 27]

🔑 **"Never wins" is a property of the 41 cases, not of the profile.** The pool
has 51 profiles and each band of cases draws a different subset of winners; the
two sets overlap only partly. Fitting a drop set on any sample and shipping it is
fitting the sample.

This is M67-D's precedent made quantitative — it measured the adaptive cuts' OOS
quality tax at +2.825% against an in-set +0.106%, a 27× gap. Here the same shape
reads 0% in set and 55% out of it.

## 4. What this closes

**Pool pruning as a route to cheaper wall is closed.** Both halves fail
independently: the deployable upside is 3.03% (≈0.9% of score at best, and
nothing on cases already at the RF floor), and the set that produces it is unsafe
on held-out data by a wide margin.

The remaining ways to cut the wall do not go through the pool:

* make the max-setting profiles themselves faster — they are load-bearing, so
  this is optimisation, not removal;
* reduce work inside a profile, which is what L137's `ICCAD_HINT_REFINE` does
  (capping the refine loop on hinted runs bought −19% wall on the heavy cases,
  though at a quality cost OOS at cap 2).

## 5. Reproduce

```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l139_fixed_dropset.py survey --minn 80 --out l139_survey.json
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l139_fixed_dropset.py survey --oos s1 --minn 80 --limit 22 --out l139_oos_survey.json
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l139_fixed_dropset.py analyse --in l139_survey.json
```

The survey is the expensive half and the analysis is free, so any further
candidate set can be tested against the existing json without re-timing.
`repeat=1` is exact for winners — selection depends on positions, not timing —
and only the dt ranking carries noise.
