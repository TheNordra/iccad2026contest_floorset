# L133 — the OOS verdict on DENSITY, and a correction to L132

L130 item 2, taken out of sample. Submission untouched.

**DENSITY=0.40 is a real mechanism and it survives OOS: NET +0.020% against
+0.001% for the shipped 0.80, a 20× separation on 240 cases.** It is still RED —
the bar is **0.30%** and this is 15× short.

And it corrects L132: that report called the in-set DENSITY peak overfitting.
The *magnitude* was inflated 2.2×, but the *direction* was real, and only OOS
could tell those apart.

## 1. What was run

`m77_oos_probe.py` sample **s1** — the same 240 cases as every historical OOS
number (M67-D/F, M72, M75, M76), judged at `--cores 48` because M76 measured a
2.7× dependence on pool shape. The manifest's "s1 is drawn from the ML training
corpus" caveat is about ML models; L129 is label-free and never trained, so s1 is
the right sample and is directly comparable to our own classical arms.

Two prerequisites did not exist and were built here:

* `m77_oos_audit.pkl` was **empty** (`data` len 0) — the 240×35 portfolio cache
  had never been built. 8400 combos, 581s, 0 failed profiles.
* `l133_oos_runner.py` — the L129 placer only knew the in-set 100. It now runs
  the OOS manifest through `m67_oos_probe._load_case`, the SAME loader the
  portfolio's own audit uses, so the candidate sees exactly what the incumbents
  saw.

## 2. The result

| | in-set gate | **OOS NET@48c** | OOS coverage | beats portfolio | proxy mistakes | infeasible |
|---|---|---|---|---|---|---|
| D=0.80 (shipped) | +0.011% | **+0.001%** | 142/240 (59%) | 6 | **1** | 1 |
| **D=0.40** | +0.045% | **+0.020%** | **175/240 (73%)** | **8** | **0** | **3** |

Selection efficiency 99.4% (D=0.40) and 98.2% (D=0.80) — the proxy is still
landing on the oracle, so neither number is limited by arbitration.

### 🚨 Those gate numbers assume the candidate is FREE. It is not.

**Every gate figure above was produced with `--dt 0`, which TELLS the tool the
candidate costs nothing.** The accompanying `dRF@48c = +0.000%, 0 of 175 cases
set a new wall` is therefore a tautology, not a measurement, and an earlier
version of this report drew "runtime is not the constraint and never was" from
it. That was wrong.

Re-scored with the candidate's measured per-case compute (`place` + `lp_polish`,
which is exactly what a deployed form would pay):

| | assume free (`--dt 0`) | **measured per-case cost** |
|---|---|---|
| in-set (D=0.40) | +0.045% | dRF **+38.342%** → NET **−38.297%** |
| OOS (D=0.40) | +0.020% | dRF **+31.701%** → NET **−31.680%** |

The candidate sets a new wall on **76/78** in-set and **155/175** OOS cases. At
48 cores the wall is the max-setter (M67-E, 100/100), so one slow candidate
raises it for the whole case:

    worker_3/layouts_1568/L53   n=117   incumbent wall 1.55s   ML 27.77s
    case 78                     n= 99   incumbent wall 1.36s   ML 35.92s

Candidate runtime against the shipped 0.98s average:

| | mean | median | max | weighted |
|---|---|---|---|---|
| v6 | 3.44s | 1.24s | 34.50s | 12.19s |
| GORDIAN+abut | 3.74s | 1.07s | 28.01s | 13.59s |
| base+abut+D=0.40 | 3.48s | 1.58s | 35.92s | 9.51s |

🔑 **The line does not merely fail to clear the bar — deployed as it stands it
would cost ~30–38% of total score.** 08-15 §4 flagged L129 as unpriced for
exactly this reason ("this form raises the wall first"). L130 priced stage A
(0.06%, correct) and never carried the pricing through to the candidate as a
whole, which is the L125 rule and the one that mattered.

The `+0.020%` figure remains meaningful as an **upper bound on the quality
contribution** — what the candidate would be worth if its runtime were free —
and that is the only way it should ever be quoted. 95.8% of the time is
`lp_polish`; removing the LP takes the solo cost from ~1.76 back to ~2.4, so the
runtime is not obviously separable from the quality.

## 3. Why this is a mechanism and not the in-set artefact

L132's in-set finding was that DENSITY 0.30 and 0.40 covered the identical 77
cases and gated 2.8× apart **entirely on case 67's cost**. The OOS breakdown has
the opposite shape:

| band | cases | D=0.80 | D=0.40 |
|---|---|---|---|
| (20, 60] | 80 | +0.183% | +0.160% |
| **(60, 100]** | 80 | **+0.000%** | **+0.106%** |
| (100, 130] | 80 | +0.000% | +0.000% |

🔑 **D=0.40's entire advantage is a band where D=0.80 wins nothing at all.** That
is exactly the signature the coverage mechanism predicts — the extra cases it
legalises are heavier ones, and heavier cases are where a candidate can beat the
portfolio. It is not one case; it is a band.

Coverage transferred almost exactly, which is the other half of the argument:

| | in-set | OOS |
|---|---|---|
| D=0.40 | 78% | 73% |
| D=0.80 | 64% | 59% |

## 4. 🚨 The correction, and the transfer ratios

L132 §3 concluded "the peak at 0.40 is not a mechanism". **That was wrong in
direction and right in magnitude.**

| | in-set | OOS | transfer |
|---|---|---|---|
| D=0.40 | +0.045% | +0.020% | **44%** |
| D=0.80 | +0.011% | +0.001% | **9%** |

M76's historical transfer figure is ~5%. D=0.80's 9% is consistent with that;
D=0.40's 44% is not, and the difference is precisely that D=0.40's in-set number
was *partly* case 67 and partly a real band effect, while D=0.80's was *entirely*
per-case noise.

🔑 **An in-set gate that is noise-dominated can still have signal inside it.**
L132 was right that the in-set instrument cannot resolve the bar, and wrong to
conclude from that that the peak was empty. The correct reading of a
noise-dominated instrument is "unresolved", not "refuted" — and the only fix is
more cases, not more thinking.

Also measured, and it is a real cost of the knob: D=0.40 produces **3 infeasible
OOS cases against D=0.80's 1**. Lower density buys coverage and pays some of it
back in layouts that do not survive the hard checks.

## 5. Where the line actually stands

    OOS quality contribution (dt=0)   +0.020%    <- upper bound, runtime free
    ship bar                           0.30%     -> 15x short even so
    OOS NET with measured runtime     -31.680%   <- what deploying it costs

Both arms RED, and on the honest number the line is not "close but short", it is
**deeply negative**: the candidate is neither good enough to beat a portfolio at
1.2935 on more than a handful of cases, nor cheap enough to be carried for free
while it tries.

**What is worth keeping from this line, in order:**

1. **The abutment fix (L131)** — +0.0758% on the shipped result, officially
   measured, a correctness fix rather than a search win. Independent of
   everything above, carries **no runtime cost at all** (it is a coordinate
   post-process), and the only item here with a number above a house bar.
2. **DENSITY=0.40** — validated OOS, worth +0.019pp of *quality contribution*
   over the shipped setting, and free as a knob. Pick it on **coverage**, which
   is smooth, monotone and transfers; never on the in-set gate. It does not
   rescue the runtime problem: it is a better setting for a candidate that still
   cannot be afforded.
3. **The GORDIAN alternation (L130)** — best solo quality ever produced by this
   line (−5.1% weighted cost, all three deficit terms) and NOT measured OOS,
   because at DENSITY=0.40 in set it gated at +0.000% and was not worth 240 more
   cases. If the line is ever resumed, that combination is the open question.

**What is closed:** using the in-set gate to choose anything for this candidate.
It has a noise floor at roughly its own bar (L132), and the only instrument that
resolves it costs a 240×35 build plus a 240-case candidate run — about 25 minutes
end to end, now that the cache exists.

## 6. 🚨 The audit cache can be destroyed by the next command

After the first successful build (8400 combos), running `score` left
`m77_oos_audit.pkl` at **33,280 bytes with `data` empty** — `index` kept, exactly
`_cload()`'s sig-mismatch branch (`m77_oos_probe.py:183`), and 10 minutes of build
gone. The symptom is `"no cached profile at cores=48 -> run build for this sample
first"`, which reads exactly like "you never built it".

Two hypotheses were tested and **both are false**: SIG is stable across processes
and PATH (`86b1f6ba`, `_exe_md5 dc47a572`, identical with and without msys), and
`_cload()` does run before mode dispatch, so it is not an ordering bug. The root
cause is **not established** — the only self-consistent explanation is that the
build wrote a different sig, and that could not be reproduced.

Mitigation used, and worth keeping: **build and score in ONE command chain**, and
`cp m77_oos_audit.pkl m77_oos_audit.pkl.bak` immediately after the build. On the
chained run, sig was `86b1f6ba` before and after, and the cache still held 8400
entries after both scores.

## 7. Reproduce

```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u m77_oos_probe.py build --sample s1
```
```bash
L129_EXACT_ABUT=1 L129_DENSITY=0.40 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l133_oos_runner.py --sample s1 --out l129_oos_s1_d040.json
```
```bash
"C:/Users/.01/anaconda3/envs/floorset/python.exe" -u m77_oos_probe.py score l129_oos_s1_d040.json --sample s1 --cores 48 --dt 0
```

Back the cache up straight after the build, and keep the build and the score in
one invocation.
