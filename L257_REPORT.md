# L257 — the twin deployment works, and is priced out at every K

**Verdict: the twin form removes L256's deployment failure exactly as intended —
quality +0.3309 %, and the proxy reaches the union's oracle perfectly — but the
pool cost curve charges more than that from the FIRST twin.** NET is negative at
K = 1 and gets worse. Not shipped.

This closes the L252→L257 line: the mechanism is real, small, and unaffordable.

---

## 1. Method

Index space: `i` = shipped profile *i* (`l252_cache.pkl`), `1000+i` = its
`ICCAD_L256` twin (`l257_cache.pkl`, 40 cases × 51 profiles, captured once). Any
twin set is then priced **exactly, offline, with no further solving** — the
`m76_escape_probe` pattern. The shipped selector is reconstructed on the union,
including `hmin` (the pool-wide minimum HPWL), which is precisely what the
global-overlay form got wrong.

**Plumbing gate: the no-twin baseline reproduces `1.511619`** — L250's and L253's
number, from a third independent path.

## 2. Result (OOS s1, n ≥ 101, 40 cases, exp(n/12), official strict scorer)

| | weighted true cost | vs base |
|---|---|---|
| no twins | **1.511619** | — |
| ALL 51 twins | **1.506617** | **−0.3309 %** |
| oracle over originals | 1.511432 | |
| oracle over originals ∪ ALL twins | **1.506617** | −0.3186 % ← the ceiling |

🔑 **The deployed value equals the ceiling to all six digits.** The proxy picks
the union's true best every time; there is zero selection loss in the twin form.
That is the whole point of the twin pattern and it worked.

Greedy K-curve (in-sample), saturating at **K = 8**:

    K=1  -0.0958%   K=2  -0.1888%   K=3  -0.2432%   K=4  -0.2693%
    K=5  -0.2952%   K=6  -0.3156%   K=7  -0.3291%   K=8  -0.3309%
    parents: [87, 28, 20, 86, 101, 95, 94, 0]

### 2.1 Why it is only 0.33 % — the bar nobody had stated

    twins that beat THEIR OWN parent      548 / 2040   (27%)
    twins that beat the BEST original      17 / 2040   <- the bar that matters
    cases with at least one such twin         9 / 40

L256's isolated result (154/259 twins beat their parent) says **nothing** about
whether a twin beats the pool champion, and those are wildly different bars.
L256 lifts *mediocre* layouts — the ones that were never going to be selected.
This is consistent with everything measured: L250 says the proxy already picks
the true best 39/40; L252 says the winner's frame is already at `s_min` (selector
slack in only 9/40 cases). **The champion has no slack for a shrink to exploit;
only the losers do.**

## 3. The price — measured here, not inherited

L248's pool curve was measured on *plain* extra profiles, so the twin's own
overhead had to be measured. Same-batch ratios, min-of-3, alternating arm order
(3 cases × 6 parents):

    per-profile   p10 1.042   p50 1.067   p90 1.346   max 1.462
    slowest twin / slowest parent          1.052x

So a twin costs ~5–7 % more than its parent, and barely moves the max-setter —
i.e. **L248's curve applies almost directly**, with an ~5 % surcharge.

L248 measured **K=6 plain profiles = 9.2 % of heavy-band wall**, taking NET from
+5.224 % to +4.194 % while quality contributed +0.364 % — so the wall alone cost
**1.394 pp for 9.2 %**, i.e. **≈0.151 pp of NET per 1 % of heavy-band wall.**

| K | quality | est. extra wall | est. wall cost | **est. NET** |
|---|---|---|---|---|
| 1 | +0.096 % | ~1.6 % | −0.24 pp | **−0.15 pp** |
| 3 | +0.243 % | ~4.8 % | −0.73 pp | **−0.49 pp** |
| 8 | +0.331 % | ~12.9 % | −1.95 pp | **−1.62 pp** |

**Negative from the first twin**, because L256's quality-per-profile (0.096 pp
for the best one, declining fast) sits ~2.5× below the pool's cost-per-profile
(~0.24 pp), and the gap widens with K.

⚠️ The wall→NET conversion is L248's, measured on this same band with min-of-3.
What is measured *here* is the twin's own multiplier (1.05–1.07×), which is the
only part L248's curve could not cover. The sign is robust: it would take the
cost-per-profile being **2.5× too pessimistic** to reach break-even at K=1, and
L248's 9.2 % was itself a measurement that overturned a modelled "the first
additions are nearly free" argument in the opposite direction.

## 4. What this closes

    L256 global overlay   NET +0.505%   -- proxy churn, mechanism invisible
    L256 isolated         -0.2506%      -- real, small, 154/105
    L257 twin form        -0.3309%      -- ceiling reached exactly, zero selection loss
    L257 twin NET         -0.15 pp at K=1, worse after   -- priced out

The twin pattern did its job: it converted an unmeasurable, churn-dominated
overlay into a clean, oracle-tight +0.33 %. The mechanism simply is not worth a
profile slot. **This is the same wall the whole "add profiles" family hits
(L248), and L256 does not escape it.**

## 5. Honest limits

* In-sample greedy on s1 heavy band. No s2, no OOS elbow — L124's discipline was
  not run, because the NET is negative before the K choice matters.
* The K-curve and the K=8 saturation are in-sample; the *direction* (saturates
  fast, ~0.1 pp for the best twin) is what the conclusion uses.
* Wall→NET is L248's conversion, not re-measured. §3 states what that rests on.
* 40 cases, heavy band only.

## 6. Files

```
l257_capture.py  the L256-ON pool capture -> l257_cache.pkl (40 x 51)
l257_twin.py     offline exact pricing of any twin set + the diagnostic
l257_wall.py     the twin's own wall multiplier, same-batch min-of-3
l257_twin.pkl    chosen order, base/full/oracle
```
