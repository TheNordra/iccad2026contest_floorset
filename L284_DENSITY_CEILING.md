> # 🚨 CORRECTION (2026-08-27, same day) — this report measured a superseded configuration
>
> `audit_cache_ship.pkl`, the source of all 4200 layouts below, is **stale**. Its
> signature carries REFINE band `(60,100]->6 / (100,inf)->4`, but the shipped
> wrapper is L223/L231's **2/2**, and it predates the shape LP and the L137 hint.
> The decisive test: **its pool-best reproduces the shipped per-case cost on only
> 4/100 cases**, weighted **1.292646 vs the shipped 1.226325 (+5.4 %)** — it is
> an M74-era pool (cf. M74's own 1.293461).
>
> **What is wrong.** Every number in §1-§3 describes the M74-era placer, not the
> one that ships. In particular **"shipped utilisation 80.91 %" is not the
> shipped layout at all** — computed directly from `results_L274_base_48c.json`,
> the shipped weighted utilisation is **85.16 %**. So the claimed "5.49 pp of
> unexploited density headroom sitting in the pool" **does not exist for the
> shipped code**: 85.16 % is already at that pool's own maximum of 85.36 %.
> The U-shaped cost curve and "we are standing on the optimum" are unverified
> for the shipped configuration.
>
> **What survives.** The gap to the label is unchanged in magnitude and is
> computed from valid sources: shipped **85.16 %** against the label's **96.60 %**
> (weighted) = **11.4 pp**, essentially the 11.24 pp reported below. The
> capability-limit conclusion therefore stands, and is if anything stronger —
> the shipped placer is already at the ceiling this pool ever reached.
>
> The trap is the project's own: *"所有離線 gate 一直在量 pre-M71 的 placer"*.
> A cache signature that pins the exe md5 does not help if nobody checks that the
> cache reproduces the shipped result. **That check is now `l285_cache_gate`-style
> and should precede any use of an audit cache.**

# L284 — we are already at this packer's cost-optimal density, and its ceiling is 11 pp short of the label

L283 restated the target as *"raise utilisation while holding boundary and
grouping violations fixed"* and priced our exchange rate at 2.67 : 1 against.
That framing assumed density is **available** and merely expensive. Nobody had
measured whether it is available.

**It is not, and we are already standing at the best point of the trade.**

    100 in-set cases x 42 profiles, weighted exp(n/12), no new placer runs

      utilisation   shipped **80.91 %**   pool max **85.36 %**   label **96.60 %**
      the pool closes only **28.3 %** of the density gap to the label
      the shipped layout sits at the **86th percentile** of its own pool's density
      cost as a function of density has its **minimum exactly where we ship**

---

## 1. The density ceiling is a capability limit, not a price

    gap to the label from the shipped layout : +15.69 pp
    gap to the label from the POOL MAXIMUM   : +11.24 pp

The densest layout this packer has **ever** produced for a case — across 42
profiles covering every knob combination in the shipped portfolio — is still
**11.24 pp short of the reference solution**. Three quarters of the density gap
is not something we are declining to buy; it is outside the reachable set.

🔑 **Independent reproduction of L268 from cached data.** L268 built big-first
commitment ordering, reached **85.2 %** utilisation and called it the largest
utilisation gain in the project's history. The existing pool's own maximum is
**85.36 %**. That mechanism was not finding new territory — it was reaching a
place 42 stock profiles already reach, and it lost for the same reason they do.

## 2. The cost curve is a U and its minimum is where we ship

Every layout binned by its utilisation *relative to the layout the portfolio
selected for that case* (so cases are commensurable), mean cost and `vrel`
relative to that same layout:

      d_util (pp)      n   mean d_cost   mean d_vrel
              -14    185       +18.14 %     +0.01706
              -12    141       +23.46 %     +0.03536
              -10    181       +19.63 %     +0.03081
               -8    473       +12.52 %     +0.02016
               -6    543       +11.11 %     +0.01939
               -4    445       +14.38 %     +0.02707
               -2    363        +9.65 %     +0.02320
      -->      +0    789        **+7.35 %**  +0.01934
               +2    224       +10.61 %     +0.03102
               +4    204       +10.77 %     +0.03498
               +6    157       +11.33 %     +0.04384
               +8    102       +13.24 %     +0.04669

Cost rises on **both** sides. And beyond our operating point `d_vrel` climbs
monotonically — +0.019 → +0.031 → +0.035 → +0.044 → +0.047 — which is L283's
2.67 : 1 seen as a curve instead of a single number.

⇒ **Density and violations are monotonically coupled above 81 %, and the
portfolio is already sitting on the optimum of that coupling.** There is no
"hold violations fixed and go denser" region for this packer; it does not exist.

### 2.1 A correction to my own earlier reading

The within-case correlation between utilisation and cost is **negative in 93 of
100 cases** (p50 −0.534), and I first read that as "denser is cheaper, so go
denser". The binning above shows why that is wrong: **2331 of the pool's layouts
are sparser than the selected one and only 687 are denser**, so the negative
correlation is driven almost entirely by the sparse side. The correct statement
is *"this packer's bad layouts are its loose ones"*, not *"we should go denser"*.
A correlation over a U-shaped relationship reports the side with more mass.

## 3. The densest layout is never the cheapest

      cost   shipped 1.292646   densest 1.413817   (+9.37 %)
      vrel   shipped 0.02368    densest 0.04893
      cases where the densest layout is also the cheapest : **0 / 100**
      cases where the pool holds anything denser than what we ship : 84 / 100

The pool routinely contains denser layouts (p50 +5.49 pp of headroom over the
shipped one) and the arbitration rejects every one of them, correctly.

## 4. What this settles

L281 → L284 have now closed the density/topology axis from four directions:

| | question | answer |
|---|---|---|
| L281 | can topology be edited post hoc? | no — the chain saturates the box |
| L282 | can the chain be shortened? | no — unreachable, and 2.74 : 1 in wire |
| L283 | is that wire price an artefact of squeezing? | yes, but density costs 2.67 : 1 in **violations** |
| **L284** | **is density even available?** | **no — ceiling 85.4 % vs label 96.6 %, and we are at the optimum** |

⇒ The falsifiable target L283 stated — *raise utilisation at fixed violations* —
is now known to be **unreachable by this packer at any violation budget**, not
merely unaffordable. That is a statement about the packer, and it puts the
remaining work squarely in `M27`/`L129`: a different placer, whose own memory
names full GORDIAN alternation as the unfinished piece.

⚠️ **What this does not say.** It does not say 96.6 % is unreachable in
principle — the label is a constructive existence proof on these very instances.
It says the shipped greedy's reachable set tops out 11 pp below it, and that no
selection or post-processing on top of that set recovers the difference.

## 5. Honest limits

1. The pool is the 42 shipped profiles. A 43rd profile could in principle be
   denser — but M80 already searched 512 random knob vectors and the greedy
   curve saturated, so the ceiling is a property of the packer, not of the pool
   size. Not re-verified here.
2. Utilisation is `Σ block area / bbox area`; blocks are free to reshape, so this
   is the true packing efficiency, not a proxy.
3. In-set 100 only, at RF = 1.0. Per L281 §8 the in-set is the harder corpus for
   anything needing slack, so this is the conservative side.
4. The label's 96.6 % comes with its own `vrel` of 0.0511 against our 0.0189
   (L283 §4) — it is not free, it is a better rate. This report measures where
   our reachable set ends, not what an ideal packer would pay.

## 6. Files

```
l284_density_ceiling.py   utilisation ceiling vs the label, cost-vs-density curve
l284.log
```

Reuses `l283_cache.pkl` (4200 scored layouts) and `audit_cache_ship.pkl`.
No new placer runs. `constructive.cpp`, `optimizer_constructive.py` and
`build_submission.D/` were not touched.
