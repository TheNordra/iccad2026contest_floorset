# L342 — the quality win is real and it is worth 0.06. The violation bill is 3.0.

**Verdict: M52's cliff eats it, and by 30–60×.** L340's B\*-tree SA beats our packer on
`quality` by −0.041 / −0.088 / −0.060 at n = 40 / 80 / 120. Add the `exp(2·V_soft)` factor
— **hard constraints still switched off** — and the same layouts lose by **+2.58 / +2.68 /
+3.80**. Switch the hard constraints on and it is **0/15 feasible, cost 10.0000, at every
n and every iteration count**.

Second finding, and the one that outlives L340: **under the full cost the label is no
longer a uniform ceiling.** It pays for its own violations, and at n = 40 our shipped
packer already beats it (1.1140 vs 1.1331).

No shipping change. Offline oracle probe (2026-08-05 ruling, same standing as L250–L253,
L344). Tool `l342_strictcost.py`, output `l342_out.txt`.

---

## 1. The gate

Our shipped packer's positions are read back out of the shipping run's own results json
(`l313_win48_rfsafe.json`) and re-scored through this file's reconstruction of the
harness. **`max|delta| = 0.000e+00` on all three n**, across `hpwl_gap`, `area_gap`,
`violations_relative` and `cost`. The numbers below are the harness's, not an
approximation of them. (This is the no-op control the ledger keeps insisting on:
`l156`, `m79 control`, `l253`.)

## 2. The three-way decomposition

`[Q] = 1 + 0.5(hpwl_gap + area_gap)` — **this is what L340 reported and nothing more.**
`SOFT` adds `exp(2·vrel)` with `target_positions=None`, so the preplaced/fixed hard
checks are off (l308: that makes the problem ~1.7× easier — which is the point of the
row, it isolates the violation cliff from the preplaced blocker). `STRICT` is the harness.

| n | [Q] ours | [Q] SA@2M | **Δ[Q]** | SOFT ours | SOFT SA@2M | **ΔSOFT** | SA feasible | label cost |
|---|---|---|---|---|---|---|---|---|
| 40 | 1.1140 | 1.0733 | **−0.0408** | 1.1140 | 3.6965 | **+2.5825** | 0/5 | 1.1331 |
| 80 | 1.2383 | 1.1500 | **−0.0883** | 1.4579 | 4.1362 | **+2.6783** | 0/5 | 1.1303 |
| 120 | 1.2136 | 1.1537 | **−0.0599** | 1.2504 | 5.0488 | **+3.7984** | 0/5 | 1.1268 |

**The violation bill is 30–60× the quality prize**, on the same layouts, with the hard
constraints still switched off.

### The exchange rate, worked at n = 80

The SA buys 7.1 % of quality (1.2383 → 1.1500) and pays `vrel` 0.0816 → 0.6327. That is
`exp(2 × 0.5511) = 3.01×` on cost. `0.9287 × 3.008 = 2.794`, and the measured ratio is
`4.1362 / 1.4579 = 2.837`. **7 % of geometry, bought at a 3.0× violation multiplier.**

This is `[[density-is-paid-in-violations]]` again — that entry measured 2.67:1 against
us and 25/25 cases getting worse on `vrel`. L342 is the same trade at a far larger
amplitude, and it reproduces the same split: **boundary dominant, grouping second.**

## 3. Iterations do not buy violations back

| | vrel @10k | @100k | @2M | [Q] @10k → @2M |
|---|---|---|---|---|
| n=40 | 0.6875 | 0.6562 | 0.6250 | 1.3754 → 1.0733 |
| n=80 | 0.8367 | 0.6939 | 0.6327 | 1.5742 → 1.1500 |
| n=120 | 0.8507 | 0.8060 | 0.7463 | 1.6565 → 1.1537 |

**200× the compute moves quality by 0.30–0.50 and `vrel` by 0.08–0.20.** Obviously so —
the objective is `area + HW·wirelength` and contains no violation term at all. But the
magnitude matters: at 2M the SA is still at `vrel` 0.63–0.75 against our 0.00–0.08 and
the label's 0.06. **There is no iteration count on this objective that reaches a
submittable layout.**

### The violation profile is not label-like at all

| | boundary | grouping | MIB |
|---|---|---|---|
| label | 2 / 3 / 4 | **0 / 0 / 0** | 0 / 0 / 0 |
| ours | 0 / 4 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| SA @2M | **16 / 24 / 30** | **7 / 7 / 12** | 0 / 1 / 3 |

The SA produces 8–10× the label's boundary violations and 7–12 grouping violations that
**neither we nor the label have a single one of.**

🔑 **This does NOT convict the B\*-tree representation.** The label *is* a B\*-tree
placement (L325: 100 % on all three invariants) and it sits at `vrel` ≈ 0.06 with zero
grouping violations. The manifold demonstrably contains low-violation solutions. What
L342 convicts is **L340's objective** — which was already dead on runtime. Keep the two
separate.

## 4. Hard constraints: 0/15, and the count

`STRICT` is 10.0000 in every cell. Dimension violations at 2M: **4 / 8 / 14** blocks,
against 6 / 13 / 16 constrained blocks (preplaced + fixed) per case. The SA violates
most of what is constrained, at every iteration count. L340 flagged this as limit #2
("not submittable"); L342 quantifies it. Nothing here is new in kind — but it is now a
measured 0/15, not an expectation.

## 5. 🚨 The finding that outlives L340: the label is not a uniform ceiling

Under `quality` the label is **1.0000 by construction** — the gaps are defined against
it. That is the frame L340, L344 and every "headroom to the label floor" statement in
this project live in. Under the **full** cost the label pays for its own soft violations:

| n | label full cost | ours full cost | label advantage |
|---|---|---|---|
| 40 | 1.1331 | **1.1140** | **−1.7 % (we win)** |
| 80 | 1.1303 | 1.4579 | +22.4 % |
| 120 | 1.1268 | 1.2504 | +9.9 % |

Under quality the label beat us by 10.2 / 19.3 / 17.6 % on these three. Under the full
cost that becomes **−1.7 / +22.4 / +9.9 %**, and it **flips sign on one of three**.

⇒ **"Perfectly reproduce the label" is not a uniformly winning target.** It is the
ceiling of path ③ by construction, and on n = 40 that ceiling is *below what we ship
today*. This is sharper than the clamp argument in `l320-l326` (which says beating the
label earns nothing): here the label is actively **losing**, because it carries boundary
violations we do not. `[[l250-l251-deficit-is-generation]]` measured this corpus-wide
(vrel −3.88 %, we already beat the label on that axis); L342 shows what it does to the
ceiling of an imitation strategy. Across these three cases the label has 9 boundary
violations and we have 5.

**One of three is not a rate.** Do not read "the label is worse than us" as general —
read it as *the label's advantage is not 10–19 %, it is 0–22 % and sometimes negative,
and any ③ business case has to be built on the full-cost column.*

## 6. 🔑 The meta-lesson

**L340's entire headline is measured on a quantity that is not the score.** `[Q]` is one
of three factors in `cost = (1 + 0.5(hg+ag)) · exp(2·V) · rt_adj`. On these layouts the
second factor turns a −0.088 win into a +2.68 loss and the hard checks then turn that
into 10.0. The `[[aggregate-is-not-its-decomposition]]` failure shape, running the other
way: **a component was reported as if it were the aggregate**, and the two factors left
out were each larger than the effect being reported.

The same reading error was live in this session — L344's framing ("the good trees head
for the label, the target is smooth") is *correct on the quality axis* and was explicitly
scoped to it in L344 §5. L342 is that scope being cashed in.

## 7. Where this leaves the lines

* **L340 (① replay-the-generator / B\*-tree SA): closed on a second independent axis.**
  It was closed on runtime (2.8× / 27× / 56× needed). It is now also closed on violations
  — and note the violation closure does **not** depend on runtime: a *free, infinitely
  fast* SA with this objective still scores 3.70 / 4.14 / 5.05 against our 1.11 / 1.46 /
  1.25. Nothing on the iteration axis, the weight axis, or the implementation axis
  touches this.
* **Path ③ (supervised tree prediction): not killed here, but re-priced.** The violation
  catastrophe above is a property of L340's objective, not of the representation, so it
  does not transfer. What does transfer is the *sensitivity*: M52 measured one near-miss
  token → wR 1.232 driven by exactly these boundary/abutment terms, and L342 shows the
  same terms swinging cost by 3× on the same manifold. Plus §5 caps the prize and §4
  quantifies the preplaced blocker at 4–14 blocks per case.
* **What a live version of ③ would have to be:** violation-aware and preplaced-aware in
  the representation itself, not repaired afterwards — `[[chain-saturation-closes-topology-repair]]`
  and L281 both closed after-the-fact repair. `L320_L326_NEW_PATHS.md` Tier 3 ⑦
  (boundary-constrained B\*-tree) and ⑧ (hierarchical B\*-tree) are the named candidates
  and neither has been touched.

## 8. Reproduce

```bash
cd ship_final
"C:/Users/.01/anaconda3/envs/floorset/python.exe" l342_strictcost.py \
    --ns 40,80,120 --seeds 5 --hw 2 --iters 10000,100000,2000000
```

~30 min. Deterministic given seed. The no-op gate is the first thing printed per n; if it
does not say PASS, stop reading.
