# L351 — evaluating the remaining Tier-3 items against this session's measurements

**Verdict: ⑪ (evolve the packer's priority/tie-break function) is the only remaining
Tier-3 item that survives, and it survives for one specific reason — it is the only one
that operates on the packer's *reachable set* rather than on its inputs, and the reachable
set is precisely what CLAUDE.md's strategic conclusion 7 identifies as the bottleneck.**

**Correction first: ⑦ and ⑧, which I nominated at the end of L342 and again in L349, are
dead.** I nominated them before L345/L347 existed. This session's own results kill them.

---

## 1. ⑦ and ⑧ are dead — two independent kills each

**⑦ boundary-constrained B\*-tree** (exact four-sided feasibility) and **⑧ hierarchical
B\*-tree** (grouping/MIB as representation invariants) are both B\*-tree representation
refinements aimed at the violation and preplaced blockers.

1. **They fix the wrong blocker.** L340's runtime wall is **27× / 56×** at n = 80 / 120 and
   is untouched by either. A perfect boundary-constrained hierarchical B\*-tree still has
   to find its tree, and still cannot find it in 1.4 s. L342 sharpened this: the violation
   closure **does not depend on runtime** — a *free, infinitely fast* SA on that objective
   still scores 3.70 / 4.14 / 5.05 against our 1.11 / 1.46 / 1.25.
2. **The blocker they fix is already closed.** ⑦ targets boundary violations, ⑧ targets
   grouping and MIB. L345/L347 closed the violation axis by measurement: selection is
   oracle-perfect (+0.0124 %), the pool's violation floor is 5.9 % below what we pick,
   `paid/δ*` = 2.30, and the favourable `N_soft` band that would flip it **does not exist**
   (0 of 201 264 heavy training layouts). MIB in-set is already 0 and L124 proved 307 is
   the *locked-aware lower bound*, i.e. provably optimal.

## 2. ⑨ and ⑩ are also closed

* **⑨ window matheuristic** (exact MILP on 10–16-block windows) is after-the-fact topology
  repair, and that whole family is closed by `[[chain-saturation-closes-topology-repair]]`:
  **62/100 graded cases have critical-chain slack of exactly 0**, with a p50 34.3 % of
  blocks on the chain. M64, L256–L262 and L281 all died of this. A window MILP has no slack
  to move into unless it is allowed to grow the bbox — which is buying area.
* **⑩ anchored SMACOF** produces a position seed. That is the advisor layer, and M68
  measured a *perfect* position seed at **+0.001 %** against the 41-profile portfolio.

## 3. Why ⑪ survives

| filter | ⑪ |
|---|---|
| **Operates on the bottleneck** | CLAUDE.md conclusion 7 is precise: perfect order +0.005 % (M26), perfect seed +0.001 % (M68), perfect shape +0.099 % (M79) against 14.343 % of headroom ⇒ *"瓶頸不在餵給 packer 的決策，在 packer 的可達集合"*. **⑪ is the only Tier-3 item that changes the reachable set rather than an input.** |
| **The family has demonstrated payoff** | **M71** — the largest recent quality win (in-set −1.589 %, OOS −4.04 %, and it shipped) — *was* a change to `make_group_item`'s candidate set and ordering key. That is a priority/tie-break function change, found by hand. |
| **Error tolerance** | Structurally perfect. Unlike every ML path the ledger killed (M52: one near-miss token → wR 1.232, a cliff), an evolved tie-break either scores better or it does not. No fragile decode, no zero-tolerance band. |
| **Not bounded by the three oracles** | The source document's own annotation, and it is right: all three bound *inputs*. |
| **L284's density ceiling does not bind it** | L284 states the 85.4 % ceiling is *"a property of our pool's reachable set, not of the instance"*. Changing the priority function changes that set. |

## 4. The honest counter-prior — and it is substantial

* **M80 already searched the coefficient space and it is saturating.** R = 512 random joint
  sampling gives a per-case oracle of **+3.081 %**, but the deployable held-out figure is
  **+0.655 % NET**, and going R 128 → 256 → 512 made held-out **worse** (fold-greedy
  overfitting). ⑪'s bet is that new *functional forms* reach beyond the *coefficient*
  space. **That is unverified and is the whole risk.**
* **M78's counter-prior applies directly**: *"the shipped candidate set is not impoverished,
  it is tuned"* — the identical crossing mechanism was −0.18 % in one call site and +0.36 %
  in another. Changing the priority function is the same family of intervention, and its
  default sign is not positive.
* **Every perfect-information ceiling this project has measured came back ≤ 0.1 %**, four
  times running. The prior on any new axis is poor.
* **Cost.** Each candidate needs a recompile plus an evaluation: single-profile 100-case
  ≈ 16–25 s, portfolio-level minutes. A meaningful evolution run is 10³–10⁴ evaluations ⇒
  hours to days. Any winner then needs `bin/constructive_linux` rebuilt, the five Linux
  lanes re-run, and re-staging.

## 5. 🚦 Pre-registered Gate 0 — run this before building any evolution machinery

Following M79's own pattern (measure the ceiling before building the loop). The question
that decides ⑪ is **narrow and cheap**:

> **Does the functional *form* reach beyond the coefficient space M80 already swept?**

Procedure, reusing what is on disk:

1. `m79_knob_cloud.pkl` already holds **512 vectors × 100 cases**. Its per-case oracle is
   the coefficient-space ceiling: **+3.081 %**.
2. Write **5–10 hand-authored new *terms*** into an isolated `constructive_l351.cpp`
   (default OFF ⇒ off-path bit-identical, so no cache is invalidated) — e.g. a step score
   normalised by remaining free area, a one-ply connectivity lookahead, a boundary-slack
   tie-break. These are *forms*, not new values of existing knobs.
3. Add each as one profile and measure the per-case oracle **increment over the 512-vector
   cloud**, using `m79_knob_cloud_probe.py`'s existing greedy/LOO machinery.

**Pre-registered decision rule:**

| increment over the cloud oracle | verdict |
|---|---|
| **≥ +0.5 %** | functional form reaches beyond coefficients ⇒ build the evolution loop |
| +0.1 – 0.5 % | ambiguous ⇒ measure held-out transfer before spending more |
| **≤ +0.1 %** | the space is coefficient-saturated ⇒ ⑪ collapses into M80, which is done ⇒ **close it** |

Ship bar if it ever gets that far is the project's standing one: **OOS NET ≥ 0.30 %**, with
M80's discipline (K chosen on OOS, held-out elbow, cores-gated deployment).

🔑 **The gate must be an oracle over the *cloud plus the new forms*, not the new forms
alone.** M79's first version reported a false +0.000 % by ranking on mean solo cost instead
of portfolio delta; and M78 showed the same mechanism flips sign between call sites. Judge
on portfolio delta, in the 48-core pool shape, or the answer is not about the mechanism.

## 6. Expected value, stated plainly

⑪ is the **best remaining candidate and still probably negative.** It earns the Gate-0
probe because it is the only item aimed at the measured bottleneck and because the one
mechanism in its family that was ever tried (M71) was the largest recent win. It does not
earn the evolution loop until Gate 0 clears +0.5 %.

⚠️ Nothing here ships by the 08-30 freeze, and nothing here should be attempted before it.

---

# L351 Gate 0 — RESULT: **+0.0005 %. Item ⑪ is closed.**

Run 2026-08-28. Tools `constructive_l351.cpp/.exe`, `l351_gate0.py`; outputs
`l351_cloud_regen.txt`, `l351_gate0_cache.pkl`.

## G0. Off-path bit-identity — PASS, after a control caught a false failure

`constructive_l351.exe` with no L351 flag is **bit-identical to the shipped
`constructive.exe` on 8/8** sampled jobs. The four form flags are inert.

⚠️ The first version of this gate compared against the **cached** cloud positions and
reported **6/10 mismatches**. The control — the same jobs through the *shipped* exe —
mismatched **the same 4/8**. The flags were inert all along; **the cache was stale.**

## G0.5 🚨 The M80 knob cloud was pre-L124, and its headline has inverted

```
cache sig      : aa2eac1e3701674e9793002855a78733   (2026-08-11)
current _sig() : 16488368e696a9c50bfe9ac8d940836e
```

`_sig()` pins the exe md5 exactly to catch this; reading `data` directly bypassed it.
L124, L131, L136 and L137 all changed `constructive.cpp` after the cloud was built.
Regenerated: **51 200 runs / 1621 s**, feasible 100/100, 83/512 distinct winners.

| | oracle total | vs the shipped portfolio of its day |
|---|---|---|
| M80's published cloud oracle (pre-L124) | 1.253613518 | **+3.081 % better** than M74's 1.293461 |
| **current-binary cloud oracle** | **1.242558394** | **2.25 % WORSE** than RF-SAFE's 1.215239 |

**The knob-cloud axis has not merely saturated — it has inverted.** Since M80 the shipped
portfolio gained 6.2 % while the cloud's per-case oracle gained 0.87 %. *"M80's per-case
oracle is +3.081 %"* is now a stale claim and should not be quoted.

## G0 result

12 form profiles (4 forms × 3 magnitudes) × 100 cases, added to the 512-vector cloud:

```
per-case oracle, 512-vector cloud                1.242558394
per-case oracle, cloud + 12 form profiles        1.242552243
*** ORACLE INCREMENT FROM THE FORMS: +0.0005 % ***

cases where a form beats the ENTIRE cloud: 1/100
   n=49   wirenorm_4.0     +2.115 %
```

**One case out of a hundred, and it is n = 49** — weight `exp(49/12)` ≈ 58 against
n = 120's ≈ 22 026, i.e. essentially zero weight. The pre-registered rule
(≥0.5 build / 0.1–0.5 ambiguous / **≤0.1 close**) fires: **CLOSE ITEM ⑪.**

The verdict is if anything *stronger* than the rule requires: the forms failed to add
anything to a baseline that the shipped portfolio already beats by 2.25 %.

## Honest scope

This tested **4 hand-authored forms at 3 magnitudes**, not the space a FunSearch-style
search would explore. It says *these* forms do not reach beyond the coefficients — not that
no form can. But they were chosen to attack the diagnosed weakness directly (greedy
short-sightedness on hpwl, the one axis L349 found with room), and the most on-target of
them returns +0.0005 %. Same standing as M79's Gate 0, which the project treated as
decisive on the same basis.

## Where this leaves the research line

Every Tier-3 item is now closed: ⑦⑧ (L351 §1), ⑨⑩ (§2), **⑪ (this gate)**, and ⑫ is
M80's own axis, which G0.5 shows has inverted. Combined with L349 §7 (hpwl has no
identified open lever) and L342/L345/L347 (violations closed by measurement), **there is no
identified open mechanism on any axis.** That is the honest state, and it is worth
recording so the next session does not re-derive it.
