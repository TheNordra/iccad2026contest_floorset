# L286 — the first OOS measurement of the whole PACKAGE, not of one mechanism. Transfer ≈ 0.

L285 projected rank 2 by assuming the in-set gain since M73/M74 carries to the
hidden set. That assumption was the softest thing in the projection and had never
been tested: every OOS number this project owns is an **arm vs the then-current
base** — one mechanism at a time — and none of them is "the package we ship vs
the package that actually scored".

Two caches make the missing comparison possible with no new placer runs:
`l252_cache.pkl` (the shipped ladder's per-profile OOS layouts) and
`m67_oos_cache_c48.pkl` (the M74-era wrapper's OOS costs), which turn out to
share **all 80 case keys**.

    OOS heavy band, n = 111..120, 40 cases per sample, m67._cost throughout

           pre-LP      post-LP     M74-era     now vs M74    LP moved
      s1  1.511619    1.486674    1.473374       **+0.90 %**    20/40
      s2  1.486799    1.447403    1.456133       **-0.60 %**    20/40

      in-set 48c for reference:  M74 1.293461 -> now 1.226325 = **-5.19 %**

**The in-set gain does not appear on this OOS band at all** — mean ≈ +0.15 %,
and the two samples disagree in sign.

---

## 1. How to read it: a caution, not a forecast

L275's central finding governs exactly this inference. The OOS heavy band is
**22–24 % harder** than the graded corpus, while **beta hidden sits ≈2.4 % from
the in-set**. The hidden set resembles the in-set, not this band. So "the gain
does not transfer to OOS heavy" is *not* the same claim as "the gain does not
transfer to the graded corpus", and reading it that way would repeat the exact
error L275 exists to prevent — in the opposite direction.

What it does do is remove the *comfort* from L285's projection. The honest range
stays **rank 2 – rank 4** with the centre moving from "rank 2" to "rank 2–3".

**The floor is unchanged and is worth stating plainly:** at zero transfer the
package scores what beta scored — rank 4 — and the runtime half is *certainly*
better (L285: 0.855–0.868× the runtime, 98/100 cases on the RF floor). To land
below rank 4 something in the package would have to be actively OOS-harmful,
which is not what this measures.

## 2. Two artefacts had to be cleared first, and both were silent

### 2.1 The apparent 2.5 % regression that was a pre-LP / post-LP mismatch

The first comparison read **1.5116 vs 1.4734 (s1)** and **1.4868 vs 1.4561
(s2)** — the current package apparently 2.1–2.6 % *worse* on OOS. It was an
artefact: `l252_cache`'s records are raw per-profile positions (**pre-LP**) while
`m67_oos_cache_c48`'s costs are the wrapper's full output (**post-LP**).

Confirmed rather than assumed: recomputing the l252 base with `m67._cost`
reproduces **L275's published 1.5116 / 1.4868 to six figures**
(1.511619 / 1.486799), which establishes that L275's "shipped placer" row is the
pre-LP number. And the shape LP is worth **+2.59 %** in-set (L285), which is the
whole of the 2.1–2.6 % discrepancy.

⇒ **There was no regression.** ⚠️ And L275's corpus-difficulty table is a pre-LP
comparison on the current side; that does not change its conclusion (it compares
corpora, not versions) but it should be known.

### 2.2 A double silent no-op that reported "LP moved 0/40"

The first attempt to add the LP back reported it moving **0 of 40** cases, which
is impossible for a mechanism worth +2.59 % in-set. Two independent silences:

1. **`m67_oos_probe` strips `ICCAD_*` at import time** — the ledger's own
   `[[probe-import-time-silent-nooks]]` — so `os.environ["ICCAD_SHAPE_LP"]="1"`
   set *before* the import was wiped, and `_shape_lp_on()` was `False`;
2. **`_shape_lp_maybe` never raises by design** ("the post-processing may
   decline, it may never take the case down with it"), so a wrong flag or a bad
   argument returns the input unchanged and looks exactly like a decision.

Setting the environment **after** the import and asserting
`oc._shape_lp_on()` before the loop turned 0/40 into **20/40**. The only reason
it was caught is the prior "a mechanism worth +2.59 % cannot move zero cases".

🔑 **The rule this yields:** an offline harness must set `ICCAD_*` *after*
importing any probe module, and must assert the flag is live before the
measurement loop — never after, and never by inspecting the output table.

## 3. Why no prior OOS run answered this

Every OOS measurement in the ledger is an **A/B of one mechanism against the
base of its day**: L124 twins, M80 tier, L147, L157, L219/L223, the whole
L25x–L28x arc against `l252_cache`'s shipped ladder. Each validates its own
mechanism. None of them composes into "the cumulative package vs the package
that actually scored", because the base moved under them and, per L271,
whether two mechanisms are additive or substitutive varies by mechanism.

This is the first measurement of the **package**.

## 4. Honest limits — four, and they are not small

1. **Band mismatch.** This is n = 111..120 only. The in-set −5.19 % is over all
   100 cases (n = 21..120). The bands are not the same population.
2. **Version mismatch.** `m67_oos_cache_c48` carries 41 profiles and exe md5
   prefix `dc47a572707c`; the shipped pool is 42. It is "M74-era", and beta
   shipped **M73**, which is a further 0.16 % away at 48 c. It is a reference
   point, not the package that scored.
3. **Sample disagreement.** +0.90 % and −0.60 % on two disjoint 40-case samples
   is a sign flip. With 40 cases the noise is comparable to the effect.
4. **Assembly mismatch.** The current side is `l252` profile outputs + the proxy
   pick + `_shape_lp_maybe` applied by hand; the M74 side is that era's wrapper
   end to end. Route A and the L137 hint are not modelled on either side.

⇒ Treat the headline as "**the in-set gain is not visible on the OOS heavy
band**", not as a point estimate of the transfer coefficient.

## 5. What would settle it

Run the shipped package and an M73 reconstruction through the *same* OOS harness
end to end (`m67_oos_probe --force-cores 48`), on the full 240-case sample rather
than the heavy 40. That is a real run, not a cache join, and it is the only thing
that removes limits 1, 2 and 4 at once. It would not change any shipping
decision — there is no alternative package — so it is worth doing only if
someone wants the rank prediction sharpened.

## 6. Files

```
l286_transfer.py   the cache join + LP re-application, with the liveness assert
```

Nothing was shipped or modified. `constructive.cpp` md5 `e2c7b2f4…`,
`op_wrapper.py` md5 `1c326784…`.
