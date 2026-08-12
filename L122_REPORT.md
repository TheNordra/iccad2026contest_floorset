# L121 / L122 — the shape step made exact, and priced

Continuation of `ship_final/HANDOFF_2026-08-11.md` §8 step 2: pick Ⓐ/Ⓑ/Ⓒ for
L-4 and continue. Ⓒ was chosen, died with a diagnosis, and the diagnosis opened a
different mechanism that works on quality and fails on price.

**Ship tree untouched.** Everything below is in the screen tree
(`l100_lp_speed.py` knobs, all default-off; `l121_route_c.py`,
`l122_area_tangent.py`, `l122_price.py`). The shipped build is bit-identical:
case 99 LP objective still `1.747467759676` at every gate.

---

## 1. Route Ⓒ — RED, and the reason is not the area constraint

Ⓒ = positions by LP (HPWL objective kept), shapes by L-3's closed form.

**The gradient works.** `fix_dsize` pins the shape columns and drops the area
band, and the reduced costs of those columns are then the true one-sided
derivatives `a = d(obj)/dw`, `b = d(obj)/dh`. Gated: two independent computations
agree to 3.8e-21, and a finite-difference re-solve reproduces `a*eps` to 1.8e-9.

This also settles why the cheap LR attempt kept 0/100. At the linearised
optimum, if the shape column is interior and the area band is tight,
stationarity reads `g_w = h*mu` and `g_h = w*mu`, hence **`g_w * w == g_h * h`**
— which is exactly the closed form's own KKT condition. Feeding the linearised
LP's duals into the closed form returns the shape it already has. The gradient
has to come from a shape-**pinned** solve or there is nothing to read.

**Pinning is infeasible.** With closed-form shapes pinned, 5/5 pilot cases go
LP-infeasible — and still do at a **0.05%** shape change, four orders below the
trust region Ⓒ was trying to escape. So it is not the size of the move.

An elastic phase-1 on the same matrix names the blocking family: **separation
rows** (5–25 of them), not area, not bbox — a 100% bbox slack removes the bbox
from the blame list and the separations remain. Mechanism: a block wedged
between two members of the same rigid unit has a hard width ceiling
(`dw_b + dw_m1 <= gap1 + gap2`), and the fixed topology cannot relax it.

⇒ **Pinning is the wrong shape of answer. The LP has to be free to back off
where the topology says no.** Ⓐ and Ⓑ inherit nothing from this; they were never
started.

## 2. 🚨 Correction to the record: the rho cap is arithmetic, not geometric

The handoff and `[[l120-lr-solver-in-flight]]` both say the area linearisation
"caps the trust region at rho=0.06; rho=0.10 already keeps 0/100 passes",
implying a quality or geometry cause. Measured:

The band rows carry `slack = rho^2 * p` to cover the second-order term `dw*dh`,
so the band on the **linearised** area is `[A(1-TOL)+slack, A(1+TOL)-slack]`,
which is **empty** once `rho^2 >= TOL = 0.008`, i.e. **rho >= 0.0894**. At
rho=0.12 the LP returns status 2 on every case. "0/100 passes" is an infeasible
program, not a rejected geometry.

**And rho really is binding**, so the cap is worth money: 44.7% of shape columns
sit exactly on the trust-region bound, and widening rho 0.06 → 0.088 moves the
LP objective by +0.13% to +0.71% per case.

## 3. L122 — tangent cuts. The quality is real and large

The band is asymmetric: the **lower** row (`w*h >= A`) binds on 259 of 338
reshapeable units; the **upper** on 9. The lower one bounds a *convex* region, so
tangents represent it exactly — no linearisation error, no trust region. The
upper one is the non-convex side and is *itself* the barrier to a large aspect
change: an exact-area widening by `r` has true area `p` but linearised area
`p*(r + 1/r - 1)`, so r=1.5 reads as **+16.7%** against a band of ±0.8%.

So: tangents for the lower side, drop the upper side to `hard_ok` verification
(same solve-then-verify contract `solve_pruned` already runs), plus a price on
each block's own area — because the dropped upper bound has exactly one failure
mode, blocks with no pressure on them running to the box corner at area `A*R^2`
(measured worst errors 44% at R=1.2, 125% at R=1.5, 300% at R=2, i.e. `R^2-1` to
the digit). The price is not a fudge: `{w*h >= A'}` is convex, so a positive
linear cost pushes those blocks onto its boundary where the area is `A'` exactly.

Gated: tangent envelope deficit 0.0567% against the algebraic bound 0.0610%,
worst true area 99.1438% of A against the 1% hard limit, and R→1 freezes every
shape.

**100 cases, official evaluator, `dep_case`'s accept guard verbatim, min-of-3
timing** (single-shot timing cannot rank solvers here — it read 1.95x where
min-of-3 reads 2.33x):

| arm | weighted quality | gain vs anchor | tLP |
|---|---|---|---|
| k=1 (shipped) | 1.236783247 | +2.3559% | ×1.00 |
| k=2 | 1.222834675 | +3.4571% | ×2.01 |
| k=3 | 1.213774500 | +4.1724% | ×3.05 |
| k=4 | 1.207580537 | +4.6614% | ×4.13 |
| k=6 | 1.200358185 | +5.2316% | ×6.00 |
| k=12 | 1.197354156 | +5.4688% | ×8.60 |
| **tangent R=1.5** | **1.202912100** | **+5.0300%** | **×2.33** |

99/100 kept, 0 regressions, 0 infeasible, 0 blocks over the 1% area limit.
**It strictly dominates k=3 through k=12** and captures **92.0% of the depth-12
gain at 27% of the depth-12 time**.

Control arm separates the mechanism from the new price: band + price alone is
worth **+0.11pp** (pilot) / **+0.02pp** (100 cases). The gain is the cuts.

## 4. 🚨 And it prices RED — monotonically, at every range

Priced on `l114`'s model (same `l86` RF floor/gamma, same alpha-calibrated
per-case M, same weights, same `t = ROUTE_A*tnow + tLP`), joint route-A + LP lane:

| arm | s=1 | s=1.5 | s=2 | s=2.5 | grid worst | vs shipped |
|---|---|---|---|---|---|---|
| shipped LP | +2.568% | +2.721% | +2.417% | +2.248% | **+2.248%** | — |
| R=1.1 (×1.32) | +3.395% | +3.537% | +2.297% | +1.990% | +1.987% | **−0.261pp** |
| R=1.2 (×1.54) | +4.117% | +3.500% | +2.344% | +1.974% | +1.974% | **−0.275pp** |
| R=1.3 (×1.83) | +4.409% | +3.226% | +1.992% | +1.456% | +1.456% | **−0.793pp** |
| R=1.5 (×2.33) | +4.620% | +2.531% | +0.667% | −0.072% | −0.072% | **−2.064pp** |

Every arm wins at s=1 — R=1.5 nearly doubles the shipped gain there. All of them
lose at s>=2, and grid worst is the shipped decision rule.

**The frontier does not cross.** The deficit shrinks toward R→1 but never turns
positive, so there is no cheaper setting that ships.

**🔑 The conclusion generalises past this mechanism.** L122 gets 92% of depth-12's
quality at 27% of its time — a **4.4× better trade than depth** — and still
fails. So the shape axis is not priced out because the mechanism was inefficient;
**there is no room to spend LP time at all.** Any future shape-range work has to
be essentially free (≈1.0×) to ship, not merely cheaper than depth. That closes
the efficiency direction that L120 was opened to pursue.

## 5. Honest scope

- Every quality number is on the offline **label-derived baseline** (`c["base"]`
  carries the evaluator's hpwl/area baselines). The shipping path substitutes a
  baseline-free one and the two do not transfer — offline +2.36% became
  +2.1817% deployed. Arm-vs-arm on the same baseline is what this measures.
- The shipped arm's own grid worst moved +1.992% → +2.248% between two runs
  purely from tLP noise on that arm. **Only within-run deltas are meaningful**,
  and all four deltas above are within-run.
- Ⓐ (relax HPWL into the multipliers) and Ⓑ (min-cost flow) remain unmeasured.
  §4 argues they are moot: both are *more* expensive than L122, and L122's
  problem is cost, not quality.
- `hard_ok` still sees none of MIB / boundary / cluster contiguity; the accept
  guard is `dep_case`'s, so a blown area costs a gain, never a wrong answer.
