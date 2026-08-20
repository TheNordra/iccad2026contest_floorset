# L152 — the two structurally-impossible boundary populations: mechanism works, gain is sub-bar

L144/L145 found two boundary populations that are **unsatisfiable in every frame
the placer can build today**, and priced them at +0.05~0.15% after the 1/7
portfolio discount. Both are now implemented and measured out of sample. The
mechanism is real; the gain lands at **+0.23% mean**, under the 0.30% ship bar,
and it costs a C++ change.

## 1. The two mechanisms

`constructive_l152.cpp`, both flags default 0 (off-path gate: **510/510
byte-identical** to the shipping exe over 10 OOS cases x the full 48-core pool).

**R2 — `ICCAD_BND_FRAME_ITEM`.** A compound item holding both an L member (flush
at `ox=0`) and an R member (flush at `ox+bw=it.w`) needs `x=0` and
`x+it.w = bbox_max` simultaneously, i.e. nothing else may extend past its right
edge. `frame_candidates()` never sizes a frame to an item's width, so today that
is unsatisfiable. The flag collects such items' widths/heights before the frame
loop and appends frames of exactly that dimension, keeping the area budget
(`h = base^2*s^2/w`) so the extra candidates are not merely bigger outlines.

**R3 — `ICCAD_BND_SNAP_BEST`.** The `CLUSTER_BND_EXPOSE` snap variant is
all-or-nothing: one overlap discards the whole candidate, so a member whose
offset leaves it inside the item's own bbox never reaches the frame edge. The
flag snaps one member at a time and reverts only the member that overlaps.

## 2. 🚨 The in-set measurement was worthless, and the census says why

First run: three arms (R2, R3, both), **0/100 cases moved, +0.0000%**. That is
not a RED — it is an empty antecedent. The L145 counts were measured on **OOS**:

| corpus | clusters | **opposite-edge-pair clusters** | cases with >=1 |
|---|---|---|---|
| in-set 100 | 359 | **1** | 1 |
| OOS s1 240 | 851 | **23** | 22 |

The in-set 100 has essentially none of the structure the mechanism targets — the
same trap as L144's light-band screen, in a different disguise. **Census the
antecedent on the corpus you intend to measure on, before you measure.**

## 3. Measured where the antecedent exists

Base is the proposed ship config (L137 + L147). Harness `l140_oos_soft_audit.py
--bin constructive_l152.exe`.

| | OOS s1 240 | OOS s2 240 (disjoint) |
|---|---|---|
| cost delta | **+0.2769%** | **+0.1877%** |
| boundary violations | 250 -> **233** | 227 -> **220** |
| cases moved | 9/240 | 14/240 |
| better / worse | 7 / 2 | 11 / 3 |
| infeasible | 0 | 0 |
| jackknife min (drop any one mover) | +0.1263% | +0.0980% |
| wall (single shot) | 1.037x | 1.029x |

Both samples positive, both jackknife-stable, 17 and 7 boundary violations
genuinely removed. But s1's gain is **55% one case** (`worker_1/layouts_112/L107`,
n=115, 1.64493 -> 1.49434), and the mean of the two samples is **+0.23%** against
a 0.30% bar.

Runtime is not the obstacle: the added work is inside the C++ pool, so it
transfers as a ratio, and ~1.03x prices at about −0.005%.

## 4. Verdict: AMBER — do not ship on its own

* it is under the pre-registered bar on the mean and on s2 alone;
* it changes `constructive.cpp`, so it forces a `bin/constructive_linux` rebuild
  plus a Linux re-verify — and the silent new-cpp/old-ELF mismatch is this
  project's most-recorded near-miss (L124, L136, route A);
* L147 ships for +1.17~1.27% with **no** C++ change at all.

Paying the ELF-rebuild risk for +0.23% is the wrong trade while L147 is in
flight. If the teammate ends up rebuilding the ELF for an unrelated reason, this
is the first thing to fold in — the flags are written, gated and measured.

## 5. What it settles regardless

L144 concluded the boundary axis was a *timing* problem ("the compliant slot is
occupied at pack time"). That was incomplete. Two other causes are now
demonstrated and fixable: **frame sizing** (R2) and **all-or-nothing snapping**
(R3). The 24 + 28 violations L145 called structurally impossible were genuinely
reachable — they just are not worth much once the portfolio has arbitrated.

That also puts a measured number on L144 §6.2's 1/7 discount: 24+28 = 52
violations targeted, ~24 removed across the two samples, +0.23% of score.

## 6. Reproduce

```bash
cd /c/ICCAD_ml/ship_final && L144_EXE=constructive_l152.exe "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l144_bnd_trace.py --sample s1 --cases 10 --profiles pool --gate
```
```bash
cd /c/ICCAD_ml/ship_final && ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0 ICCAD_BND_FRAME_ITEM=1 ICCAD_BND_SNAP_BEST=1 "C:/Users/.01/anaconda3/envs/floorset/python.exe" -u l140_oos_soft_audit.py run --sample s1 --cores 48 --bin constructive_l152.exe --out l152_oos_s1_on.json
```
