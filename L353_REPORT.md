# L353 — three cheap closures, and the shipped-flag space is now enumerated

Looking for anything researchable after L351 closed Tier 3. Three probes, all negative, and
one of them closes a *space* rather than a point.

No shipping change. Tool `l353_pinchannel.py` plus two inline checks.

---

## 1. The pin channel is not the deficit

`hpwl = hpwl_b2b + hpwl_p2b`, and no analysis in this project had ever split it. The prior
was good: L327 measured the pin channel's positional law as **8.4× sharper** than b2b
(`u_pin` 0.0514 against a 0.4341 baseline), and Q&A A23/A24 confirm the hidden set uses the
same terminal generation.

| | b2b | p2b | total | p2b share |
|---|---|---|---|---|
| label | 429.5 | 38.3 | 467.8 | 8.2 % |
| ours (RF-SAFE) | 530.5 | 50.7 | 581.2 | 8.7 % |
| **excess** | **+101.0 (+23.5 %)** | **+12.4 (+32.3 %)** | +113.4 (+24.2 %) | |

**p2b carries 10.9 % of our hpwl excess** — and only **10.6 %** in the 101–120 band that
holds 81 % of the score weight. Mildly over-represented in *relative* terms (+32.3 % against
b2b's +23.5 %), small in absolute terms.

🔑 **And there is no re-balancing lever anyway.** The packer's step score sums both channels
weighted by each edge's own weight — exactly how `calculate_hpwl_b2b` and
`calculate_hpwl_p2b` sum them. **The relative weighting is already correct by
construction.** (We are worse than the label on p2b in 100/100 cases and on b2b in 99/100 —
uniformly worse, no structure to exploit.)

## 2. The gap baselines are exact — a validation, not a finding

The evaluator prefers the *stored* label metrics over recomputing from `fp_sol`
(`iccad2026_evaluate.py:826-833`), so every `hpwl_gap` and `area_gap` in this project is
measured against those stored values rather than against the label layout itself. Checked:

```
over 100 cases:  hpwl rel diff  min -0.0000 %  p50 +0.0000 %  max +0.0000 %
                 area rel diff  min +0.0000 %  p50 +0.0000 %  max +0.0000 %
cases where stored != recomputed by >0.01 %:  hpwl 0/100   area 0/100
```

Exact on both channels, 100/100. Worth having: it rules out a whole class of silent error
sitting under every gap number the project quotes.

## 3. `ICCAD_WIRE_FOR_ALL` — never tested alone, and worth nothing

`constructive.cpp:1122` gates the wire term on `bp == 0 || WIRE_FOR_ALL`: **for any
candidate carrying a boundary miss, wirelength is not scored at all.** The flag defaults
OFF, is **not** in M80's 512-vector cloud, and appears exactly once in the tree —
`l271_quality.py`, always **coupled to `ICCAD_L268=4`**. M80's lesson read the other way
round: *a flag only ever tested in combination has not been tested.*

Measured alone, and with `WIRE_MULT` at 0.5 and 2.0, against the 512-vector cloud oracle
(L351's Gate-0 machinery): **0 cases beat the cloud**, oracle increment unchanged at
**+0.0005 %**. Closed.

## 4. 🔑 The shipped-flag space is enumerated, not sampled

Rather than close that by one example: `constructive.cpp` exposes **48** `ICCAD_*` env
flags. **20 are in M80's cloud.** The other 28 resolve as:

| category | flags |
|---|---|
| ledger-RED, kept gated off | `BFS_NORM`, `CLUSTER_ORD`, `REFRAME`, `CLUSTER_BND_CORNER`, `CLUSTER_BND_PERMUTE`, `ANCHORED_BND_REPACK`, `HPWL_SAFE_CLUSTER_SLIDE` (M75) |
| recorded bit-identical no-ops upward | `PUSH_PASSES`, `COMPACT_ITERS` |
| ablation switches, not improvements | `NO_COMPACT`, `NO_PUSH`, `NO_REFINE`, `NO_SWAP`, `NO_JUMP`, `NO_BND_PUSH` |
| offline oracle probes, never shippable | `ORDER_FILE`, `ORDER_GLOBAL` |
| debug / instrumentation | `MIB_DBG`, `FRAME_REPORT`, `FORCE_FRAME_IDX` |
| shipped machinery, already live | `HINT_MODE`, `HINT_REFINE`, `MIB_BUCKET`, `REFINE_ITERS`, `CLUSTER_BND_EXPOSE`, `CLUSTER_BND_EDGE_PACK` |
| **never swept** | `WIRE_FOR_ALL` → §3, worth 0 |

The one flag with **zero mentions anywhere in the tree** was `ICCAD_WIRE_ORDER` — and the
answer is written in the source itself at `constructive.cpp:1805`: *"WIRE_ORDER's failure
was wire-first ordering, vBd 390"*. It broke boundary priority, and `WIRE_TIEBREAK` is its
boundary-safe successor, which **is** in the cloud.

⇒ **No unswept structural knob remains in the shipped packer.** That closes the space, not
a point in it.

## 5. State

Combined with L349 §7 (hpwl has no identified open lever), L351 (Tier 3 all closed) and
L342/L345/L347 (violations closed by measurement):

> **There is no identified open mechanism on any axis, and the shipped flag space is
> exhausted. Anything further needs a new mechanism, not a new measurement.**
