# L274 — can `l269p1` be turned into a new submission package?

**Technically yes. It was built, and both equivalence gates pass. Do not upload it.**

The mechanism was priced on the OOS heavy band and it is **negative on the full
100-case deployment path**, which is the shape that gets graded.

    in-set 100 cases, official strict scorer, 48 cores, weighted exp(n/12)

      base (shipped)        1.226325
      l269p1                1.226626   +0.0245 %   31 movers (15 better / 16 worse)
      l269p2                1.231530   +0.4244 %   42 movers (16 better / 26 worse)

Both budgets are on the wrong side of zero, and they are ordered the same way the
other session's NET ranked them (p1 better than p2) — the ranking transfers, the
**sign does not**.

🔑 **And p2's damage is concentrated in exactly the band it was measured on.**
9 of the 20 heavy in-set cases move, contributing **+0.3364 %** of the +0.4244 %.
`L267_L269_REPORT.md` measured −0.8143 % (s1) / −0.7285 % (s2) on the heavy band.
**Same band, opposite sign, different corpus.**

---

## 1. What was built, and the gates it passes

`l274_ship_patch.py` takes the *measured* probe source (`constructive_l267.cpp`,
snapshotted to `constructive_ship_src.cpp`) and flips three things:

    L269         0 -> 1      the mechanism becomes a code default
    L269_PROBES  5 -> 1      the p1 budget
    the parser   accept 0    so ICCAD_L269=0 is a working kill switch

Why the measured source rather than a fresh minimal patch: the entire L267–L269
evidence chain attaches to `constructive_l270.exe` run with `ICCAD_L269=1
ICCAD_L269_PROBES=1`. Re-deriving the mechanism would produce a *different*
artefact that merely resembles the one carrying the numbers. Building it this way
makes the equivalence checkable, and `l274_gate.py` checks it:

| gate | | result |
|---|---|---|
| **G1** | `constructive_ship.exe` with **no env** == `constructive_l270.exe` with `ICCAD_L269=1 ICCAD_L269_PROBES=1` | **102/102 PASS** |
| **G2** | `constructive_ship.exe` with `ICCAD_L269=0` == stock `constructive.exe` | **102/102 PASS** |

G1 is what transfers the evidence to a binary that reads no environment — which
matters because per L158 an env-only mechanism is **inert inside the package**.

### 1.1 🚨 G2 caught a missing kill switch that would have shipped

First run: **G2 FAILED 34/102.** The probe's parser is

    if (const char* e=getenv("ICCAD_L269")){ int v=atoi(e); if (v==1||v==2) L269=v; }

A value of **0 is ignored**. That is harmless while the default is 0 (env-only),
and becomes a *missing kill switch* the instant the default is flipped to 1 —
`ICCAD_L269=0` would have left the mechanism on. Every shipped mechanism in this
project has an off switch (`ICCAD_M71=0`, `ICCAD_M80_TIER=0`, `ICCAD_HINT_MODE=0`,
`ICCAD_SHAPE_LP_DEPTH2=0`), and without one there is no way to produce the "off"
control that every future bit-equality gate needs. Fixed to `v>=0&&v<=2`.

## 2. The Linux ELF is a second, independent problem

Changing `constructive.cpp` forces a `bin/constructive_linux` rebuild, and Windows
cannot see whether that succeeded (`_ensure_compiled` skips the bundle on `nt`).

WSL Ubuntu **is** available now (WSL2, unlike the L137-era note that recorded no
distro) — but it carries **g++ 15.2 / glibc 2.43**:

| | max GLIBC symbol required |
|---|---|
| shipped `bin/constructive_linux` | **GLIBC_2.34** |
| fresh build here, shipped flags (`-static-libstdc++ -static-libgcc`) | **GLIBC_2.38** |
| fresh build here, `-static` | **none** (fully static, smoke passes, 1.49 MB) |

GLIBC 2.34 ≈ Ubuntu 22.04; 2.38 ≈ 23.10+. Rebuilding with the *shipped* flags
would raise the floor and very likely make the ELF unloadable on the grader —
silently, falling through to on-site compilation. `-static` fixes it, but that is
a **link-flag change to the shipped artefact that has never run on the grader**,
and it triples the binary size (550 KB → 1.49 MB).

⇒ Even if the quality case were good, shipping would mean changing the binary's
link model one day before the deadline.

## 3. Why the two corpora disagree, and which one to believe

| corpus | vrel | what it says about `l269p1` |
|---|---|---|
| in-set 100 (= the alpha test set) | 0.0140 | **+0.0245 % (worse)** |
| OOS s1 / s2, heavy band only | 0.0967 | −0.5706 % / −0.5507 % (better) |
| **beta hidden (the real grader)** | **0.0425** | unmeasured — and it sits *between* them, nearer in-set |

The project's standing lesson is `[[inset-identity-is-not-oos-identity]]`: in-set
no-movement does not mean a mechanism is worthless OOS. That lesson is real, and it
is **not** what is happening here. This is not "in-set is flat"; it is **in-set is
negative, on 31 and 42 movers respectively, deterministically**.

And the hidden corpus that actually decides the score is closer to in-set than to
OOS on the one axis where they differ most. That is a reason to weight the in-set
signal here, not to dismiss it.

⚠️ **Honest statement of what is NOT established**: the in-set 100 is one corpus
too, and a small negative there does not prove the mechanism is negative on the
hidden set. The claim is only that **the evidence is now contradictory**, and a
contradictory case is not a case for replacing a working submission.

## 4. What is NOT wrong with it

Two things I checked and had to withdraw:

* **"It is slower end-to-end."** The first read was +5.9 % runtime (127.4 s →
  135.0 s). A baseline repeat came back at 132.5 s — a **4.0 % run-to-run spread on
  the baseline alone**. The runtime delta is inside noise; I cannot claim it.
* **"The heavy band does not move in-set."** It does — 4 of 20 for p1, 9 of 20 for
  p2. The first mover table was sorted by weighted contribution and the heavy rows
  simply were not at the top.

Quality *is* deterministic: two baseline runs differ on **0/100** cases, so the
+0.0245 % is exact, not sampling.

## 5. Recommendation

**Keep `build_submission.D`. Do not build or upload a replacement.**

| | |
|---|---|
| shipped now | 48c Linux **1.2264069637381392**, graded 0.87818, **rank 2**, margin 1.00 pp over r3 |
| candidate upside | 0 … +0.4 pp on the graded score at best; **does not reach rank 1** (needs 2.2 %) |
| candidate downside | negative on the only three-band full-path measurement; a never-validated static-link ELF; the whole gate chain re-run with ~1 day left |

The asymmetry decides it: the upside cannot change the rank, and the downside
includes losing a working rank-2 package.

## 6. What to hand back to the L267–L269 lane

`L267_L269_REPORT.md` §4 limit 1 says *"Heavy band only — 40 cases, n ≥ 101,
sample s1. The deployed score is 100 cases across three bands."* **That is the limit
that mattered**, and it is cheap to close: one 100-case official eval per arm,
~10 minutes each, no new tooling.

The generalisable version: **8/8 split-halves across two disjoint OOS corpora did
not predict the sign on the deployment corpus.** Transfer between two samples drawn
the same way is not transfer to a differently-shaped one. Pair it with
`[[l271-no-constant-still-needs-s2]]`, which is the same lesson from the other
direction — there, a constant-free mechanism that passed 4/4 split-halves failed s2.

## 7. Files

```
l274_ship_patch.py       constructive_l267.cpp -> constructive_ship.cpp (3 flips)
constructive_ship_src.cpp   frozen snapshot of the probe source it was cut from
constructive_ship.cpp / .exe
l274_gate.py             G1 default==measured arm, G2 kill switch==stock

results_L274_base_48c.json       1.226325   stock, 48c, 100 cases
results_L274_base_48c_rep2.json  repeat -- 0/100 quality diff, 4.0% runtime spread
results_L274_ship_48c.json       l269p1
results_L274_p2_48c.json         l269p2
l274_base.log l274_base2.log l274_ship.log l274_p2.log
```

`constructive.cpp` and `build_submission.D/` were not modified.
