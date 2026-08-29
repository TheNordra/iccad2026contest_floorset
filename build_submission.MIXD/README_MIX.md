# `build_submission.MIXD` — the `mix` candidate, staged and verified (L303)

**This is the RESERVE. Nothing here has been uploaded.**

> ⚠️ **Superseded as the candidate, 2026-08-28.** The user chose
> **`build_submission.RFSAFE`** (`op_wrapper.py` md5 `62db6ee4…`) as the upload
> target; `build_submission.D` (`1c326784…`) is the previous upload and the
> fallback. `mix` is held in reserve for the case where there is positive
> evidence the Final medians widened — it is the only arm whose leaderboard slot
> moves, and it moves in BOTH directions (rank 1 at f_eff 3.17, rank 4 at 1.90).
> See `UPLOAD_THIS.md` and `SHIP_DECISION_2026-08-28.md`.

## What it is

The shipped package with **two wrapper table defaults changed and nothing else**:

    _L196_LPGATE  ->  all 1s          the shape LP runs on every block count (71 -> 100)
    _L157_DEPTH   ->  2 on the old 1-set, 1 on the old 0-set
                                      a second pass only where L196 judged the
                                      case can afford it

`constructive.cpp` and `bin/constructive_linux` are **byte-identical** to the
shipped package, so no ELF rebuild is involved.

| | |
|---|---|
| `op_wrapper.py` / `op_src.py` | **`2b795995f4a3be3f2bc7154db1de1c49`** |
| `constructive.cpp` | `e2c7b2f418ef2b70b6bff99f7adfbd37` (== shipped) |
| `bin/constructive_linux` | `bc9912072cd97b45b47a03adec7170ce` (== shipped) |
| `requirements.txt` | `6c59feb458f1f48247373d8b69f401c2` (== shipped) |
| tar | 409950 bytes, 8 members, **no vendor/** |

## What it scores

    in-set 100, 48c   1.226325126 -> 1.195229398   +2.5357 %
    LINUX     48c     1.226406964 -> 1.196905117   +2.4056 %   <- the grader's platform
    OOS s1 (240)                                   +2.8874 %
    OOS s2 (240)                                   +2.6750 %
    NET after the RF bill                          +0.26 % … +1.35 %

Projected onto the hidden set: **0.86305 – 0.87240, still rank 2**, but the gap
to rank 1 narrows from **+1.92 %** to **+0.51 % … +1.60 %**.

## Verification actually run (L303)

| lane | result |
|---|---|
| `make_submission.py stage` | **PASS** — binary/source OK, vendor byte-identical to the wheel, hygiene OK |
| `l245_novendor.sh` | **PASS** — six graded files byte-identical to the stage, 0 vendor members |
| Windows, default cores | **1.280025697**, bit-identical to the shipped default, 100/100 cost **and** positions — inert below the ≥40-core gate |
| Windows, 48c | **1.195229398**, bit-identical to the measured `mix` arm, 100/100 |
| Linux 1a, ship, default | PASS, \|d\| 2.2e-16 vs Windows |
| Linux 1b, **mix**, default | **PASS, bit-identical** — inert on Linux too |
| Linux 2, 48c LP off | `1.2589744529416786`, −0.0000 % vs the Windows anchor |
| Linux 3, 48c shipped band | `1.2264069637381392`, LP on exactly 71, **0 regressions ⇒ budget 0** |
| Linux 4, **48c mix, judged** | **PASS** — 100/100 feasible, **0 regressions at budget 0**, **+2.4056 %** over the shipped band, LP on exactly 100/100 |
| Linux 5, t4 | PASS — corrupt ELF falls through to the package's own g++, case 50 reproduces to 0.000e+00 |

Both Windows lanes ran with the package **compiling its own binary** on site. The
resulting `constructive.exe` has a different md5 from the tree's (newer g++) and
produces **bit-identical results**, so the placer is compiler-version stable.

## To reproduce, or to ship

The tree was **restored** after staging, because a second session is editing
`optimizer_constructive.py` concurrently. The change is saved as
`l303_mix.patch` (2 hunks, the two tables). To rebuild:

```
patch optimizer_constructive.py < l303_mix.patch
PATH="C:\msys64\ucrt64\bin;$PATH"   # or the compile chain dies silently -> SA
python make_submission.py stage
sh l245_novendor.sh build_submission.MIXD
```

⚠️ **`build_submission/` currently holds the MIX stage, not the shipped one.**
Re-run `make_submission.py stage` on the restored tree before using it for
anything else.

⚠️ **`C:\msys64\ucrt64\bin` must be on PATH.** Without it `cc1plus.exe` cannot
load its DLLs, `g++` exits 1 with no output, and the package degrades **silently**
to the Python SA — printing `Total Score: 10.0000` with `Feasible: 100/100`,
which is the SA's *feasible* ceiling 9.999999 rounded by `%.4f`, not an
infeasible run.

## Not done

* `regression_suite.py` has **4 pre-existing failures** (`rf`, `m49big`,
  `m49mid` ×2), all `cache profile signature != current pool`. Verified
  **pre-existing**: the un-patched tree fails them identically, and CLAUDE.md
  already records these three as red from a stale `audit_cache`. They are not
  caused by, and do not gate, this change.
* The upload itself. That is the user's decision.
