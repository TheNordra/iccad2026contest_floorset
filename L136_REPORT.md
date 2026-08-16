# L136 — the MARGIN fix, built and verified on Linux: +0.6272% graded

Implements L135. The uploaded artefact is untouched; `build_submission/` was
backed up, used, and restored byte-for-byte. The new package is in
**`build_submission.L136FIX/`** and is **not uploaded**.

    48c LINUX (GRADED)  1.2362075522257698 -> 1.2284538948373953   +0.6272%
    48c Windows         1.2367916697725434 -> 1.2284738198320346   +0.6725%
    32c default (Win)   1.293461035226291  -> 1.2772224039603648   +1.2555%
    feasible            100/100 -> 100/100  (both platforms)
    avg runtime         0.9799s -> 0.9795s  (Windows)

## 1. The change

One constant and two lines. `constructive.cpp` gains

    static const double FRAME_EPS = 1e-9;

and the two frame/seed sizing sites (`:695`, `:1984`) use it instead of MARGIN:

    w = max(w, max(pre_w + FRAME_EPS, max_iw + FRAME_EPS));   // was MARGIN

MARGIN (1e-4) is retained everywhere else — the packing candidate offsets
(`:750-751`) and the escape rows (`:1216-1223`) are untouched.

**Why this is the fix.** MARGIN is a hundred times the evaluator's boundary
tolerance (`TOL = 1e-6`, matching `iccad2026_evaluate.py:527`). The frame was
built 1e-4 larger than the preplaced extent it had to contain, so a preplaced
block carrying a boundary requirement sat at the true edge while the bbox ran to
edge+1e-4 — a violation it could never satisfy, because its position is a hard
constraint and the packing is too tightly abutted for the bbox to be pulled in
(L135 §4). FRAME_EPS keeps the strict containment MARGIN was there for while
staying three orders of magnitude inside TOL.

## 2. It is worth about twice what L135 predicted, and that is expected

| | 48c | vs shipped |
|---|---|---|
| shipped | 1.2367916697725434 | — |
| L131 abutment only | 1.2358546851248895 | +0.0758% |
| **L131 + L136 MARGIN** | **1.2284738198320346** | **+0.6725%** |

MARGIN's own contribution over L131 is **+0.5972%**, against L135's estimate of
+0.3566%.

🔑 **L135's figure was an upper bound on REMOVING THE VIOLATIONS, not on the
fix.** It assumed nothing else changed. But the frame is an input to the packing,
so a smaller frame also packs better: 36 of 100 cases changed cost and 50 changed
positions. The extra ~0.24pp is packing quality, not violations. An estimate that
holds everything else fixed is a lower bound whenever the knob is upstream of the
search — the opposite of the L131 §4 error, where the estimate was 3× too high.

## 3. Linux binary: rebuilt, and it had to be

🚨 **`constructive.cpp` alone would not have changed the graded result.** The
README's run order is (1) bundled `bin/constructive_linux`, (2) on-site compile.
The grader uses the bundled binary, so a source-only change would have shipped
the OLD behaviour while every local Windows measurement showed the new one —
because on Windows `_ensure_compiled` skips the bundle (`os.name != "nt"`,
`optimizer_constructive.py:1011`) and compiles `constructive.exe`.

Rebuilt with the M67-C command (`m67c_make_linux_bundle.py:90`):

    g++ -O3 -std=c++17 -static-libstdc++ -static-libgcc -o bin/constructive_linux constructive.cpp

on WSL2 **Ubuntu 26.04 / g++ 15.2.0** — a much newer host than the original
Ubuntu 22.04 / g++11, which was the risk to check:

| | shipped | rebuilt |
|---|---|---|
| type | ELF 64-bit LSB **pie** executable, x86-64 | **same** |
| max GLIBC | **2.38** | **2.38** |
| md5 | `62602aba…` | `6d43cf2c…` |

Max GLIBC is unchanged at **2.38** against the grader's 2.41, because the program
only references old symbols — the build host's glibc does not matter unless newer
symbols are actually used. 1-block smoke passes.

**Equivalence check**: the new Linux binary's stdout is **byte-identical** to the
new Windows `constructive.exe` on cases 7, 21, 54 and 66 (including the
MARGIN-affected ones). So the two builds agree, and the known Windows/Linux score
gap (1.2367916697725434 vs 1.2362075522257698) comes from the Python/LP layer,
not the C++.

## 3b. ✅ Verified under Linux, on the grader's path

Installed CPU torch **2.12** (the grader's version — closer than the Windows
env's 2.11), shapely, scipy, tqdm and matplotlib into WSL: ~1.1 GB, no files
deleted. The official command was then run against the built package with
`ICCAD_ADAPTIVE_CORES=48`.

| | Windows | Linux |
|---|---|---|
| shipped (48c) | 1.2367916697725434 | 1.2362075522257698 |
| **L136 (48c)** | 1.2284738198320346 | **1.2284538948373953** |
| **gain vs shipped** | **+0.6725%** | **+0.6272%** |

100/100 feasible on both. **The gain holds on the platform that is actually
graded**, at +0.6272%.

🔑 **The bundled binary was genuinely used**: the run left no `constructive.exe`
compile artefact, which is the hard proof that the M67-A bundled-binary-first
path fired rather than the on-site compile fallback. That is the path a
source-only change would have missed entirely (§3).

Side observation, not chased: the Windows/Linux disagreement **shrank 29×**, from
`+0.000584` on the shipped package to `+0.0000199` on L136. The two platforms now
agree to 2e-5. Plausibly the 1e-4 frame slack was itself a source of
platform-dependent divergence, but the mechanism was not investigated and this is
reported as an observation, not a claim.

⚠️ Avg runtime reads **5.11s** under WSL against 0.98s on Windows. That is drvfs
overhead on `/mnt/c`, not a regression — local scoring uses
`RuntimeFactor = 1.0` (the evaluator says so explicitly), so it does not enter
the score.

## 4. 🚨 A packaging mistake worth recording

The first `build_submission.L136FIX/` I saved contained the **old** Linux binary.
`make_submission.py verify` does **not** re-stage — it verifies the existing tar —
and the last `stage` had run during the l113 gate, *before* the binary was
swapped in. The scores were unaffected (Windows never reads the bundled binary),
so nothing in the measurements was wrong; only the artefact was.

🔑 **`stage` after every input change, and check the binary's md5 in the staged
tree, not in the repo root.** The package can be a version behind its own sources
with no symptom at all on Windows.

## 5. Both gates FAIL, and that is still correct

`make_submission.py verify` and `l113_ship_gate.py --cores 48` are bit-exactness
checks against the frozen anchor, so any improvement fails them:

    verify:  new total_score=1.2772224039603648 != 1.293461035226291
    gate:    G4 total 1.2284738198320346 != 1.2367916697725434
             cost differs on 36 case(s), positions on 50

100/100 feasible on both. **A new anchor is required before either gate means
anything again.**

## 6. Package identity

| file | shipped | L136FIX |
|---|---|---|
| `op_wrapper.py` / `op_src.py` | `ad8c5dcb…` | `2967efb6876f70685a18e1a160644fdd` |
| `constructive.cpp` | `937d0e15…` | `570ee27001df8c04afb07a8da4ecb1f2` |
| `bin/constructive_linux` | `62602aba…` | `6d43cf2cbfd9e4d578cd692277a7f868` |
| `cadc1075.tar.gz` | `c08a0844…`, 373,844 B | `2db47211…`, 377,124 B |

The tar md5 is **not reproducible** (gzip embeds an mtime — `.gitattributes`
says so); track identity by the three file md5s.

Backups: `constructive.cpp.preL136`, `bin/constructive_linux.preL136`,
`build_submission.SHIPPED.bak/`.

## 7. Reproduce

```bash
PATH="/c/msys64/ucrt64/bin:$PATH" "C:/Users/.01/anaconda3/envs/floorset/python.exe" l113_ship_gate.py --cores 48 --anchor results_L114_48c_lp_anchor.json
```
```bash
wsl.exe -e bash -lc "cd /mnt/c/ICCAD_ml/ship_final && g++ -O3 -std=c++17 -static-libstdc++ -static-libgcc -o bin/constructive_linux constructive.cpp && objdump -T bin/constructive_linux | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1"
```

The Linux score (the graded one). WSL needs `torch shapely scipy tqdm matplotlib`
(~1.1 GB; use torch 2.12 to match the grader). `_l136_linux/` is built by
extracting `build_submission.L136FIX/cadc1075.tar.gz`, overlaying
`iccad2026contest/iccad2026_evaluate.py`, copying `make_submission._LOADER_FILES`
and linking `LiteTensorDataTest` — the same preparation `l113_ship_gate.py` does:
```bash
wsl.exe -e bash -lc "cd /mnt/c/ICCAD_ml/ship_final/_l136_linux/cadc1075 && PYTHONIOENCODING=utf-8 ICCAD_ADAPTIVE_CORES=48 python3 -u iccad2026_evaluate.py --evaluate op_wrapper.py -o results_l136_linux.json"
```
Afterwards check that **no `constructive.exe` exists** in that directory — that is
the proof the bundled binary ran rather than the on-site compile fallback.

## 8. What is now on the table

Two built, verified, unuploaded packages:

| | 48c LINUX (graded) | binary rebuilt? | Linux-verified? |
|---|---|---|---|
| `build_submission.L131FIX/` | not measured (Windows +0.0758%) | no — md5 unchanged | no |
| **`build_submission.L136FIX/`** | **1.2284538948373953 (+0.6272%)** | **yes** | **yes, 100/100** |

L136 is ~8× the gain, and it is now the better-VERIFIED of the two: it is the
only package measured on the platform that is actually graded, through the
bundled-binary path, at 100/100 feasible. Its residual risk is that the packing
moved on half the cases and the binary comes from a much newer toolchain
(glibc verified equal at 2.38, PIE preserved, output byte-identical to the
Windows build). Both remain decisions, not tasks.
