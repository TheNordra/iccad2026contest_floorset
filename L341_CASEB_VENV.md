# L341 — the Case B venv lane, run end to end (2026-08-28)

Closes the peer review's **§E "Case B untested"**. Log: `l341_caseb.log`,
script: `l341_caseb_venv.sh`.

## Why it had to be run, and why now

`C_QA_20260827.pdf` removed the "if":

* **A25 / A29** — "please include all the dependencies and their versions in
  requirements.txt (option 2). **We will create a venv and use it.**"
* **A27** — "Our pipeline **now always creates a venv when requirements.txt is
  non-empty**."
* **A17** — beta submissions **did** break on environment issues and had to be re-run.

Our `requirements.txt` is non-empty (7 entries) ⇒ the venv path is **certain**, not
hypothetical. And it is **new exposure since beta**: the beta package shipped an
*empty* `requirements.txt` (the "leave it empty" reading was only overturned
2026-08-23), so no venv was ever created for us.

**L313's five lanes do not cover this.** They all run against a pre-existing
`$HOME/iccadvenv` with the libraries already present — that tests the *code* on
Linux, not whether `pip install -r requirements.txt` resolves at all.

⚠️ **This lane is not RF-SAFE-specific.** `requirements.txt` is **byte-identical**
between D and RF-SAFE (`6c59feb458f1f48247373d8b69f401c2`), so the exposure was
already committed the moment D went up. A failure here would have meant **both**
packages needed a fix; it could never have been a reason to hold RF-SAFE back.

## Stage 1 — fresh venv, no system site packages

    base interpreter   /usr/bin/python3   Python 3.14.4
    pip install -r requirements.txt       exit 0, 4m11s

    torch        2.13.0+cu130       matplotlib   3.11.1
    numpy        2.5.2              tqdm         4.70.0
    scipy        1.18.1             requests     2.34.2
    shapely      2.1.2              python       3.14.4

All seven import. **`>=` floors resolved cleanly** — which settles the one place
`C_QA_20260827` A27 ("pinned compatible versions") reads as being in tension with
the `l246` rule taken from report B 75-76 ("no pinned versions, so Python 3.13 can
resolve"). A27 answers Q27, *"Recommendation for **Numba** in final submission"*, and
is scoped to that team's Numba/llvmlite problem; we ship no Numba, and the floors
resolve. **No change to `requirements.txt`.**

🔑 **This is a stricter test than the grader's.** Report B indicates Python 3.13;
this ran on **3.14.4**, where wheel availability is thinner. Resolving on 3.14 makes
resolving on 3.13 close to certain — not the other way round.

## Stage 2 — the official command, inside that venv

`l117_linux_verify.py` launches the evaluator with `sys.executable`, so invoking it
with the fresh venv's interpreter *is* the Case B path.

    official cmd (cwd=cadc1075): iccad2026_evaluate.py --evaluate op_wrapper.py
    eval done in 105 s, exit 0
    Tests 100 | Feasible 100 | Avg Cost 1.3170 | Avg Runtime 0.75 s

    total new = 1.280025696987226
    total old = 1.2800256969872261      |d| = 2.220e-16   (one ULP)
    worst |dcost| = 0.000e+00           every case bit-identical
    scipy: system                       bundled-first OK (no on-site compile)

    L117 LINUX-VERIFY [final/l341_caseb]: PASS

## What this does and does not establish

**Does:** `pip install` resolves from our file on a clean interpreter; all imports
work; the official command completes; 100/100 feasible; the result is bit-identical
to the Windows anchor; the shipped ELF is used rather than an on-site compile; and
`scipy: system` — the peer's `logcheck` gate.

**Does not:** this ran at WSL's 32 cores, i.e. **below** the `≥40` gate, so RF-SAFE
is inert here by design and matches D bit-for-bit — that is the expected result, and
it is also an independent re-confirmation of gate inertness. The 48-core behaviour
and the +0.6994 % gain are **L313 lane 4's** result, not this one's.

## Trap

Calling WSL from Git Bash mangles `/mnt/c/...` into `C:/Program Files/Git/mnt/c/...`
(MSYS path conversion) and the script dies with exit 127. Prefix with
`MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'`.
