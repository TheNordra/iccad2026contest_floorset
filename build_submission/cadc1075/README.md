# ICCAD 2026 Problem C -- team cadc1075

Entry point: `op_wrapper.py` (class `MyOptimizer`), evaluated as
`python iccad2026_evaluate.py --evaluate op_wrapper.py`.
`op_src.py` is a byte-identical backup copy of the same source.

The solver is a deterministic C++ constructive placer (`constructive.cpp`)
driven by a Python portfolio wrapper. Binary resolution happens once at
optimizer load time (outside the scored per-case window):

1. bundled prebuilt Linux binary `bin/constructive_linux` -- chmod +x, then a
   1-block smoke test; used only if the smoke passes;
2. on-site compile fallback: g++ / clang++ / c++  x  -O3 / -O2
   (`g++ -O3 -std=c++17 -o constructive.exe constructive.cpp`), each candidate
   accepted only after the same 1-block smoke test;
3. pure-Python SA fallback (embedded in op_wrapper.py) if no binary runs.

`requirements.txt` lists every third-party package `op_wrapper.py` imports:
torch, numpy, shapely, scipy. Floors match the contest's own
`requirements.txt` so nothing here can force a version change in the
evaluation environment; `scipy>=1.6.0` is the release that added the HiGHS
backend this code asks for by name (`linprog(..., method="highs")`).

scipy is used by one post-processing step (a shape LP over the selected
layout). It is imported inside a try/except and the step is skipped entirely if
scipy is absent, so the solver still produces valid, feasible layouts without
it -- but the results are meaningfully better with it.
