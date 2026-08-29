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

`requirements.txt` is the contest's own dependency list plus `scipy`, which the
shape-legalisation LP uses (`scipy.optimize.linprog`). Every entry is a `>=`
constraint, none are pinned. A copy of scipy is also vendored under `vendor/`
and is appended to `sys.path` ONLY if `import scipy` fails, so a system scipy
always wins and the LP never depends on the vendored copy being reachable.
