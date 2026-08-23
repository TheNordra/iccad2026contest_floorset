#!/usr/bin/env bash
# L163c -- the vendored wheel is cp313 and nothing here runs cp313 by default
# (conda 3.10, WSL 3.14). pyenv has 3.13.5 -- same ABI tag, which is all the
# wheel cares about. Prove it actually imports and solves.
set -u
P="$HOME/.pyenv/versions/3.13.5/bin/python3.13"
V=/mnt/c/ICCAD_ml/ship_final/vendor
echo "interpreter: $($P --version 2>&1)"
# The grader HAS numpy (contest requirements.txt lists it, torch needs it);
# this bare pyenv interpreter does not, so install it to stand in for the
# grader's. scipy 1.16.3 declares numpy<2.6,>=1.25.2.
$P -c "import numpy" 2>/dev/null || $P -m pip install -q "numpy>=1.25.2,<2.6"
echo "numpy for the test: $($P -c 'import numpy;print(numpy.__version__)' 2>&1 | tail -1)"
echo "--- 1. scipy absent from this interpreter to begin with? ---"
$P -c "import scipy" 2>&1 | tail -1
echo "--- 2. does the VENDORED wheel import and solve a real LP? ---"
$P - "$V" <<'PY'
import sys
sys.path.append(sys.argv[1])
import numpy, scipy
from scipy import sparse
from scipy.optimize import linprog
print("   scipy", scipy.__version__, "loaded from vendor:", "/vendor/" in scipy.__file__)
print("   numpy", numpy.__version__)
r = linprog(c=[-1, -2], A_ub=[[1, 1], [1, -1]], b_ub=[4, 2], bounds=(0, None))
print(f"   linprog status={r.status} fun={r.fun:.6f}")
assert r.status == 0 and abs(r.fun + 8.0) < 1e-6, f"unexpected optimum {r.fun}"
m = sparse.csr_matrix([[1, 0], [0, 2]])
print("   sparse ok, nnz =", m.nnz)
print("   *** VENDORED SCIPY WORKS ON cp313 ***")
PY
