#!/bin/sh
# L244 -- build and verify variant D: the shipped package MINUS vendor/.
#
# WHY D AND NOT B. The 2026-08-23 Beta evaluation report settles the two
# questions the team emailed about:
#   §2(a) scipy is NOT pre-installed -- importing it without listing it is named
#         as a cause of runtime failures -- and requirements.txt "must list EVERY
#         package your code imports ... scipy ... MUST appear". So removing the
#         scipy line (variant B) would violate an explicit instruction. B is dead.
#   §4    the checklist prescribes `python -m venv` + `pip install -r
#         requirements.txt` + run. So a non-empty requirements.txt is the
#         sanctioned mechanism, not a risk, and the hypothesised
#         "no network -> venv build fails -> zero" path is not supported.
# Consequently scipy arrives via pip, vendor/ is never reached, and a 116 MB
# payload that is genuinely never loaded is exactly what "no unused large
# binaries" prohibits.
#
# D changes NOTHING in the code: op_wrapper.py, op_src.py, constructive.cpp,
# bin/constructive_linux and requirements.txt are byte-identical to the shipped
# package. Only vendor/ is absent. That is asserted below, not assumed.
#
# Three gates:
#   G1  every remaining file byte-identical to build_submission/
#   G2  with scipy AVAILABLE, the package reproduces results_L237_post.json
#       100/100 on cost AND positions  (vendor/ was never being used anyway)
#   G3  with scipy BLOCKED, the package still runs: 100/100 feasible, 0 LP,
#       no SA fallback, no crash -- this is the rank-4 floor, measured
set -u
LOCK=/c/ICCAD_ml/ship_final/.l244.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
cd "$R" || exit 1
export PYTHONIOENCODING=utf-8

echo "=== L244 build variant D  $(date -u +%FT%TZ) ==="
rm -rf build_submission.D
mkdir -p build_submission.D
cp -r build_submission/cadc1075 build_submission.D/cadc1075
rm -rf build_submission.D/cadc1075/vendor
( cd build_submission.D && tar czf cadc1075.tar.gz cadc1075 )
echo "  D tar: $(ls -l build_submission.D/cadc1075.tar.gz | awk '{print $5}') bytes"
echo "  A tar: $(ls -l build_submission/cadc1075.tar.gz | awk '{print $5}') bytes"

echo; echo "=== G1: every remaining file byte-identical to the shipped package ==="
FAIL=0
for f in op_wrapper.py op_src.py requirements.txt README.md constructive.cpp bin/constructive_linux; do
  a=$(md5sum "build_submission/cadc1075/$f" | cut -c1-32)
  b=$(md5sum "build_submission.D/cadc1075/$f" | cut -c1-32)
  if [ "$a" = "$b" ]; then s="OK  "; else s="FAIL"; FAIL=1; fi
  printf "  %s %-24s %s\n" "$s" "$f" "$a"
done
V=$(tar tzf build_submission.D/cadc1075.tar.gz | grep -c 'vendor/')
N=$(tar tzf build_submission.D/cadc1075.tar.gz | wc -l)
echo "  vendor entries in D: $V (must be 0)   total entries: $N (must be 8)"
[ "$V" -eq 0 ] || FAIL=1
[ "$FAIL" -eq 0 ] || { echo "  G1 FAIL"; exit 1; }
echo "  G1 PASS"

echo; echo "=== G2: scipy AVAILABLE -- D must reproduce the shipped results ==="
cd "$R/iccad2026contest" || exit 1
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS=../l244_D_scipy_stats.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../build_submission.D/cadc1075/op_wrapper.py \
  -o ../results_L244_D_scipy.json > ../l244_D_scipy.log 2>&1
echo "  exit=$?  LPran=$(wc -l < ../l244_D_scipy_stats.txt 2>/dev/null || echo 0)  scipy=$(grep -o 'scipy] source=[a-z]*' ../l244_D_scipy.log | head -1)"
cd "$R" || exit 1

echo; echo "=== G3: scipy BLOCKED -- the rank-4 floor, measured not assumed ==="
mkdir -p _l244_noscipy
cat > _l244_noscipy/scipy.py <<'PYS'
raise ImportError("L244: scipy deliberately blocked to measure the no-scipy floor")
PYS
cd "$R/iccad2026contest" || exit 1
env ICCAD_ADAPTIVE_CORES=48 PYTHONPATH="$R/_l244_noscipy" \
    ICCAD_SHAPE_LP_STATS=../l244_D_noscipy_stats.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../build_submission.D/cadc1075/op_wrapper.py \
  -o ../results_L244_D_noscipy.json > ../l244_D_noscipy.log 2>&1
echo "  exit=$?  LPran=$(wc -l < ../l244_D_noscipy_stats.txt 2>/dev/null || echo 0) (must be 0)"
echo "  SAfallback=$(grep -c 'SA fallback' ../l244_D_noscipy.log)  scipy=$(grep -o 'scipy] source=[a-z]*' ../l244_D_noscipy.log | head -1)"
cd "$R" || exit 1

"$PY" - <<'PYX'
import json, math, sys
L = lambda f: {r["block_count"]: r for r in json.load(open(f))["test_results"]}
ref = L("results_L237_post.json")
a = L("results_L244_D_scipy.json")
b = L("results_L244_D_noscipy.json")
c = sum(1 for n in ref if ref[n]["cost"] == a[n]["cost"])
p = sum(1 for n in ref if ref[n]["positions"] == a[n]["positions"])
print(f"  G2  D(with scipy) vs shipped: cost {c}/100  positions {p}/100"
      f"   {'PASS' if c == p == 100 else 'FAIL'}")
W = lambda n: math.exp(n / 12.0)
SW = sum(W(n) for n in ref)
q = lambda d: sum(W(n) * d[n]["cost"] for n in d) / SW
fa = sum(1 for n in a if a[n]["is_feasible"])
fb = sum(1 for n in b if b[n]["is_feasible"])
print(f"  G3  D(no scipy): feasible {fb}/100, weighted cost {q(b):.9f}"
      f"   (with scipy {q(a):.9f})")
print(f"      the LP is worth {100*(q(b)-q(a))/q(b):+.4f}% of quality; losing it"
      f" is the rank-4 floor, and the package does NOT crash")
sys.exit(0 if (c == 100 and p == 100 and fa == 100 and fb == 100) else 1)
PYX
[ $? -eq 0 ] || { echo "  !! a gate failed"; exit 1; }
echo
echo "  D tar md5 (not reproducible): $(md5sum build_submission.D/cadc1075.tar.gz | cut -c1-32)"
echo "  D identity (op_wrapper):      $(md5sum build_submission.D/cadc1075/op_wrapper.py | cut -c1-32)"
echo L244_DONE $(date -u +%FT%TZ)
