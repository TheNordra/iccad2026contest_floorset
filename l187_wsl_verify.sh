#!/usr/bin/env bash
# L187 -- Linux verify of the MERGED package (L172 x0.90 depth map + L172
# requirements.txt + L171 hb predictor), op_wrapper.py md5
# 815d02dae4639b880c4985ca63827b33.
#
# Modelled on l166_wsl_verify.sh, with three things corrected:
#
#  1. l166 line 16 greps the tar for `_L157_NSET`, which has not existed since
#     L165 replaced it with `_L157_DEPTH`. It printed "0 hits" and nobody
#     noticed -- an informational line that silently reads zero is how this
#     project loses a session, so it now greps for the constant that IS there
#     AND asserts the depth histogram of the map inside the tar.
#  2. l166 line 40 judges against `results_L165_det1.json`, the OLD map's
#     in-set run. The merged package's in-set anchor is
#     `results_L177_det1.json` (L177 gates: ALL PASS).
#  3. The tar must carry a NON-EMPTY requirements.txt now. l166 predates that
#     change and would not have caught a silent revert.
#
# Run INSIDE WSL:
#   wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l187_wsl_verify.sh
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l187
TAR=$R/build_submission/cadc1075.tar.gz
cd "$R" || exit 1
mkdir -p "$L117_WORK"

echo "== L187 Linux verify -- merged package  $(date -u +%FT%TZ)  nproc=$(nproc)"
echo "   tar md5:        $(md5sum $TAR | cut -c1-32)"
echo "   op_wrapper md5: $(tar xzOf $TAR cadc1075/op_wrapper.py | md5sum | cut -c1-32)"
echo "                   (expect 815d02dae4639b880c4985ca63827b33)"

# --- hard preconditions on the tar itself, before spending an hour on lanes ---
DEPTH=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "_L157_DEPTH")
REQB=$(tar xzOf $TAR cadc1075/requirements.txt | wc -c)
SCIPY=$(tar xzOf $TAR cadc1075/requirements.txt | grep -c "^scipy")
HB=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "_LP_HB_K")
VEND=$(tar tzf $TAR | grep -c 'cadc1075/vendor/')
echo "   _L157_DEPTH hits: $DEPTH   _LP_HB_K hits: $HB"
echo "   requirements.txt: $REQB bytes, scipy listed: $SCIPY"
echo "   vendor files:     $VEND"
FAIL=0
[ "$DEPTH" -ge 2 ] || { echo "   !! _L157_DEPTH missing from the tar"; FAIL=1; }
[ "$REQB" -gt 50 ] || { echo "   !! requirements.txt is empty/short -- the 0-byte revert is back"; FAIL=1; }
[ "$SCIPY" -ge 1 ] || { echo "   !! scipy not listed in requirements.txt"; FAIL=1; }
[ "$VEND" -gt 1000 ] || { echo "   !! vendor/ missing"; FAIL=1; }
# the map that must be inside the tar: 52 ones, 18 twos, 30 threes
tar xzOf $TAR cadc1075/op_src.py > "$L117_WORK/_op_src.py"
"$V" - "$L117_WORK/_op_src.py" <<'PY'
import re, sys
from collections import Counter
s = open(sys.argv[1], encoding="utf-8").read()
d = eval(re.search(r"^_L157_DEPTH = \{.*?^\}", s, re.S | re.M).group(0).split("=", 1)[1])
h = dict(sorted(Counter(d.values()).items()))
ok = h == {1: 52, 2: 18, 3: 30}
print(f"   depth map in the tar: {h}  {'OK' if ok else '!! NOT the x0.90 map'}")
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || FAIL=1
if [ "$FAIL" -ne 0 ]; then
  echo "L187 ABORT: the tar is not the package we think it is."; exit 1
fi
echo "   WSL system scipy: $($V -c 'import scipy;print(scipy.__version__)' 2>&1 | tail -1)"

echo; echo "########## LANE 1 -- 48c, LP off: the Linux pre-LP base ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_lpoff --no-judge \
     --base results_L153_lpoff_L137.json --env ICCAD_SHAPE_LP=0
echo "LANE1_RC=$?"
BASE=$L117_WORK/t_lpoff/cadc1075/results_l117_t_lpoff.json

echo; echo "########## LANE 2 -- 48c, kill switch: the pre-L147 band ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_off --no-judge --stats \
     --base "$BASE" --env ICCAD_SHAPE_LP_L147=0
echo "LANE2_RC=$?"
CTRL=$L117_WORK/t_off/cadc1075/results_l117_t_off.json

echo; echo "########## BUDGET ##########"
"$V" l153_budget.py "$BASE" "$CTRL" | tee "$R/l187_budget_linux.txt"
B=$(grep '^BUDGET' "$R/l187_budget_linux.txt" | awk '{print $2}')
echo "   budget = $B"

echo; echo "########## LANE 3 -- 48c, THE SHIPPED DEFAULT, nothing set ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_ship --stats \
     --base "$BASE" --ctrl "$CTRL" --win results_L177_det1.json \
     --budget "${B:-0}" --live-min 1.5
echo "LANE3_RC=$?"

echo; echo "########## LANE 4 -- DETERMINISM: the same run again ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_ship2 --no-judge --stats \
     --base "$BASE"
echo "LANE4_RC=$?"
A=$L117_WORK/t_ship/cadc1075/results_l117_t_ship.json
Bb=$L117_WORK/t_ship2/cadc1075/results_l117_t_ship2.json
"$V" - "$A" "$Bb" <<'PY'
import json, sys
L = lambda f: {r["test_id"]: r for r in json.load(open(f))["test_results"]}
a, b = L(sys.argv[1]), L(sys.argv[2])
ids = sorted(set(a) & set(b))
c = sum(1 for i in ids if a[i]["cost"] == b[i]["cost"])
p = sum(1 for i in ids if a[i]["positions"] == b[i]["positions"])
print(f"   DETERMINISM on Linux: cost {c}/{len(ids)}  positions {p}/{len(ids)}  "
      f"{'PASS' if c == len(ids) and p == len(ids) else 'FAIL'}")
PY
echo; echo "L187_DONE $(date -u +%FT%TZ)"
