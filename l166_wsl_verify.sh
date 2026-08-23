#!/usr/bin/env bash
# L158 -- Linux verify of the SHIPPED configuration: tangent by code default,
# depth by the deterministic n-set. Nothing is set by environment here except
# the core count, because that is the whole point -- the grader strips ICCAD_*.
# Run INSIDE WSL:  wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l166_wsl_verify.sh
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l166
TAR=$R/build_submission/cadc1075.tar.gz
cd "$R" || exit 1
mkdir -p "$L117_WORK"

echo "== L166 Linux verify -- per-case depth map  $(date -u +%FT%TZ)  nproc=$(nproc)  $(uname -r)"
echo "   tar: $(md5sum $TAR | cut -c1-32)"
echo "   n-set in tar: $(tar xzOf $TAR cadc1075/op_src.py | grep -c _L157_NSET) hits"
echo "   vendor files in tar: $(tar tzf $TAR | grep -c 'cadc1075/vendor/')"
echo "   WSL HAS system scipy: $($V -c 'import scipy;print(scipy.__version__)' 2>&1 | tail -1)"
echo "   => these lanes exercise the SYSTEM path; vendor/ must sit there unused."

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
"$V" l153_budget.py "$BASE" "$CTRL" | tee "$R/l166_budget_linux.txt"
B=$(grep '^BUDGET' "$R/l166_budget_linux.txt" | awk '{print $2}')
echo "   budget = $B"

echo; echo "########## LANE 3 -- 48c, THE SHIPPED DEFAULT, nothing set ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_ship --stats \
     --base "$BASE" --ctrl "$CTRL" --win results_L165_det1.json \
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

echo; echo "########## PASSES SPENT -- col 4 ##########"
for t in t_off t_ship t_ship2; do
  S=$L117_WORK/$t/lp_stats.txt
  [ -f "$S" ] && echo "   $t  $(awk '{h[$4]++} END {for (k in h) printf "passes=%s:%d  ", k, h[k]}' "$S")" \
              || echo "   $t  NO STATS FILE"
done
echo; echo "L166_WSL_DONE $(date -u +%FT%TZ)"
