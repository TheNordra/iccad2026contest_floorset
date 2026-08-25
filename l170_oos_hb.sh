#!/bin/sh
# L170 -- does the predicted-HPWL-baseline objective transfer out of sample?
#
# ⚠️ The first attempt at this ran with the flag set but the port had silently
# FAILED (an assert in the patch script, masked by a passing syntax check that
# followed it), so the arm was a no-op measuring nothing. Hence the liveness
# check below: the flag must CHANGE the in-set result before either OOS sample
# is worth 35 minutes.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
LP="ICCAD_SHAPE_LP_R=1.5 ICCAD_SHAPE_LP_G=1.10 ICCAD_SHAPE_LP_TOL=0.006 ICCAD_SHAPE_LP_PRICE=1.0"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
env ICCAD_ADAPTIVE_CORES=48 ICCAD_LP_HB_PRED=0.2994 "$PY" -u iccad2026_evaluate.py \
  --evaluate ../optimizer_constructive.py -o ../results_L170_live.json \
  > ../l170_live.log 2>&1
echo "liveness arm exit=$?"
cd /c/ICCAD_ml/ship_final || exit 1
"$PY" - <<'PY'
import json, math
W=lambda n: math.exp(n/12.0)
J=lambda f:{r["test_id"]:r for r in json.load(open(f))["test_results"]}
a,b = J("results_L170_live.json"), J("results_L165_det1.json")
ids=sorted(set(a)&set(b))
same=sum(1 for i in ids if a[i]["cost"]==b[i]["cost"])
t=lambda q: sum(W(q[i]["block_count"])*q[i]["cost"] for i in ids)
print(f"  identical to the shipped arm on {same}/{len(ids)} cases")
print(f"  quality {100*(t(b)-t(a))/t(b):+.4f}%")
open("_l170_live.flag","w").write("LIVE" if same < len(ids) else "NOOP")
PY
if [ "$(cat _l170_live.flag)" != "LIVE" ]; then
  echo "ABORT: the flag changed nothing -- not spending 70 minutes on a no-op"; exit 1
fi
for S in s1 s2; do
  echo "=== $S with ICCAD_LP_HB_PRED=0.2994 ==="
  env $LP ICCAD_LP_HB_PRED=0.2994 ICCAD_SHAPE_LP_STATS=l170_oos_stats_${S}.txt \
    "$PY" -u l140_oos_soft_audit.py run --sample $S --cores 48 \
    --out l170_oos_${S}_hb.json > l170_${S}_hb.log 2>&1
  echo "  exit=$?"
done
echo L170_OOS_DONE
