#!/bin/sh
# L206 -- everything that has to happen after G2c, in order, unattended.
#
#   1. L205   profile-duration imbalance, 2 runs, route A OFF (the measurement
#             that decides whether route A can pay for its 1.44x work).
#   2. INERT  prove the ICCAD_PROFILE_TIMING instrument added to
#             optimizer_constructive.py changed NOTHING on the shipped path,
#             by re-running the default arm and bit-comparing to
#             results_L199_det1.json. The tree was edited AFTER the package was
#             staged and after the L199 gates ran, so this is not a formality:
#             an instrument that perturbs the shipped path would invalidate
#             every gate above it. Asserted by measurement, not by argument.
#   3. RESTAGE only if INERT passes -- so the tar matches the tree again.
set -u
R=/c/ICCAD_ml/ship_final
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd "$R" || exit 1
while ! grep -q "L202_CHAIN_DONE" l202_chain.out 2>/dev/null; do sleep 20; done

echo "=== L205 profile imbalance $(date -u +%FT%TZ) ==="
sh l205_run.sh

echo; echo "=== INERTNESS of the ICCAD_PROFILE_TIMING instrument ==="
cd "$R/iccad2026contest" || exit 1
rm -f ../l199_det3_stats.txt
env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP_STATS=../l199_det3_stats.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
  -o ../results_L199_det3.json > ../l199_det3.log 2>&1
echo "det3 exit=$?  LPran=$(wc -l < ../l199_det3_stats.txt)"
cd "$R" || exit 1
"$PY" - <<'PY'
import json, sys
J = lambda f: {r["test_id"]: r for r in json.load(open(f))["test_results"]}
a, b = J("results_L199_det1.json"), J("results_L199_det3.json")
ids = sorted(set(a) & set(b))
c = sum(1 for i in ids if a[i]["cost"] == b[i]["cost"])
p = sum(1 for i in ids if a[i]["positions"] == b[i]["positions"])
ok = c == len(ids) and p == len(ids)
print("   INSTRUMENT INERT: cost {}/{}  positions {}/{}   {}"
      .format(c, len(ids), p, len(ids), "PASS" if ok else "FAIL"))
open("_l206_inert.flag", "w").write("PASS" if ok else "FAIL")
PY
if [ "$(cat _l206_inert.flag)" = "PASS" ]; then
  echo; echo "=== RESTAGE (tree == tar again) ==="
  "$PY" -u make_submission.py stage 2>&1 | tail -8
  echo "   op_wrapper md5: $(md5sum build_submission/cadc1075/op_wrapper.py | cut -c1-32)"
  echo "   tar md5:        $(md5sum build_submission/cadc1075.tar.gz | cut -c1-32)"
else
  echo "!! NOT RESTAGING -- the instrument perturbed the shipped path."
  echo "!! Revert the two ICCAD_PROFILE_TIMING hunks before shipping."
fi
echo "L206_CHAIN_DONE $(date -u +%FT%TZ)"
