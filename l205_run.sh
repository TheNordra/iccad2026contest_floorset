#!/bin/sh
# L205 (v2) -- profile durations in the SHIPPED 48-core configuration, route A
# OFF, from the PROBE COPY. The shipped tree carries no instrument: it is
# byte-identical to the package that passed the eight in-set gates and the five
# Linux lanes (op_wrapper md5 bb44bb147231fee7bc9670cdc28448bc), and it stays
# that way. optimizer_l205probe.py differs from it only by a locked file
# writer.
#
# v2 fixes the v1 measurement, which printed from 51 threads to one stderr and
# lost ~10% of its records to interleaving (5100 emitted, 4588 parseable).
set -u
LOCK=/c/ICCAD_ml/ship_final/.l205.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
for r in 1 2; do
  rm -f "../l205_prof_r${r}.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_ROUTE_A=0 \
      ICCAD_PROFILE_TIMING="../l205_prof_r${r}.txt" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
    -o "../results_L205_r${r}.json" > "../l205_r${r}.log" 2>&1
  echo "r${r} exit=$?  records=$(wc -l < "../l205_prof_r${r}.txt" 2>/dev/null || echo 0) (expect 5100)"
done
cd /c/ICCAD_ml/ship_final || exit 1
# route A is documented result-neutral; these runs have it OFF while the L199
# gates had it ON, so this is a free cross-check of that claim.
"$PY" - <<'PYX'
import json
J = lambda f: {r["test_id"]: r for r in json.load(open(f))["test_results"]}
try:
    a, b = J("results_L199_det1.json"), J("results_L205_r1.json")
except Exception as e:
    print("   route-A neutrality cross-check skipped:", e); raise SystemExit(0)
ids = sorted(set(a) & set(b))
c = sum(1 for i in ids if a[i]["cost"] == b[i]["cost"])
p = sum(1 for i in ids if a[i]["positions"] == b[i]["positions"])
print("   ROUTE A RESULT-NEUTRAL (on in L199, off here): cost {}/{}  "
      "positions {}/{}   {}".format(c, len(ids), p, len(ids),
                                    "PASS" if c == len(ids) == p else "FAIL"))
PYX
echo; echo "=== run 1 ==="; "$PY" -u l205_imbalance.py l205_prof_r1.txt
echo; echo "=== run 2 ==="; "$PY" -u l205_imbalance.py l205_prof_r2.txt
echo L205_DONE
