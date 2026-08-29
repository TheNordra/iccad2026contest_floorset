#!/bin/sh
# L185 -- re-gate with L171 pulled (_LP_HB_K = "0").
#
# Only TWO arms are needed, because two of the four gates already cover the
# L171-off configuration:
#
#   G2 kill switch  l147off already ran with L171 off -- the `_LP_HB_K if _on
#                   else "0"` guard takes it down with ICCAD_SHAPE_LP_L147=0 --
#                   and it PASSED 100/100 against results_L165_l147off.json.
#   G3 map fired    results_L177_hboff's stats give {1:54, 2:26, 3:20}, inside
#                   the map, so the depth gate still fires with L171 off.
#
# What is missing:
#   det   a SECOND sample of the L171-off configuration, to pair with
#         results_L177_hboff.json for determinism. Those two are comparable
#         because `os.environ.get("ICCAD_LP_HB_PRED", _LP_HB_K)` returns "0"
#         whether it comes from the env or from the new code default -- the
#         same value reaches the same branch, so the run is identical.
#   k1    the depth kill switch WITH L171 off, so G4's quality delta is
#         measured against an anchor in the same configuration as the arm.
#         The L177 k1 arm had L171 ON, so it cannot serve.
#
# Run AFTER editing _LP_HB_K to "0". Both arms set ICCAD_LP_HB_PRED=0 as well,
# so the script is valid either side of that edit and cannot silently measure
# the wrong thing if the edit is forgotten -- and l185_verdict checks that the
# default and the env agree.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l185.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () {
  tag="$1"; shift
  rm -f "../l185_${tag}_stats.txt"
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_LP_HB_PRED=0 \
      ICCAD_SHAPE_LP_STATS="../l185_${tag}_stats.txt" "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "../results_L185_${tag}.json" > "../l185_${tag}.log" 2>&1
  echo "$tag exit=$?  SAfallback=$(grep -c 'SA fallback' "../l185_${tag}.log")  scipy=$(grep -c 'scipy] source=' "../l185_${tag}.log")"
}
run det2
run k1 ICCAD_SHAPE_LP_DEPTH2=0
echo L185_GATES_DONE
