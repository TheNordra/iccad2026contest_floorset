#!/bin/sh
# L173 -- ATTRIBUTE the 6.2x wall regression against M73.
#
# l166 LANE 1 measured the current tree at 8.78 s/case against M73's 1.41 s/case
# on the same box, same harness, same ICCAD_ADAPTIVE_CORES=48, with the shape LP
# OFF. So it is the POOL. The pool only grew 41 -> 51 profiles (1.24x), which
# cannot explain 6x, so it must be per-profile cost -- and the prime suspect is
# the M80 knob-cloud tier, because CLAUDE.md records that vectors of that family
# run "5-12 s per case and become the 48-core max-setter on their own", which is
# exactly why ORDER_SWAP/MOVE were excluded from the M79 sampling.
#
# If the wall is max-setter bound, a slower max-setter costs the same on ANY
# core count -- so this would transfer to the grader in full.
#
# Single cases via --test-id, on the heaviest block counts, where the measured
# regression is worst (9.16x on n=101-120, which carries 71.4% of the weight).
# LP off throughout: this is a pool question and the LP would only add noise.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l173.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
CASES="99 96 93 90"
printf '%-16s %-6s %-10s\n' arm case runtime_s
for arm in default nom80 nom80_not5 nopool; do
  case $arm in
    default)    E="" ;;
    nom80)      E="ICCAD_M80_TIER=0" ;;
    nom80_not5) E="ICCAD_M80_TIER=0 ICCAD_M67F_TIER5=0" ;;
    nopool)     E="ICCAD_M80_TIER=0 ICCAD_M67F_TIER5=0 ICCAD_ADAPTIVE_POOL=0" ;;
  esac
  for c in $CASES; do
    env ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP=0 $E \
      "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
      --test-id "$c" -o "../_l173_${arm}_${c}.json" \
      > "../_l173_${arm}_${c}.log" 2>&1
    t=$("$PY" -c "import json,sys
try:
    d=json.load(open('../_l173_${arm}_${c}.json'))['test_results'][0]
    print('%.3f %d' % (d['runtime_seconds'], d['block_count']))
except Exception as e:
    print('ERR', e)")
    printf '%-16s %-6s %s\n' "$arm" "$c" "$t"
  done
done
echo L173_ATTRIB_DONE
