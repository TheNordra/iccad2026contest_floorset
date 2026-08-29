#!/bin/sh
# L173b -- SPLIT the per-case wall into the part that scales with cores and the
# part that does not. This is the number that decides whether the 6.2x wall
# regression against M73 transfers to a 48-core grader.
#
# l173_attrib.sh established that this box is SUM-BOUND, not max-setter bound:
# dropping the 8 M80 profiles (51 -> 43, none of them the max-setter) moved
# n=120 from 8.433s to 6.497s. Under max-setter binding that change is a no-op.
#
# So wall(C) = a + b/C   with `a` the serial part (the M47 proxy tail, the
# wrapper, the LP when on) and `b/C` the pool. Fit on C = 4, 8, 16, 32 real
# cores via CPU affinity -- affinity is inherited by the constructive.exe
# children, so the whole case is confined -- then read off a and b/48.
#
#   a  transfers to the grader in full, on any core count.
#   b/C  is what a bigger box buys back.
#
# ICCAD_ADAPTIVE_CORES stays 48 throughout: that is what selects the shipped
# tier configuration, and we are measuring THAT configuration on fewer cores,
# not a different configuration.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l173b.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
# affinity masks: 4, 8, 16, 32 logical cores
printf '%-6s %-6s %-10s\n' cores case runtime_s
for pair in "4:F" "8:FF" "16:FFFF" "32:FFFFFFFF"; do
  C="${pair%%:*}"; MASK="${pair##*:}"
  for c in 99 93; do
    cmd //c "start \"\" /affinity $MASK /wait /b \"$PY\" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py --test-id $c -o ../_l173c_${C}_${c}.json > ../_l173c_${C}_${c}.log 2>&1" >/dev/null 2>&1
    t=$("$PY" -c "import json
try:
    d=json.load(open('../_l173c_${C}_${c}.json'))['test_results'][0]
    print('%.3f n=%d' % (d['runtime_seconds'], d['block_count']))
except Exception as e:
    print('ERR', e)")
    printf '%-6s %-6s %s\n' "$C" "$c" "$t"
  done
done
echo L173_CORES_DONE
