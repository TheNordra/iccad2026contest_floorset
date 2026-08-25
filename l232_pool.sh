#!/bin/sh
# L232 -- the mirror question: REFINE=2 and the drop sets both PAY QUALITY FOR
# WALL. Now that L223 freed the heavy band to 1.28x slack, what does BUYING
# quality back cost?
#
# Four arms, one batch, 3 repeats, all instrumented so the wall is measured:
#
#   base    the shipped configuration (heavy REFINE=2 via L219_REFINE_TABLE)
#   full    ICCAD_ADAPTIVE_POOL=0 -- every drop set off AND full REFINE. NOT a
#           candidate: it is the joint UPPER BOUND on "stop paying quality for
#           wall", and it is one run instead of a per-set sweep.
#   hint    ICCAD_HINT_POOL=1 -- the L137 GORDIAN hint tier (4 profiles). It is
#           in the shipped tree, measured quality-positive (+0.0437% in set,
#           +0.0889% OOS s1) and left DEFAULT-OFF with the comment "the quality
#           is measured and the runtime is not yet". This prices the runtime.
#   lensd   ICCAD_LENSD_POOL=1 -- the L167 32-vector tier. Measured +0.4519% in
#           set, OOS +0.3383%/+0.5501% at 75%/122% transfer, and killed ONLY on
#           the serial proxy tail (~71 ms/profile) against a budget that has
#           since changed. Re-priced here, not re-argued.
#
# ICCAD_PROFILE_TIMEOUT is raised for the two big-pool arms: the L1 lesson is
# that an oversubscribed pool stretches nominal-48s profiles past 120 s and they
# are then SILENTLY dropped, which would read as "free".
set -u
LOCK=/c/ICCAD_ml/ship_final/.l232.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
while [ -d "$R/.l230.lock" ] || [ -d "$R/.l231.lock" ]; do sleep 30; done
cd "$R/iccad2026contest" || exit 1
echo "=== L232 pool-restore pricing  $(date -u +%FT%TZ) ==="
for I in 1 2 3; do
  for ARM in base full hint lensd; do
    case $ARM in
      base)  set -- ;;
      full)  set -- ICCAD_ADAPTIVE_POOL=0 ICCAD_PROFILE_TIMEOUT=600 ;;
      hint)  set -- ICCAD_HINT_POOL=1 ;;
      lensd) set -- ICCAD_LENSD_POOL=1 ICCAD_PROFILE_TIMEOUT=600 ;;
    esac
    rm -f "../l232_prof_${ARM}_${I}.txt"
    env ICCAD_ADAPTIVE_CORES=48 L219_REFINE_TABLE=../l231_mid6.json "$@" \
        ICCAD_PROFILE_TIMING="../l232_prof_${ARM}_${I}.txt" \
      "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
      -o "../results_L232_${ARM}_${I}.json" > "../l232_${ARM}_${I}.log" 2>&1
    rc=$?
    # profiles-per-case is the liveness proof: an arm that did not change the
    # pool has the same record count as base and is measuring nothing.
    echo "  arm=$ARM rep$I exit=$rc records=$(wc -l < "../l232_prof_${ARM}_${I}.txt" 2>/dev/null || echo 0) SAfallback=$(grep -c 'SA fallback' "../l232_${ARM}_${I}.log")"
  done
done
echo L232_POOL_DONE $(date -u +%FT%TZ)
