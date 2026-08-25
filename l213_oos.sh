#!/bin/sh
# L213 -- the OOS test the pool drop has to pass, and the one this project has
# twice failed on this exact mechanism.
#
# L211/L212 measured the drop in set: k=8 costs -0.1242% of quality and buys
# -5.50% of case wall, NET +2.305% vs beta against today's +1.260%. That is a
# licence to run this, not a result. L138/L139 both found fixed drop sets that
# looked fine in sample and removed 12 of 22 held-out winners.
#
# THE SPECIFIC RISK, which the in-set number structurally cannot see: the table
# is keyed on BLOCK COUNT, and in set there is exactly one case per block count
# -- the very case whose timings the table was fitted on. The OOS samples carry
# 2-4 DIFFERENT floorplans at each block count, where a dropped profile may well
# be the winner. The timing side transports (durations track problem size); the
# winner side is exactly what is being tested here.
#
# Four runs, ~35 min each: baseline and k=8, on both disjoint 240-case samples.
#
# L211_DROP_TABLE is deliberately NOT prefixed ICCAD_: l140_oos_soft_audit
# imports m67_oos_probe first, which strips every ICCAD_* before the optimizer
# module is imported, so an ICCAD_-named knob would be SILENTLY ignored here and
# the k8 arm would quietly measure the baseline. The abort below is the belt to
# that braces: the probe prints "drop table loaded" at import, and an arm that
# does not print it did not run the thing it claims to.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l213.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -q "L212_DONE" l212_bigk.out 2>/dev/null; do sleep 20; done
echo "=== L213 OOS for the pool drop  $(date -u +%FT%TZ) ==="
for S in s1 s2; do
  for ARM in base k8; do
    if [ "$ARM" = "base" ]; then DROP=""; else DROP="l211_drop_k8.json"; fi
    env ICCAD_ROUTE_A=0 ICCAD_OOS_OPT=optimizer_l205probe L211_DROP_TABLE="$DROP" \
      "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
      --out "l213_${S}_${ARM}.json" > "l213_${S}_${ARM}.log" 2>&1
    rc=$?
    T=$(grep -c 'drop table loaded' "l213_${S}_${ARM}.log")
    M=$(grep -c 'optimizer module -> optimizer_l205probe' "l213_${S}_${ARM}.log")
    echo "$S/$ARM exit=$rc  probe_module=$M  drop_table=$T  SAfallback=$(grep -c 'SA fallback' "l213_${S}_${ARM}.log")"
    if [ "$M" -eq 0 ]; then
      echo "  !! ABORT: the probe module was not used -- this arm measured the shipped tree"; exit 1
    fi
    if [ "$ARM" = "k8" ] && [ "$T" -eq 0 ]; then
      echo "  !! ABORT: the drop table was NOT loaded -- this arm measured the baseline"; exit 1
    fi
    if [ "$ARM" = "base" ] && [ "$T" -ne 0 ]; then
      echo "  !! ABORT: the baseline arm loaded a drop table"; exit 1
    fi
  done
done
"$PY" -u l213_score.py
echo L213_DONE
