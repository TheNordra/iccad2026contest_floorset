#!/bin/sh
# L186 -- are the L124 MIB twins worth anything OUT of sample, with M80 present?
#
# IN SET they are worth exactly nothing: cur (51 profiles = M80 + twins) and
# m80only (43 = M80, twins off) are identical on 100/100 cases on BOTH cost and
# positions. The twins cost 10.59s of pool wall for that, which prices at
# +0.53pp of RF on the 2026-08-23 medians.
#
# WHY THAT IS NOT ENOUGH TO ACT ON. CLAUDE.md's M74 doctrine is this exact trap:
# "strict in-sample 等價 != OOS 等價" -- tier-3's pruning was strictly in-sample
# identical and still lost 0.702% on held-out cases. And L124's own verdict has
# the twins at OOS s1 +1.35% / s2 +0.59% with a proxy realisation rate of
# 68-88%, i.e. the proxy DOES pick twins out of sample. In-set 0.00% and OOS
# +1.35% cannot both describe the same pool, so this measures it directly.
#
# ONE variable: ICCAD_M124_TWIN. Everything else is the shipped default.
#
# ICCAD_ROUTE_A=0 in BOTH arms, for wall only. Route A is verified
# result-neutral -- single cases are bit-identical with it on and off
# (n=60 1.210020, n=120 1.267585), and L177 det1 vs det2 matched 100/100 on
# cost AND positions with it live. It costs 2.9x on this 16-physical-core box
# and changes nothing that is being measured here.
#
# LOCKFILE: four measurements were destroyed today by two agents sharing this
# box (see _quarantine/).
set -u
LOCK=/c/ICCAD_ml/ship_final/.l186.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ABORT: $LOCK exists -- another copy is running"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
for S in s1 s2; do
  for ARM in twins notwins; do
    if [ "$ARM" = notwins ]; then TW="ICCAD_M124_TWIN=0"; else TW="ICCAD_M124_TWIN_UNSET=1"; fi
    env ICCAD_ROUTE_A=0 $TW \
      "$PY" -u l140_oos_soft_audit.py run --sample "$S" --cores 48 \
      --out "l186_${S}_${ARM}.json" > "l186_${S}_${ARM}.log" 2>&1
    echo "$S/$ARM exit=$?  SAfallback=$(grep -c 'SA fallback' "l186_${S}_${ARM}.log")  $(grep -E 'feasible|weighted cost' "l186_${S}_${ARM}.log" | tr -s ' ' | tr '\n' ' ')"
  done
done
echo L186_TWIN_OOS_DONE
