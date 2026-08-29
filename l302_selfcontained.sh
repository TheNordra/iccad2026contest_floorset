#!/bin/bash
# L302 -- pin mix's dt with a SELF-CONTAINED estimator, repeated.
#
# THE KEY. Every arm runs pass 1 on the 71 exactly as `ship` does (proven
# bit-for-bit, L296 G1), so k1_71 CANCELS in every arm's dt. It only entered the
# earlier numbers because whole-LP walls were being differenced across runs --
# and it happened to be the noisiest unit (34 % over 5 observations). Removing
# the differencing removes it entirely:
#
#     gate0 dt = LP wall on the 29                       (ship spends 0 there)
#     mix   dt = LP wall on the 29  +  pass 2+ on the 71 (pass 1 cancels)
#
# Both terms come from the SAME process in the SAME run. L159 built the per-pass
# timer for exactly this: "Timing both passes in the SAME process removes the
# drift entirely instead of trying to average it away."
#
# Repeats give min-of-N on the two terms that actually matter. `ship` is repeated
# too, to demonstrate the cancellation rather than assume it.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
BIN="C:/ICCAD_ml/ship_final/constructive.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
run () { tag="$1"; opt="$2"; shift 2
  env ICCAD_ADAPTIVE_CORES=48 ICCAD_CONSTRUCTIVE_BIN="$BIN" ICCAD_LP_TIMING=1 "$@" \
    "$PY" -u iccad2026_evaluate.py --evaluate "../$opt" \
    -o "../l302_${tag}.json" > "../l302_${tag}.log" 2>&1
  echo "$tag exit=$? $(grep -o '\[lptime\]' "../l302_${tag}.log" | wc -l) lines $(grep -o 'Total Score: [0-9.]*' "../l302_${tag}.log")"; }
for r in 1 2 3 4; do run mix_$r l296_mix_optimizer.py; done
run ship_1 optimizer_constructive.py
run ship_2 optimizer_constructive.py
echo L302_DONE
