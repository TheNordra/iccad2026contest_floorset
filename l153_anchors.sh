#!/usr/bin/env bash
# L153 step 1: the two Windows anchors the Linux verify needs, on THIS tree
# (L137 base + L147 patch). The existing results_L147_lpoff.json / _default
# anchors were taken on the L136 base, so they price L137 into the Linux
# verdict if reused -- silent-failure mode #4 (mismatched measurement bases).
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1

run () {  # run <tag> <env...>
  tag="$1"; shift
  echo "=== $tag  $* ==="
  env -u ICCAD_SHAPE_LP -u ICCAD_SHAPE_LP_R -u ICCAD_SHAPE_LP_G \
      -u ICCAD_SHAPE_LP_TOL -u ICCAD_SHAPE_LP_PRICE -u ICCAD_ADAPTIVE_CORES \
      "$@" "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
      -o "../results_${tag}.json" > "../l153_${tag}.log" 2>&1
  echo "  $tag exit=$?"
  # silent-degradation gate: the C++ lane must not have fallen back to the SA
  grep -inE "fallback|unavailable|all profiles failed|\[constructive\]" \
      "../l153_${tag}.log" | head -5
}

# pre-LP anchor for judge48 (whole LP lane off, 48-core config otherwise)
run L153_lpoff_L137 ICCAD_ADAPTIVE_CORES=48 ICCAD_SHAPE_LP=0
# default-cores anchor for the Linux `final` (bundled-ELF) lane
run L153_default_L137
echo L153_ANCHORS_DONE
