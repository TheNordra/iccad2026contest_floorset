#!/bin/bash
# L300 step 1 -- the default-cores anchor, run on the PACKAGES (not the tree).
#
# Below the >=40-core gate `_shape_lp_on()` is False, so the LP never runs and
# the mix tables are unreachable. The two packages must therefore be BIT-IDENTICAL
# at default cores. That is the inertness gate, and it also produces the anchor
# the Linux `final` lane compares against.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
S="C:/Users/0150B8~1/AppData/Local/Temp/claude/C--ICCAD-ml/4a197078-0638-4e16-b996-5ce08b9cbf48/scratchpad/l300"
rm -rf "$S"; mkdir -p "$S"
cd /c/ICCAD_ml/ship_final || exit 1
for arm in ship mix; do
  tar=build_submission.D/cadc1075.tar.gz
  [ "$arm" = mix ] && tar=build_submission.MIX/cadc1075.tar.gz
  d="$S/$arm"; mkdir -p "$d"; tar xzf "$tar" -C "$d"
  cp iccad2026contest/iccad2026_evaluate.py "$d/cadc1075/"
  for f in litetestLoader.py lite_dataset_test.py liteLoader.py lite_dataset.py \
           prime_dataset.py cost.py utils.py visualize.py; do cp "$f" "$d/"; done
  cp -r LiteTensorDataTest "$d/" 2>/dev/null || ln -s "$PWD/LiteTensorDataTest" "$d/LiteTensorDataTest"
  echo "=== $arm (default cores) ==="; date +"    start %H:%M:%S"
  ( cd "$d/cadc1075" && env -u ICCAD_ADAPTIVE_CORES \
      "$PY" -u iccad2026_evaluate.py --evaluate op_wrapper.py \
      -o "/c/ICCAD_ml/ship_final/l300_win32_${arm}.json" ) \
      > "l300_win32_${arm}.log" 2>&1
  echo "    exit=$?"; date +"    end   %H:%M:%S"
  grep -E "Total Score|^Feasible" "l300_win32_${arm}.log"
  grep -Ei "fallback|unavailable|\[constructive\]" "l300_win32_${arm}.log" | head -3
done
echo L300_WIN32_DONE
