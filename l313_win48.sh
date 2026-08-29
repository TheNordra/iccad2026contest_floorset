#!/bin/bash
# L313 step 1 -- reproduce the peer session's Windows 48c headline independently,
# and produce the --win reporting anchor the Linux lane wants.
# Claimed: 1.215239132, +0.9040 % vs D's 1.226325126, 12 movers, 0 worse, 100/100.
set -u
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
S="$LOCALAPPDATA/Temp/claude/C--ICCAD-ml/819ab4de-1538-4ac8-8df8-e763f29dd928/scratchpad/l313"
mkdir -p "$S"; cd /c/ICCAD_ml/ship_final || exit 1
d="$S/rfsafe"; mkdir -p "$d"; tar xzf build_submission.RFSAFE/cadc1075.tar.gz -C "$d"
cp iccad2026contest/iccad2026_evaluate.py "$d/cadc1075/"
for f in litetestLoader.py lite_dataset_test.py liteLoader.py lite_dataset.py \
         prime_dataset.py cost.py utils.py visualize.py; do cp "$f" "$d/"; done
cmd //c mklink //J "$(cygpath -w "$d/LiteTensorDataTest")" "$(cygpath -w "$PWD/LiteTensorDataTest")" >/dev/null 2>&1
echo "=== RFSAFE @ 48c (Windows) ==="; date +"    start %H:%M:%S"
( cd "$d/cadc1075" && env ICCAD_ADAPTIVE_CORES=48 \
    ICCAD_CONSTRUCTIVE_BIN="C:/ICCAD_ml/ship_final/constructive.exe" \
    "$PY" -u iccad2026_evaluate.py --evaluate op_wrapper.py \
    -o /c/ICCAD_ml/ship_final/l313_win48_rfsafe.json ) > l313_win48.log 2>&1
echo "    exit=$?"; date +"    end   %H:%M:%S"
grep -E "Total Score|^Feasible" l313_win48.log
echo L313_WIN48_DONE
