#!/usr/bin/env bash
# L160b -- the one extrapolation left: f was measured on C++-dominated work, but
# the LP is Python+scipy. Measure the LP itself on WSL (scipy 1.18.0) against the
# Windows number (scipy 1.15.3, 19.3-22.6s over 100 cases) to see how sensitive
# LP timing is to environment at all.
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
cd "$R/iccad2026contest" || exit 1
env ICCAD_ADAPTIVE_CORES=48 ICCAD_LP_TIMING=1 ICCAD_SHAPE_LP_L147=0 \
    "$V" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py \
    -o "$R/results_L160b_wsl_band.json" > "$R/l160b_wsl_band.log" 2>&1
echo "exit=$?"
"$V" - "$R/l160b_wsl_band.log" <<'PY'
import re,sys,statistics as st
d=[float(m.group(2)) for m in re.finditer(r"\[lptime\] n=(\d+) cpu=[\d.]+ wall=([\d.]+)",
    open(sys.argv[1],errors="ignore").read())]
print(f"   WSL LP wall: {len(d)} cases  sum {sum(d):.2f}s  p50 {st.median(d):.4f}s")
print(f"   Windows was: sum 19.3-22.6s (3 reps)")
PY
echo L160B_DONE
