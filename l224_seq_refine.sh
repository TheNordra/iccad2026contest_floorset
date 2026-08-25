#!/bin/sh
# L224 -- the REFINE ratio measured UNCONTENDED, because the contended one is
# barely above its own noise.
#
# Two IDENTICAL REFINE=4 runs disagree per block count by a median of 17%
# (range 0.269-1.861). The REFINE 4->2 signal is a 20% median cut, so the
# per-case signal-to-noise is 1.2x. Aggregated over the band the noise falls to
# ~3% and the 21.5% band cut is safe -- but NET is computed from PER-CASE cuts
# and is dominated by ~10 heavy cases, which is exactly where that 17% lands.
# It is why REFINE=1 and REFINE=2 scored 0.9pp apart on near-identical band
# cuts.
#
# ICCAD_PROF_SEQ=1 runs the profiles one at a time: no scheduler, so the
# R2/R4 ratio is the workload property it is supposed to be. Same trick L205b
# used to settle route A, and for the same reason.
#
# Only ONE run is needed: the REFINE=4 sequential already exists as
# l205b_prof_seq.txt (route A off, 48c pool, default bands).
set -u
LOCK=/c/ICCAD_ml/ship_final/.l224.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -qE "L223_DONE|ABORT" l223_validate.out 2>/dev/null; do sleep 30; done
echo "=== L224 uncontended REFINE=2  $(date -u +%FT%TZ) ==="
cd /c/ICCAD_ml/ship_final/iccad2026contest || exit 1
rm -f ../l224_prof_seq_r2.txt
env ICCAD_ADAPTIVE_CORES=48 ICCAD_ROUTE_A=0 ICCAD_PROF_SEQ=1 \
    L219_REFINE_HEAVY=2 ICCAD_PROFILE_TIMING=../l224_prof_seq_r2.txt \
  "$PY" -u iccad2026_evaluate.py --evaluate ../optimizer_l205probe.py \
  -o ../results_L224_seq_r2.json > ../l224_seq_r2.log 2>&1
echo "seq R=2 exit=$?  records=$(wc -l < ../l224_prof_seq_r2.txt 2>/dev/null || echo 0) (expect 5100)"
cd /c/ICCAD_ml/ship_final || exit 1
"$PY" -u l224_seq_score.py
echo L224_DONE
