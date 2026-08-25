#!/bin/sh
set -u
cd /c/ICCAD_ml/ship_final || exit 1
while ! grep -q "L205_DONE" l205_run.out 2>/dev/null; do sleep 20; done
echo "=== L205 parallel done; starting the uncontended run $(date -u +%FT%TZ) ==="
sh l205b_seq.sh
echo "L205C_CHAIN_DONE $(date -u +%FT%TZ)"
