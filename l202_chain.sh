#!/bin/sh
# L202 -- chain G2c onto the end of the Linux lanes.
#
# Deliberately a SEPARATE process rather than an edit to l200_drive.sh: that
# script is currently executing, and sh reads a running script incrementally
# from a byte offset, so editing it in place can make the shell resume in the
# middle of a different line and execute garbage. Appending a waiter costs one
# process and cannot corrupt a run that is 40 minutes in.
set -u
R=/c/ICCAD_ml/ship_final
cd "$R" || exit 1
while ! grep -q "L200_DRIVE_DONE" l200_drive.out 2>/dev/null; do sleep 20; done
echo "=== Linux lanes finished, running G2c $(date -u +%FT%TZ) ==="
sh l201_g2c.sh
echo "L202_CHAIN_DONE $(date -u +%FT%TZ)"
