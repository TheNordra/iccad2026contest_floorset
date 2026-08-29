#!/usr/bin/env bash
# Wait for the OTHER session's L271 jobs to clear, then run the two measurements
# p1 still needs:
#   1. a real stopwatch wall (it only has a proxy) -- MUST have the box to itself
#   2. an s2 re-score (only p2 has one)
# The quiet check requires 3 consecutive clear polls, because that session has
# been launching follow-on jobs; a single clear poll would let a new one start
# in the middle of the timing run and silently corrupt it.
set -u
cd /c/ICCAD_ml/ship_final
PY=/c/Users/.01/anaconda3/envs/floorset/python.exe

others() {
  powershell.exe -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -notlike '*l269p1_followup*' -and (\$_.CommandLine -like '*l271*' -or \$_.CommandLine -like '*l267_*' -or \$_.CommandLine -like '*l252_frames*') }).Count" \
    2>/dev/null | tr -d '\r\n '
}

echo "[followup] waiting for the box to go quiet ..."
clear_streak=0
while [ "$clear_streak" -lt 3 ]; do
  n=$(others)
  case "$n" in ''|*[!0-9]*) n=1 ;; esac
  if [ "$n" -eq 0 ]; then
    clear_streak=$((clear_streak + 1))
  else
    if [ "$clear_streak" -ne 0 ]; then echo "[followup] a new job appeared -- streak reset"; fi
    clear_streak=0
  fi
  sleep 40
done
echo "[followup] box quiet at $(date). starting the WALL run (needs exclusivity)."

"$PY" l267_wall.py --cases 3 --reps 3 \
  --arms ship,l269p1,l269p2,l269p3 \
  --probe constructive_l270.exe --out l269_wall3.pkl > l269_wall3.log 2>&1
echo "[followup] wall done rc=$?"

echo "[followup] starting the s2 re-score for p1 (deterministic, exclusivity not needed)."
"$PY" l267_quality.py --sample s2 --limit 40 \
  --arms ship,l269p1 \
  --probe constructive_l270.exe --out l269p1_s2.pkl > l269p1_s2.log 2>&1
echo "[followup] s2 done rc=$?"
echo "[followup] ALL DONE"
