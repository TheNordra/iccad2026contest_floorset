#!/bin/sh
cd /c/ICCAD_ml/ship_final || exit 1
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
export PYTHONIOENCODING=utf-8
"$PY" -u l235_lpbench.py ab --minn 0 --limit 0 --reps 3
echo L235_AB_ALL_DONE
