#!/bin/sh
cd /c/ICCAD_ml/ship_final || exit 1
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
export PYTHONIOENCODING=utf-8
"$PY" -u l239_solver.py --minn 100 --limit 12 --reps 3
echo L239_DONE
