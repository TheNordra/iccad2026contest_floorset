#!/usr/bin/env bash
# L200 -- Linux verify of the L196 package: _L157_DEPTH flattened to all-1s
# plus the per-case LP gate _L196_LPGATE (s=1.2, 63 of 100 block counts).
# op_wrapper.py md5 bb44bb147231fee7bc9670cdc28448bc.
#
# Supersedes l187_wsl_verify.sh, whose preconditions describe the package that
# was staged BEFORE L196 and would now abort on a correct tar:
#
#  1. l187 asserts the depth histogram is {1:52, 2:18, 3:30} -- the x0.90 map,
#     superseded. The flat map is {1:100}. l187 also had no way to see the LP
#     gate at all, so a package with the gate stripped out would have passed it.
#  2. l187 judges lane 3 against results_L177_det1.json, the merged tree's
#     in-set anchor. The L196 tree's anchor is results_L199_det1.json.
#  3. NEW LANE 5. The gate is the entire change, and a table that silently kept
#     its old values passes determinism, the kill switch, and every lane l187
#     runs while changing nothing -- the failure this project records most
#     often. ICCAD_SHAPE_LP_STATS only writes a line when the LP actually
#     executes, so the line count IS the liveness proof: 63 by default, 100
#     under ICCAD_LP_GATE=0. l117_linux_verify.py now checks the SET, not the
#     count, against the table inside the tar.
#
# Run INSIDE WSL. The bare `wsl -d Ubuntu -- bash /mnt/c/...` form gets
# MSYS-mangled into C:/Program Files/Git/mnt/c/... -- that is what l187's
# 99-byte log is. `bash -lc '...'` alone is NOT enough either: MSYS still
# rewrites /mnt/... inside the quoted string, and the failure mode is a
# variable that expands to empty rather than an error, so the script runs and
# measures nothing. The form that actually works, verified 2026-08-25:
#
#   MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
#     wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l200_wsl_verify.sh'
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l200
TAR=$R/build_submission/cadc1075.tar.gz
WANT_MD5=bb44bb147231fee7bc9670cdc28448bc
cd "$R" || exit 1
mkdir -p "$L117_WORK"

echo "== L200 Linux verify -- L196 package  $(date -u +%FT%TZ)  nproc=$(nproc)"
echo "   tar md5:        $(md5sum $TAR | cut -c1-32)"
GOT_MD5=$(tar xzOf $TAR cadc1075/op_wrapper.py | md5sum | cut -c1-32)
echo "   op_wrapper md5: $GOT_MD5"
echo "                   (expect $WANT_MD5)"

# --- hard preconditions on the tar itself, before spending an hour on lanes ---
DEPTH=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "_L157_DEPTH")
GATE=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "_L196_LPGATE")
GOK=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "not _lp_gate_ok(a\[0\])")
REQB=$(tar xzOf $TAR cadc1075/requirements.txt | wc -c)
SCIPY=$(tar xzOf $TAR cadc1075/requirements.txt | grep -c "^scipy")
HB=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "_LP_HB_K")
VEND=$(tar tzf $TAR | grep -c 'cadc1075/vendor/')
echo "   _L157_DEPTH hits: $DEPTH   _L196_LPGATE hits: $GATE   _LP_HB_K hits: $HB"
echo "   gate consulted in _shape_lp_maybe: $GOK"
echo "   requirements.txt: $REQB bytes, scipy listed: $SCIPY"
echo "   vendor files:     $VEND"
FAIL=0
[ "$GOT_MD5" = "$WANT_MD5" ] || { echo "   !! op_wrapper md5 is not the L196 package -- restage"; FAIL=1; }
[ "$DEPTH" -ge 2 ] || { echo "   !! _L157_DEPTH missing from the tar"; FAIL=1; }
[ "$GATE" -ge 2 ] || { echo "   !! _L196_LPGATE missing from the tar -- the gate is not in the package"; FAIL=1; }
[ "$GOK" -ge 1 ] || { echo "   !! _lp_gate_ok is never consulted -- the table is inert"; FAIL=1; }
[ "$REQB" -gt 50 ] || { echo "   !! requirements.txt is empty/short -- the 0-byte revert is back"; FAIL=1; }
[ "$SCIPY" -ge 1 ] || { echo "   !! scipy not listed in requirements.txt"; FAIL=1; }
[ "$VEND" -gt 1000 ] || { echo "   !! vendor/ missing"; FAIL=1; }

# The two tables that must be inside the tar: depth all-1s, gate 63 on / 37 off.
tar xzOf $TAR cadc1075/op_src.py > "$L117_WORK/_op_src.py"
"$V" - "$L117_WORK/_op_src.py" <<'PY'
import re, sys
from collections import Counter
s = open(sys.argv[1], encoding="utf-8").read()
def table(name):
    m = re.search(r"^" + name + r" = \{.*?^\}", s, re.S | re.M)
    return eval(m.group(0).split("=", 1)[1]) if m else None
d, g = table("_L157_DEPTH"), table("_L196_LPGATE")
hd = dict(sorted(Counter(d.values()).items())) if d else None
hg = dict(sorted(Counter(g.values()).items())) if g else None
ok_d = hd == {1: 100}
ok_g = hg == {0: 37, 1: 63}
n_hi = sum(1 for k, v in g.items() if v and k > 100) if g else -1
print(f"   depth map in the tar: {hd}  {'OK' if ok_d else '!! NOT the flat k=1 map'}")
print(f"   LP gate  in the tar: {hg}  {'OK' if ok_g else '!! NOT the s=1.2 gate'}")
print(f"   gate fires on {n_hi} block counts above 100 (s=1.0 keeps only 2)")
sys.exit(0 if ok_d and ok_g else 1)
PY
[ $? -eq 0 ] || FAIL=1
if [ "$FAIL" -ne 0 ]; then
  echo "L200 ABORT: the tar is not the package we think it is."; exit 1
fi
echo "   WSL system scipy: $($V -c 'import scipy;print(scipy.__version__)' 2>&1 | tail -1)"

echo; echo "########## LANE 1 -- 48c, LP off: the Linux pre-LP base ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_lpoff --no-judge \
     --base results_L153_lpoff_L137.json --env ICCAD_SHAPE_LP=0
echo "LANE1_RC=$?"
BASE=$L117_WORK/t_lpoff/cadc1075/results_l117_t_lpoff.json

echo; echo "########## LANE 2 -- 48c, kill switch: the pre-L147 band ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_off --no-judge --stats \
     --base "$BASE" --env ICCAD_SHAPE_LP_L147=0
echo "LANE2_RC=$?"
CTRL=$L117_WORK/t_off/cadc1075/results_l117_t_off.json

echo; echo "########## BUDGET ##########"
"$V" l153_budget.py "$BASE" "$CTRL" | tee "$R/l200_budget_linux.txt"
B=$(grep '^BUDGET' "$R/l200_budget_linux.txt" | awk '{print $2}')
echo "   budget = $B"

echo; echo "########## LANE 3 -- 48c, THE SHIPPED DEFAULT, nothing set ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_ship --stats \
     --base "$BASE" --ctrl "$CTRL" --win results_L199_det1.json \
     --budget "${B:-0}" --live-min 1.5
echo "LANE3_RC=$?"

echo; echo "########## LANE 4 -- DETERMINISM: the same run again ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_ship2 --no-judge --stats \
     --base "$BASE"
echo "LANE4_RC=$?"
A=$L117_WORK/t_ship/cadc1075/results_l117_t_ship.json
Bb=$L117_WORK/t_ship2/cadc1075/results_l117_t_ship2.json
"$V" - "$A" "$Bb" <<'PY'
import json, sys
L = lambda f: {r["test_id"]: r for r in json.load(open(f))["test_results"]}
a, b = L(sys.argv[1]), L(sys.argv[2])
ids = sorted(set(a) & set(b))
c = sum(1 for i in ids if a[i]["cost"] == b[i]["cost"])
p = sum(1 for i in ids if a[i]["positions"] == b[i]["positions"])
print(f"   DETERMINISM on Linux: cost {c}/{len(ids)}  positions {p}/{len(ids)}  "
      f"{'PASS' if c == len(ids) and p == len(ids) else 'FAIL'}")
PY

echo; echo "########## LANE 5 -- THE GATE IS LIVE ON LINUX ##########"
# The whole of L196 is this table. Everything above passes unchanged if the
# gate is inert, so this lane is the one that can actually fail on it.
"$V" l117_linux_verify.py final48 "$TAR" --tag t_gateoff --no-judge --stats \
     --base "$BASE" --env ICCAD_LP_GATE=0
echo "LANE5_RC=$?"
ON=$L117_WORK/t_ship/lp_stats.txt
OFF=$L117_WORK/t_gateoff/lp_stats.txt
NON=$(wc -l < "$ON" 2>/dev/null || echo 0)
NOFF=$(wc -l < "$OFF" 2>/dev/null || echo 0)
echo "   LP executions: default $NON   ICCAD_LP_GATE=0 $NOFF   (expect 63 / 100)"
if [ "$NON" -eq 63 ] && [ "$NOFF" -eq 100 ]; then
  echo "   GATE LIVENESS on Linux: PASS"
else
  echo "   GATE LIVENESS on Linux: FAIL -- the table is inert or the wrong one"
fi
# and the skipped set must be exactly the 0s of the table, not merely 37 of them
"$V" - "$L117_WORK/_op_src.py" "$ON" "$OFF" <<'PY'
import re, sys
s = open(sys.argv[1], encoding="utf-8").read()
g = eval(re.search(r"^_L196_LPGATE = \{.*?^\}", s, re.S | re.M).group(0).split("=", 1)[1])
rd = lambda f: {int(l.split()[0]) for l in open(f) if l.split()}
on, off = rd(sys.argv[2]), rd(sys.argv[3])
want = {n for n, v in g.items() if v}
ok = on == want and off == set(g)
print(f"   gate SET on Linux: default=={{table 1s}} {on == want}   "
      f"gateoff==all {off == set(g)}   {'PASS' if ok else 'FAIL'}")
if not ok:
    print(f"   ran-but-must-not {sorted(on - want)[:12]}  "
          f"must-but-did-not {sorted(want - on)[:12]}")
sys.exit(0 if ok else 1)
PY
echo "LANE5_SET_RC=$?"

echo; echo "L200_DONE $(date -u +%FT%TZ)"
