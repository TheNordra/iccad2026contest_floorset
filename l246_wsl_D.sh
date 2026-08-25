#!/usr/bin/env bash
# L246 -- Linux verify of the NO-VENDOR package (build_submission.D).
#
# Derived from l238_wsl_final.sh with ONE assertion inverted. l238 (and l187
# and l207 before it) assert vendor/ as a HARD PRECONDITION -- `VEND > 1000`
# -- and abort before running a single lane. That check was written when
# vendor/ was load-bearing; against D it is exactly backwards and would
# reject the package we are shipping while looking like a real failure.
#
# Everything else in it is still correct and is kept, in particular the
# requirements.txt checks, which are now MORE load-bearing rather than less:
# guidelines Section 2 Case B says a non-empty requirements.txt makes the
# grader build the venv from that file ALONE, so a file that lost scipy would
# take the LP down with it and nothing else in the package would notice.
#
# The package under test:
#   _L157_DEPTH flat all-1s (unchanged)
#   _L196_LPGATE 71 of 100 block counts on (L234: +8, none removed)
#   _M49_REFINE_BAND (60,100,"2") + (100,inf,"2")  (L231: the mid band 6->2)
#   the L235 LP row-construction rewrite, which must be INVISIBLE
#
# Derived from l207_wsl_final.sh. Three thresholds moved because the package
# moved, none because a gate was inconvenient: the gate histogram 63->71, the
# lane-3 in-set anchor L227->L237, and a new pair of assertions for the mid
# band. Everything the old file asserted is still asserted.
# Run against the RESTAGED tar (post-L205 instrument); the md5 is not a
# constant here, it is asserted equal to the staged tree. See note 2 below.
#
# Supersedes l200_wsl_verify.sh (which supersedes l187_wsl_verify.sh), whose preconditions describe the package that
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
#     wsl -d Ubuntu -- bash -lc 'bash /mnt/c/ICCAD_ml/ship_final/l207_wsl_final.sh'
#
# ── L207 differs from L200 in exactly two places, both stale thresholds ──────
#
# 1. `--live-min 1.5`  ->  0.40.  G-D asserts the shipped default is ahead of
#    the L147-off control by at least this much, and 1.5 was set when the LP ran
#    on ALL 100 cases. Under L196 it runs on 63, so the tangent's contribution
#    is diluted: measured in set, the shipped band is +0.6761% ahead of the
#    control, not +1.5%. Left at 1.5 this FAILS a correct package -- the same
#    stale-anchor shape as G2's L165 comparison and l117's LP-liveness count.
#    0.40 is 59% of the measured gap. It is NOT tuned to pass: the failure this
#    gate exists to catch (L147 silently not applying) produces 0.000%, so the
#    margin to the failure mode is the whole 0.40, while the margin to a correct
#    package is 0.28pp on top of it.
#
# 2. the op_wrapper md5 constant -> "equal to the staged tree", plus a positive
#    assertion that route A is off and the instrument is absent. A hardcoded md5 cannot survive a
#    deliberate restage, and silently comparing against the OLD constant would
#    abort on the very package this lane exists to verify.
#
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l246
TAR=$R/build_submission.D/cadc1075.tar.gz
cd "$R" || exit 1
mkdir -p "$L117_WORK"

echo "== L246 Linux verify -- NO-VENDOR package (D)  $(date -u +%FT%TZ)  nproc=$(nproc)"
echo "   tar md5:        $(md5sum $TAR | cut -c1-32)"
GOT_MD5=$(tar xzOf $TAR cadc1075/op_wrapper.py | md5sum | cut -c1-32)
echo "   op_wrapper md5: $GOT_MD5"
echo "                   (must equal the staged tree, checked below)"

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
echo "   vendor files:     $VEND (must be 0 for D)"
FAIL=0
STG=$(md5sum $R/build_submission.D/cadc1075/op_wrapper.py | cut -c1-32)
echo "   staged dir md5: $STG   (tar must equal the staged tree, not a constant)"
[ "$GOT_MD5" = "$STG" ] || { echo "   !! tar does not match the staged tree -- restage"; FAIL=1; }
# The L205 decision: route A off as a CODE default. Asserted positively, and
# the instrument must NOT be here -- it lived in the shipped tree for one hour
# and was moved to optimizer_l205probe.py precisely so the artefact stays the
# one that was verified.
RA=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "OFF since L205")
PD=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "_L211_POOLDROP")
echo "   _L211_POOLDROP hits in the tar: $PD (the L211/L213 pool drop)"
[ "$PD" -ge 3 ] || { echo "   !! the pool drop table is gone from the package"; FAIL=1; }
PDOFF=$(tar xzOf $TAR cadc1075/op_src.py | grep -c 'ICCAD_L211_POOLDROP", "") != "1"')
echo "   pool drop default-OFF guard: $PDOFF (expect 1; it came off at L224)"
[ "$PDOFF" -eq 1 ] || { echo "   !! the pool drop is not default-off -- L224 said it is NET -0.096pp on top of REFINE=2"; FAIL=1; }
RB=$(tar xzOf $TAR cadc1075/op_src.py | grep -c '(100, 10\*\*9, "2")')
RB4=$(tar xzOf $TAR cadc1075/op_src.py | grep -c '(100, 10\*\*9, "4")')
echo "   REFINE heavy band: \"2\" hits $RB, stale \"4\" hits $RB4"
[ "$RB" -ge 1 ] || { echo "   !! the L223 REFINE band is not in the package"; FAIL=1; }
[ "$RB4" -eq 0 ] || { echo "   !! the pre-L223 band value is still present"; FAIL=1; }
RM=$(tar xzOf $TAR cadc1075/op_src.py | grep -c '(60, 100, "2")')
RM6=$(tar xzOf $TAR cadc1075/op_src.py | grep -c '(60, 100, "6")')
echo "   REFINE mid band: \"2\" hits $RM, stale \"6\" hits $RM6"
[ "$RM" -ge 1 ] || { echo "   !! the L231 mid band is not in the package"; FAIL=1; }
[ "$RM6" -eq 0 ] || { echo "   !! the pre-L231 mid value is still present"; FAIL=1; }
MKS=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "ICCAD_L231_REFINE_MID")
echo "   mid-band kill switch present: $MKS (expect >=2)"
[ "$MKS" -ge 2 ] || { echo "   !! the L231 kill switch is missing"; FAIL=1; }
# L235: the rewritten row construction must be the one that was A/B'd, and the
# construct it replaced must be gone -- a partially applied patch would pass
# every other check in this file.
L235=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "np.triu_indices")
# match the ASSIGNMENT, not the prose: the L235 comment quotes the construct it
# replaced, so grepping for "max(cands, key=lambda" hits the explanation.
L235OLD=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "br = max(cands")
echo "   L235 rewrite: np.triu_indices hits $L235, stale separation assignment hits $L235OLD"
[ "$L235" -ge 1 ] || { echo "   !! the L235 separation rewrite is not in the package"; FAIL=1; }
[ "$L235OLD" -eq 0 ] || { echo "   !! the pre-L235 separation loop is still present"; FAIL=1; }
PROF=$(tar xzOf $TAR cadc1075/op_src.py | grep -c "_PROF_TIMING")
echo "   route A off marker: $RA   _PROF_TIMING (must be 0): $PROF"
[ "$RA" -ge 1 ] || { echo "   !! route A is still on -- this tar predates the L205 decision"; FAIL=1; }
[ "$PROF" -eq 0 ] || { echo "   !! the L205 instrument is in the shipping artefact"; FAIL=1; }
[ "$DEPTH" -ge 2 ] || { echo "   !! _L157_DEPTH missing from the tar"; FAIL=1; }
[ "$GATE" -ge 2 ] || { echo "   !! _L196_LPGATE missing from the tar -- the gate is not in the package"; FAIL=1; }
[ "$GOK" -ge 1 ] || { echo "   !! _lp_gate_ok is never consulted -- the table is inert"; FAIL=1; }
[ "$REQB" -gt 50 ] || { echo "   !! requirements.txt is empty/short -- the 0-byte revert is back"; FAIL=1; }
[ "$SCIPY" -ge 1 ] || { echo "   !! scipy not listed in requirements.txt"; FAIL=1; }
# INVERTED for D: vendor/ must be ABSENT. Under Section 2 Case B the grader
# installs scipy from requirements.txt, so a bundled copy is never loaded and
# matches "large binary files ... allowed only if ACTIVELY USED".
[ "$VEND" -eq 0 ] || { echo "   !! vendor/ is STILL in the D package"; FAIL=1; }

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
ok_g = hg == {0: 29, 1: 71}
n_hi = sum(1 for k, v in g.items() if v and k > 100) if g else -1
print(f"   depth map in the tar: {hd}  {'OK' if ok_d else '!! NOT the flat k=1 map'}")
print(f"   LP gate  in the tar: {hg}  {'OK' if ok_g else '!! NOT the L234 gate (71 on)'}")
print(f"   gate fires on {n_hi} block counts above 100 (s=1.0 keeps only 2)")
sys.exit(0 if ok_d and ok_g else 1)
PY
[ $? -eq 0 ] || FAIL=1
if [ "$FAIL" -ne 0 ]; then
  echo "L246 ABORT: the tar is not the package we think it is."; exit 1
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
"$V" l153_budget.py "$BASE" "$CTRL" | tee "$R/l246_budget_linux.txt"
B=$(grep '^BUDGET' "$R/l246_budget_linux.txt" | awk '{print $2}')
echo "   budget = $B"

echo; echo "########## LANE 3 -- 48c, THE SHIPPED DEFAULT, nothing set ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag t_ship --stats \
     --base "$BASE" --ctrl "$CTRL" --win results_L237_post.json \
     --budget "${B:-0}" --live-min 0.40
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
echo "   LP executions: default $NON   ICCAD_LP_GATE=0 $NOFF   (expect 71 / 100)"
if [ "$NON" -eq 71 ] && [ "$NOFF" -eq 100 ]; then
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

echo; echo "L246_DONE $(date -u +%FT%TZ)"
