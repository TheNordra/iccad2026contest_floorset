#!/bin/sh
# L234 -- implement the two changes L230/L231 priced, and re-stage.
#
#   1. _M49_REFINE_BAND mid band 6 -> 2       (L231)
#   2. _L196_LPGATE + {62,72,74,90,102,103,106,119}, NOTHING REMOVED  (L230)
#
# They are one change, not two: the mid-band cut frees the wall the eight new
# block counts spend, and four of the eight (62,72,74,90) only become affordable
# because of it. Implementing either alone leaves value on the table and
# re-deriving the gate afterwards would mean two gate cycles.
#
# ABORTS unless the mid band's OOS cost has actually been measured and came in
# above the bar. In set the cut is +0.0003%, i.e. free -- and it was free in set
# for the heavy band too (+0.0400%) before OOS turned it into -0.1788%. M50/M74
# derived the 6 strictly selection-preserving, so going below it changes
# selections BY CONSTRUCTION and the in-set null is not evidence.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l234.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
cd "$R" || exit 1
while [ -d "$R/.l233.lock" ]; do sleep 30; done

"$PY" -u l233_score.py | tee l233_score.out
grep -q "L233_VERDICT=GO" l233_score.out || {
  echo "L234 NOT IMPLEMENTING -- l233_score.py did not print GO."; exit 1; }

echo; echo "=== L234 implementing ==="
"$PY" - <<'PYX'
import pathlib
p = pathlib.Path("optimizer_constructive.py"); s = p.read_text(encoding="utf-8")

OLD = '''_M49_REFINE_BAND: Tuple[Tuple[int, int, str], ...] = (
    (60, 100, "6"),                              # M50 universal tier (M74: 8 -> 6)
    (100, 10**9, "2"),                           # M49 -> L223 (was "4")
)'''
NEW = '''_M49_REFINE_BAND: Tuple[Tuple[int, int, str], ...] = (
    # L231 (2026-08-25): the MID band goes 6 -> 2, for the same reason the heavy
    # band went 4 -> 2 at L223 and on the same evidence shape. M50 set 8 and M74
    # took it to 6 by a STRICTLY SELECTION-PRESERVING argument -- the free half,
    # and then it stopped. After L223, 67.3% of the REMAINING RF deficit sat in
    # this band with 25 of its 40 cases still ABOVE the floor.
    #   wall     the mid-band profile phase drops 21.2% (min-of-3 on both sides,
    #            against control bands reading -4.7% at n<=60 and -0.1% at
    #            n>100, i.e. the estimator's own bias -- and the bias points the
    #            conservative way, so 21.2% is a floor on the cut). 6 -> 4 buys
    #            NOTHING (+0.9%, inside the noise) and 6 -> 3 buys 9.7%: the
    #            refine loop early-breaks around 4.
    #   quality  in set +0.0003%, i.e. free -- which is exactly what the heavy
    #            band read before OOS turned it into -0.1788%. The OOS arms are
    #            l233_{s1,s2}_mid2.json against l223_{s1,s2}_r2.json.
    #   joint    with the eight added LP-gate block counts, NET +5.05% vs beta
    #            against +4.27% shipped, and still +4.75% at a -0.30% OOS cost.
    # Kill switch ICCAD_L231_REFINE_MID=<n> restores any value; =6 is pre-L231.
    (60, 100, "2"),                              # M50 -> M74 "6" -> L231 "2"
    (100, 10**9, "2"),                           # M49 -> L223 (was "4")
)'''
assert s.count(OLD) == 1, "REFINE band not in the expected pre-L231 state"
s = s.replace(OLD, NEW)

OLD2 = '''            _kv = os.environ.get("ICCAD_L223_REFINE_HEAVY", "")
            if _kv and block_count > 100:
                return {"ICCAD_REFINE_ITERS": _kv}
            return {"ICCAD_REFINE_ITERS": iters}'''
NEW2 = '''            _kv = os.environ.get("ICCAD_L223_REFINE_HEAVY", "")
            if _kv and block_count > 100:
                return {"ICCAD_REFINE_ITERS": _kv}
            # L231 kill switch, mid band only, same contract as L223's.
            _km = os.environ.get("ICCAD_L231_REFINE_MID", "")
            if _km and block_count <= 100:
                return {"ICCAD_REFINE_ITERS": _km}
            return {"ICCAD_REFINE_ITERS": iters}'''
assert s.count(OLD2) == 1, "band kill-switch block not in the expected state"
s = s.replace(OLD2, NEW2)

ADDS = (62, 72, 74, 90, 102, 103, 106, 119)
import re
m = re.search(r"^_L196_LPGATE = \{.*?^\}", s, re.S | re.M)
tbl = eval(m.group(0).split("=", 1)[1])
for n in ADDS:
    assert tbl[n] == 0, f"block count {n} is already on -- the table is not the pre-L234 one"
    tbl[n] = 1
lines = []
for start in range(21, 121, 10):
    lines.append("    " + " ".join(
        f"{n}: {tbl[n]}," for n in range(start, min(start + 10, 121))))
new_tbl = "_L196_LPGATE = {\n" + "\n".join(lines) + "\n}"
s = s[:m.start()] + new_tbl + s[m.end():]
p.write_text(s, encoding="utf-8", newline="\n")
print(f"   mid band 6 -> 2 (+ ICCAD_L231_REFINE_MID kill switch)")
print(f"   LP gate + {list(ADDS)} -> {sum(tbl.values())} on, 0 removed")
PYX
[ $? -eq 0 ] || { echo "L234 patch failed."; exit 1; }

"$PY" -m py_compile optimizer_constructive.py || exit 1
ICCAD_ADAPTIVE_CORES=48 "$PY" -c "
import sys, os; sys.argv=['x']
import optimizer_constructive as O
assert O._band_env(80).get('ICCAD_REFINE_ITERS')=='2', O._band_env(80)
assert O._band_env(120).get('ICCAD_REFINE_ITERS')=='2', O._band_env(120)
assert O._band_env(40)=={}, O._band_env(40)
os.environ['ICCAD_L231_REFINE_MID']='6'
assert O._band_env(80).get('ICCAD_REFINE_ITERS')=='6'
assert O._band_env(120).get('ICCAD_REFINE_ITERS')=='2', 'mid kill switch leaked into the heavy band'
os.environ['ICCAD_L223_REFINE_HEAVY']='4'
assert O._band_env(120).get('ICCAD_REFINE_ITERS')=='4'
assert O._band_env(80).get('ICCAD_REFINE_ITERS')=='6'
del os.environ['ICCAD_L231_REFINE_MID'], os.environ['ICCAD_L223_REFINE_HEAVY']
on=[n for n in range(21,121) if O._lp_gate_ok(n)]
assert len(on)==71, len(on)
for n in (62,72,74,90,102,103,106,119): assert O._lp_gate_ok(n), n
print('   band + gate checks pass: mid 2 / heavy 2 / both kill switches isolated / gate 71 on')
" 2>&1 | grep -v Warning | grep -vE "^\[scipy" || exit 1
"$PY" -u make_submission.py stage 2>&1 | tail -3
echo "   op_wrapper: $(md5sum build_submission/cadc1075/op_wrapper.py | cut -c1-32)"
echo "   tar       : $(md5sum build_submission/cadc1075.tar.gz | cut -c1-32)"
echo L234_STAGED
