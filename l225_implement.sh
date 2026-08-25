#!/bin/sh
# L225 -- implement k=8 + REFINE=2 and re-verify the package end to end.
#
# Waits for L224's uncontended timing and ABORTS if it contradicts the
# contended measurement, because the whole rank-2 claim rests on the wall half.
set -u
LOCK=/c/ICCAD_ml/ship_final/.l225.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "ABORT: lock"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
PY="C:/Users/.01/anaconda3/envs/floorset/python.exe"
R=/c/ICCAD_ml/ship_final
cd "$R" || exit 1
while ! grep -qE "L224_DONE|ABORT" l224_seq.out 2>/dev/null; do sleep 30; done
echo "=== L224 said: ==="; sed -n '/UNCONTENDED/,/^r3 = /p' l224_seq.out | head -30
"$PY" - <<'PYX'
import re, sys, pathlib
t = pathlib.Path("l224_seq.out").read_text(errors="replace")
mb = re.search(r"band cut n>100 : ([+-][\d.]+)%", t)
mc = re.search(r"CONTROL n<=100 : ([+-][\d.]+)%", t)
if not (mb and mc):
    print("!! L224 produced no band/control numbers -- not implementing on a"
          " measurement that did not land."); sys.exit(1)
band, ctrl = float(mb.group(1)), float(mc.group(1))
print(f"   uncontended band cut {band:+.2f}%   control {ctrl:+.2f}%")
# The gate: the cut must clearly exceed the control band, which is the residual
# noise of a measurement that CANNOT legitimately move (REFINE is untouched
# below n=101). 3x is the same margin the contended run cleared.
if band < 10.0 or band < 3 * abs(ctrl):
    print(f"!! ABORT: the uncontended cut does not clear its own control band."
          f" The contended runs read +21..27%; this reads {band:+.2f}% against"
          f" a control of {ctrl:+.2f}%. The rank-2 claim was resting on the"
          f" wall half and the wall half did not hold up.")
    sys.exit(1)
print("   uncontended timing CONFIRMS the contended measurement.")
PYX
[ $? -eq 0 ] || { echo "L225 NOT IMPLEMENTING."; exit 1; }

echo; echo "=== L225 implementing REFINE=2 on the heavy band ==="
"$PY" - <<'PYX'
import pathlib
p = pathlib.Path("optimizer_constructive.py"); s = p.read_text(encoding="utf-8")
OLD = '''_M49_REFINE_BAND: Tuple[Tuple[int, int, str], ...] = (
    (60, 100, "6"),                              # M50 universal tier (M74: 8 -> 6)
    (100, 10**9, "4"),                           # M49
)'''
NEW = '''# L219/L223 (2026-08-25): the heavy band goes 4 -> 2. M49 derived 4 STRICTLY
# SELECTION-PRESERVING -- it took the part that costs no quality and stopped --
# and the ledger then recorded "do not stack more wall cuts, the floor is
# saturated". That premise is false in the current package: 45 of 100 cases sit
# ABOVE the RF floor and 10 of them carry 86% of the deficit, so cutting REFINE
# there collects real RF. M49's own number was also measured on a tree that has
# since taken L131, L136, L147 and L124.
#
# MEASURED, not assumed:
#   wall     the n>100 profile phase drops 21.5% (single pair) / 27.2% (min-of-3
#            against a +4.1% control band that must read 0, i.e. the estimator's
#            own bias). 81% of the 1020 heavy-band profiles get faster, median
#            ratio 0.801, while the untouched n<=100 control sits at 1.021.
#   quality  IN SET it is +0.0400%, a GAIN, with 2 of 100 cases moving. OUT OF
#            SAMPLE it FLIPS SIGN: -0.0941% (s1) and -0.2635% (s2), mean
#            -0.1788%, both samples the same sign, 0 infeasible in either.
#            Going below a selection-preserving bound changes selections BY
#            CONSTRUCTION, so the in-set null was never going to survive -- but
#            -0.18% against a ~2.3pp RF gain is a trade worth taking.
#   joint    with the L211 pool drop the OOS cost is -0.4996% (both samples) for
#            NET +4.33..+4.53% vs beta, against +1.260% without either.
#
# Kill switch ICCAD_L223_REFINE_HEAVY=<n> restores any value; =4 is pre-L223.
_M49_REFINE_BAND: Tuple[Tuple[int, int, str], ...] = (
    (60, 100, "6"),                              # M50 universal tier (M74: 8 -> 6)
    (100, 10**9, "2"),                           # M49 -> L223 (was "4")
)'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
OLD2 = '''    for lo, hi, iters in _M49_REFINE_BAND:
        if lo < block_count <= hi:
            return {"ICCAD_REFINE_ITERS": iters}
    return {}'''
NEW2 = '''    for lo, hi, iters in _M49_REFINE_BAND:
        if lo < block_count <= hi:
            # L223 kill switch: an explicit value replaces the heavy band, so
            # ICCAD_L223_REFINE_HEAVY=4 reproduces the pre-L223 package exactly.
            _kv = os.environ.get("ICCAD_L223_REFINE_HEAVY", "")
            if _kv and block_count > 100:
                return {"ICCAD_REFINE_ITERS": _kv}
            return {"ICCAD_REFINE_ITERS": iters}
    return {}'''
assert s.count(OLD2) == 1
p.write_text(s.replace(OLD2, NEW2), encoding="utf-8", newline="\n")
print("   _M49_REFINE_BAND heavy band 4 -> 2, kill switch added")
PYX
"$PY" -m py_compile optimizer_constructive.py || exit 1
ICCAD_ADAPTIVE_CORES=48 "$PY" -c "
import sys,os; sys.argv=['x']
import optimizer_constructive as O
b=O._band_env(120); assert b.get('ICCAD_REFINE_ITERS')=='2', b
os.environ['ICCAD_L223_REFINE_HEAVY']='4'
assert O._band_env(120).get('ICCAD_REFINE_ITERS')=='4'
del os.environ['ICCAD_L223_REFINE_HEAVY']
assert O._band_env(80).get('ICCAD_REFINE_ITERS')=='6', 'mid band must not move'
print('   band checks: n=120 -> 2, kill switch -> 4, n=80 -> 6 (unchanged)')
" 2>&1 | grep -v Warning | grep -vE "^\[scipy" || exit 1
"$PY" -u make_submission.py stage 2>&1 | tail -3
echo "   op_wrapper: $(md5sum build_submission/cadc1075/op_wrapper.py | cut -c1-32)"
echo "   tar       : $(md5sum build_submission/cadc1075.tar.gz | cut -c1-32)"
echo L225_STAGED
