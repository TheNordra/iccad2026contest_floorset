#!/usr/bin/env bash
# L244b -- verify variant D the only way that means anything: the OFFICIAL
# command, on Linux, from the tar.
#
# L244's G2/G3 pointed the evaluator at the PACKAGED op_wrapper.py on Windows.
# There `_ensure_compiled` skips the bundled ELF and compiles constructive.cpp
# itself, msys is not on PATH, the compile fails and every case falls back --
# which is why both arms read cost 9.9999. That is the failure mode
# [[windows-msys-path-silent-sa-fallback]] records, reproduced exactly.
#
# Two lanes, both on the D tar:
#   D1  default. Must reproduce the shipped package's Linux number
#       (1.2264069637) -- vendor/ was never being used, so removing it can only
#       be a no-op, and this is where that stops being an argument.
#   D2  scipy BLOCKED via a PYTHONPATH shim that raises on import. This is the
#       floor D accepts if pip somehow does not deliver scipy: the LP turns off,
#       and the question is whether the package degrades or dies.
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
export L117_WORK=$HOME/l244
TAR=$R/build_submission.D/cadc1075.tar.gz
mkdir -p "$L117_WORK" "$HOME/l244_noscipy"
cat > "$HOME/l244_noscipy/scipy.py" <<'PYS'
raise ImportError("L244b: scipy deliberately blocked to measure the floor")
PYS
cd "$R" || exit 1
echo "== L244b  $(date -u +%FT%TZ)  nproc=$(nproc)"
echo "   D tar:          $(md5sum $TAR | cut -c1-32)  $(stat -c%s $TAR) bytes"
echo "   D op_wrapper:   $(tar xzOf $TAR cadc1075/op_wrapper.py | md5sum | cut -c1-32)"
echo "   vendor entries: $(tar tzf $TAR | grep -c 'vendor/')  (must be 0)"
echo "   WSL scipy:      $($V -c 'import scipy;print(scipy.__version__)' 2>&1 | tail -1)"

echo; echo "########## D1 -- 48c, default (scipy available) ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag d_ship --no-judge --stats \
     --base results_L153_lpoff_L137.json
echo "D1_RC=$?"

echo; echo "########## D2 -- 48c, scipy BLOCKED ##########"
"$V" l117_linux_verify.py final48 "$TAR" --tag d_noscipy --no-judge \
     --base results_L153_lpoff_L137.json --env PYTHONPATH=$HOME/l244_noscipy
echo "D2_RC=$?"

echo; echo "########## verdict ##########"
"$V" - <<'PY'
import json, math, os, sys
W = os.environ["L117_WORK"]
def load(tag):
    p = f"{W}/{tag}/cadc1075/results_l117_{tag}.json"
    if not os.path.exists(p):
        return None
    return {r["test_id"]: r for r in json.load(open(p))["test_results"]}
d1, d2 = load("d_ship"), load("d_noscipy")
if not d1 or not d2:
    print("   missing arm"); sys.exit(1)
w = lambda r: math.exp(r["n"] / 12.0) if "n" in r else 1.0
def wq(d):
    s = sum(w(r) for r in d.values())
    return sum(w(r) * r["cost"] for r in d.values()) / s
f1 = sum(1 for r in d1.values() if r.get("is_feasible", r.get("feasible")))
f2 = sum(1 for r in d2.values() if r.get("is_feasible", r.get("feasible")))
print(f"   D1 (scipy on)  weighted cost {wq(d1):.10f}   feasible {f1}/{len(d1)}")
print(f"   D2 (scipy off) weighted cost {wq(d2):.10f}   feasible {f2}/{len(d2)}")
print(f"   the LP is worth {100*(wq(d2)-wq(d1))/wq(d2):+.4f}% here")
print(f"   shipped package on Linux was 1.2264069637 -- D1 must match it")
PY
echo "L244B_DONE $(date -u +%FT%TZ)"
