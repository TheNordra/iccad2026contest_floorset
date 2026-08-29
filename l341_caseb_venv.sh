#!/usr/bin/env bash
# L341 -- the Case B lane the ledger has never run.
#
# WHY IT MATTERS NOW. C_QA_20260827 A25/A27/A29: "Our pipeline now ALWAYS creates a
# venv when requirements.txt is non-empty. We will create a venv and use it."  Our
# requirements.txt is non-empty (7 entries), so the Final run takes the venv path
# with certainty. A17 says environment issues DID break beta submissions and had to
# be re-run. And our BETA package shipped an EMPTY requirements.txt, so the venv path
# has never executed for us -- it is new exposure introduced after beta, not inherited.
#
# WHAT THE EXISTING LANES DO NOT COVER. L313's five lanes all run against a
# pre-existing $HOME/iccadvenv with the libraries already present. That tests the
# CODE on Linux. It does not test whether `pip install -r requirements.txt` resolves
# at all, nor what versions it lands on.
#
# NOTE this is IDENTICAL exposure for D and RF-SAFE -- their requirements.txt files
# are byte-identical (6c59feb4). So this lane cannot gate the RF-SAFE upload; a
# failure here means BOTH packages need a fix.
#
#   wsl -d Ubuntu -- bash /mnt/c/ICCAD_ml/ship_final/l341_caseb_venv.sh
set -u
R=/mnt/c/ICCAD_ml/ship_final
TAR=$R/build_submission.RFSAFE/cadc1075.tar.gz
W=$HOME/l341
export L117_WORK=$W/work

echo "== L341 Case B venv lane  $(date -u +%FT%TZ)  nproc=$(nproc)"
rm -rf "$W"; mkdir -p "$W" "$L117_WORK"
cd "$W" || exit 1

tar xzOf "$TAR" cadc1075/requirements.txt > requirements.txt
echo "-- requirements.txt (md5 $(md5sum requirements.txt | cut -c1-32)):"
sed 's/^/     /' requirements.txt

echo; echo "########## STAGE 1 -- fresh venv, no system site packages ##########"
SYSPY=$(command -v python3)
echo "   base interpreter: $SYSPY  ($($SYSPY -V 2>&1))"
"$SYSPY" -m venv "$W/venv" || { echo "STAGE1_RC=90 venv creation FAILED"; exit 90; }
V=$W/venv/bin/python
"$V" -m pip install --upgrade pip -q
echo "   installing (this is the real test -- fresh resolution, no cache reuse of our env)"
time "$V" -m pip install -r requirements.txt 2>&1 | tail -25
RC=${PIPESTATUS[0]}
echo "STAGE1_PIP_RC=$RC"
[ "$RC" -ne 0 ] && { echo "🚨 STAGE 1 FAILED -- pip could not resolve. BOTH D and RF-SAFE affected."; exit 1; }

echo; echo "-- resolved versions:"
"$V" - <<'PY'
import importlib
for m in ("torch","numpy","scipy","shapely","matplotlib","tqdm","requests"):
    try:
        mod = importlib.import_module(m)
        print("   %-12s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("   %-12s IMPORT FAILED: %s" % (m, type(e).__name__))
import sys; print("   python       %s" % sys.version.split()[0])
PY

echo; echo "########## STAGE 2 -- official eval INSIDE that venv ##########"
# l117_linux_verify.py runs the evaluator with sys.executable, so invoking it with
# the fresh venv's python is exactly the Case B path.
cd "$R" || exit 1
"$V" l117_linux_verify.py final "$TAR" --tag l341_caseb --anchor l300_win32_ship.json
echo "STAGE2_RC=$?"
echo "== L341 done  $(date -u +%FT%TZ)"
