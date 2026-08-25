#!/bin/sh
# L245 -- the NO-VENDOR standby package.
#
# Built by REPACKING the shipped stage without vendor/, never by re-staging, so
# the six graded files stay byte-identical to the verified package by
# construction rather than by re-running a gate on them.
#
# 🚨 DO NOT UPLOAD THIS unless the organizers answer that vendor/ violates the
# "unused large binary" rule. Measured cost if scipy is absent on the grader:
# NET +5.224% -> +2.875%, graded 0.87819 -> 0.89995, rank 2 -> rank 4. If scipy
# IS present the two packages are bit-identical, so this is not a quality
# trade-off, it is a two-rank bet on the environment.
set -u
R=/c/ICCAD_ml/ship_final
D=$R/build_submission.NOVENDOR
cd "$R" || exit 1
rm -rf "$D"; mkdir -p "$D"
cp -r build_submission/cadc1075 "$D/cadc1075"
rm -rf "$D/cadc1075/vendor"
( cd "$D" && tar czf cadc1075.tar.gz cadc1075 )
echo "=== 六個計分檔必須與出貨包逐位相同 ==="
FAIL=0
for f in op_wrapper.py op_src.py requirements.txt README.md constructive.cpp bin/constructive_linux; do
  a=$(md5sum "build_submission/cadc1075/$f" | cut -c1-32)
  b=$(md5sum "$D/cadc1075/$f" | cut -c1-32)
  if [ "$a" = "$b" ]; then s="OK  "; else s="!!!!"; FAIL=1; fi
  printf "%s %-26s %s\n" "$s" "$f" "$a"
done
echo "=== vendor 必須不在 ==="
V=$(tar tzf "$D/cadc1075.tar.gz" | grep -c "vendor/")
N=$(tar tzf "$D/cadc1075.tar.gz" | wc -l)
echo "vendor 成員數 $V (必須 0)   總成員 $N (必須 8)"
[ "$V" -eq 0 ] && [ "$N" -eq 8 ] || FAIL=1
ls -l "$D/cadc1075.tar.gz" | awk '{print "tar 大小: "$5" bytes"}'
[ "$FAIL" -eq 0 ] && echo "L245_OK" || echo "L245_FAIL"
