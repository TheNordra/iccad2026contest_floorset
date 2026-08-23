#!/usr/bin/env bash
# L163b -- build cadc1075/vendor/ from the OFFICIAL scipy manylinux wheel.
# Not built here: this box's glibc is 2.43 and the grader's is 2.41, and glibc
# is not forward compatible, so anything we link fails to exec there. The
# manylinux wheel is built by scipy's maintainers against glibc 2.17 and has
# the opposite property -- its .so files need at most GLIBC_2.14.
set -eu
R=/mnt/c/ICCAD_ml/ship_final
V=$R/vendor
S=$R/vendor_src
rm -rf "$V" "$S"; mkdir -p "$V" "$S"
"$HOME/iccadvenv/bin/pip" download scipy --only-binary=:all: \
    --python-version 3.13 --abi cp313 --platform manylinux2014_x86_64 \
    -d "$S" --no-deps -q
W=$(ls "$S"/*.whl | head -1)
echo "wheel:  $(basename "$W")"
echo "sha256: $(sha256sum "$W" | cut -d' ' -f1)"
"$HOME/iccadvenv/bin/python" -c "
import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$W" "$V"
sha256sum "$W" | cut -d' ' -f1 > "$V/../vendor_wheel.sha256"
basename "$W" >> "$R/vendor_wheel.sha256"
echo "unpacked: $(du -sh "$V" | cut -f1), $(find "$V" -type f | wc -l) files"
echo "top level: $(ls "$V")"
echo "max glibc symbol needed: $(find "$V" -name '*.so*' | xargs -r objdump -T 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)"
