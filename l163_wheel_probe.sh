#!/usr/bin/env bash
# L163 -- can we VENDOR scipy instead of relying on the grader having it?
# PyInstaller is blocked at the build end: this box's WSL is glibc 2.43 and the
# grader is Debian 13 / glibc 2.41, and glibc is not forward compatible, so
# anything linked here fails to exec there. A prebuilt manylinux wheel is built
# by scipy's own maintainers against glibc 2.17, so it has the opposite property.
set -u
D=$HOME/l163_whl
rm -rf "$D"; mkdir -p "$D"
"$HOME/iccadvenv/bin/pip" download scipy --only-binary=:all: --python-version 3.13 \
    --abi cp313 --platform manylinux2014_x86_64 -d "$D" --no-deps -q 2>&1 | tail -2
W=$(ls "$D"/*.whl 2>/dev/null | head -1)
echo "wheel: $(basename "$W")"
"$HOME/iccadvenv/bin/python" - "$W" <<'PY'
import zipfile, sys
z = zipfile.ZipFile(sys.argv[1])
meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
for l in z.read(meta).decode("utf-8", "ignore").splitlines():
    if l.startswith(("Requires-Python", "Requires-Dist")):
        print("   ", l)
c = sum(i.compress_size for i in z.infolist())
u = sum(i.file_size for i in z.infolist())
print(f"    compressed {c/1e6:.1f} MB  ->  unpacked {u/1e6:.1f} MB")
PY
echo "--- glibc symbols the wheel's .so files actually need ---"
mkdir -p "$D/x" && cd "$D/x" && "$HOME/iccadvenv/bin/python" -c "
import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$W"
find . -name '*.so' | head -40 | xargs -r objdump -T 2>/dev/null \
  | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -3
echo "  (target Debian 13 provides GLIBC 2.41)"
