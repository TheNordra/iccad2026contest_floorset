#!/usr/bin/env bash
# L168 -- can WSL produce an ELF the grader can actually exec? The box is
# glibc 2.43 and Debian 13 is 2.41; glibc is not forward compatible. The
# shipped bin/constructive_linux needs at most GLIBC_2.34, so it was NOT built
# here -- or it was built with something that caps the requirement. Find out,
# because L152 needs a rebuilt ELF and PyInstaller already died on this.
set -u
R=/mnt/c/ICCAD_ml/ship_final
O=$HOME/l168_probe.out
echo "build box glibc: $(ldd --version | head -1 | awk '{print $NF}')   target: 2.41"
echo "shipped ELF needs: $(objdump -T $R/bin/constructive_linux | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)"
g++ -O3 -std=c++17 -static-libstdc++ -static-libgcc -o "$O" "$R/constructive.cpp" 2>&1 | tail -3
if [ -x "$O" ]; then
  echo "freshly built needs: $(objdump -T "$O" | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)"
  echo "all symbols above 2.34: $(objdump -T "$O" | grep -o 'GLIBC_[0-9.]*' | sort -Vu | awk -F_ '$2+0>2.34' | tr '\n' ' ')"
else
  echo "BUILD FAILED"
fi
