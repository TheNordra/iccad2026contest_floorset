#!/usr/bin/env bash
# L160 -- back out f from beta by running the EXACT beta package (M73, 7f38893)
# here and comparing per-n against beta's own grader runtimes.
#
# WHY IT WORKS: CLAUDE.md records that at >=24 cores the wall is MAX-SETTER
# bound (c* = sum dt / max dt is p50 19.3, max 22.5). The grader has 48 cores,
# this WSL has 32 -- both above 22.5 -- so on BOTH the wall is the single
# slowest profile, a SINGLE-THREADED quantity. The whole-case ratio for an
# IDENTICAL package is therefore ~ the single-thread ratio, which is f.
# The earlier 5.0x/5.7x/15.7x numbers were useless not because of parallelism
# but because they compared DIFFERENT packages.
#
# The beta hidden set and our in-set 100 have the IDENTICAL n multiset
# (21..120, one case each), so per-n matching is exact in n.
#
# The evaluator does sys.path.insert(0, parent.parent), so it must run from the
# real iccad2026contest/ to find the loaders and the dataset; only --evaluate
# points at the reconstructed M73.
set -u
V=$HOME/iccadvenv/bin/python
R=/mnt/c/ICCAD_ml/ship_final
echo "== L160  $(date -u +%FT%TZ)  nproc=$(nproc)"
echo "   M73 ELF md5:  $(md5sum $R/_m73probe/bin/constructive_linux | cut -c1-32)"
echo "   _shape_lp in M73: $(grep -c _shape_lp $R/_m73probe/optimizer_constructive.py) (must be 0)"
chmod +x "$R/_m73probe/bin/constructive_linux"
cd "$R/iccad2026contest" || exit 1
env ICCAD_ADAPTIVE_CORES=48 "$V" -u iccad2026_evaluate.py \
    --evaluate ../_m73probe/optimizer_constructive.py \
    -o "$R/results_L160_m73_local.json" > "$R/l160_m73.log" 2>&1
echo "exit=$?"
grep -E "Tests:|Feasible:|Avg Cost|Avg Runtime" "$R/l160_m73.log" | head -5
grep -ciE "fallback|all profiles failed" "$R/l160_m73.log" | sed 's/^/   fallback lines: /'
echo "L160_DONE $(date -u +%FT%TZ)"
