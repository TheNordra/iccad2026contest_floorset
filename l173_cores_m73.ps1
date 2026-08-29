# L173e -- the SAME core scan, on an M73-equivalent wrapper, on the SAME box.
#
# WHY. Projecting the current tree's grader runtime through `f` requires
# knowing f for THIS configuration, and f (3.17) was calibrated on M73's WSL
# run. WSL turns out to be 3.0-4.1x slower than Windows for an identical
# configuration (n=120: 27.485s vs 8.433s), so f cannot be reused across boxes.
#
# The calibration-free quantity is the RATIO of the current tree to M73 on ONE
# box at ONE core count. Multiply t_beta by it and f cancels:
#
#     t_current_grader(n)  ~=  t_beta(n) * wall_current(C,n) / wall_M73(C,n)
#
# M73's wrapper is `git show 7f38893:optimizer_constructive.py` -- 1124 lines,
# `_shape_lp` zero times, no `_M80_EXTRA`. It is paired with TODAY's
# constructive.exe, not the era's: the compilers on this box are broken, and
# the C++ delta since (L136's FRAME_EPS fix) changes placement, not the amount
# of work per profile. That is the one approximation in this measurement and it
# affects the pool's per-profile cost only marginally.
$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$REPO = 'C:\ICCAD_ml\ship_final'
$env:ICCAD_ADAPTIVE_CORES = '48'
$env:ICCAD_SHAPE_LP = '0'
$self = Get-Process -Id $PID
$orig = $self.ProcessorAffinity
Set-Location "$REPO\iccad2026contest"
'{0,-6} {1,-6} {2,-10} {3}' -f 'cores','case','runtime_s','n'
foreach ($pair in @(@(4,0xF), @(8,0xFF), @(16,0xFFFF))) {
  $C = $pair[0]; $mask = $pair[1]
  $self.ProcessorAffinity = [IntPtr]$mask
  foreach ($case in 99, 93) {
    $out = "$REPO\_l173m_${C}_${case}.json"
    & $PY -u iccad2026_evaluate.py --evaluate ../_m73win/optimizer_constructive.py `
        --test-id $case -o $out *> "$REPO\_l173m_${C}_${case}.log"
    if (Test-Path $out) {
      $j = Get-Content $out -Raw | ConvertFrom-Json
      $rec = $j.test_results[0]
      '{0,-6} {1,-6} {2,-10:N3} {3}' -f $C, $case, $rec.runtime_seconds, $rec.block_count
    } else {
      '{0,-6} {1,-6} MISSING' -f $C, $case
    }
  }
}
$self.ProcessorAffinity = $orig
'L173_M73_DONE'
