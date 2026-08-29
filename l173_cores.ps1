# L173b -- SPLIT the per-case wall into the part that scales with cores and the
# part that does not. This decides whether the 6.2x wall regression against M73
# transfers to a 48-core grader.
#
# l173_attrib.sh established this box is SUM-BOUND, not max-setter bound:
# dropping the 8 M80 profiles (51 -> 43, none of them the max-setter) moved
# n=120 from 8.433s to 6.497s. Under max-setter binding that is a no-op.
#
# So wall(C) = a + b/C, with `a` the core-INDEPENDENT part (the M47 serial
# _proxy_metrics tail, GIL-bound, plus wrapper and harness) and b/C the pool.
#   a    transfers to the grader in full, on any core count.
#   b/C  is what a bigger box buys back.
#
# Affinity is set on THIS PowerShell process before launching, so python and
# every constructive.exe child INHERITS it -- no race, unlike setting it after
# Start-Process. (`cmd /c start /affinity` returns Access Denied here.)
#
# ICCAD_ADAPTIVE_CORES stays 48: that selects the SHIPPED tier configuration,
# and we are measuring that configuration on fewer real cores, not a different
# configuration.
$ErrorActionPreference = 'Continue'
$PY = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$R  = 'C:\ICCAD_ml\ship_final'
$env:ICCAD_ADAPTIVE_CORES = '48'
$env:ICCAD_SHAPE_LP = '0'
$self = Get-Process -Id $PID
$orig = $self.ProcessorAffinity
Set-Location "$R\iccad2026contest"
'{0,-6} {1,-6} {2,-10} {3}' -f 'cores','case','runtime_s','n'
foreach ($pair in @(@(4,0xF), @(8,0xFF), @(16,0xFFFF), @(32,0xFFFFFFFF))) {
  $C = $pair[0]; $mask = $pair[1]
  $self.ProcessorAffinity = [IntPtr]$mask
  foreach ($case in 99, 93) {
    $out = "$R\_l173c_${C}_${case}.json"
    & $PY -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py `
        --test-id $case -o $out *> "$R\_l173c_${C}_${case}.log"
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
'L173_CORES_DONE'
