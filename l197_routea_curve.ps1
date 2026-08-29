# L197 -- route A's cost as a FUNCTION OF REAL CORES.
#
# Route A has never run on the grader (beta was M73, which lacks it) and its
# -32.2% at 48 cores is a projection. `ICCAD_ADAPTIVE_CORES=48` does not help:
# it only makes _effective_cores() RETURN 48, selecting the 48-core pool and
# switching the cores-gated tiers on. Throughput stays whatever the box has.
#
# What CAN be measured is the trend. Route A converts IDLE cores into wall
# reduction, so its on/off ratio should fall as cores rise. Measure that ratio
# at 4, 8 and 16 REAL cores (affinity, inherited by every constructive.exe
# child) and see where it is heading.
#
# ⚠️ This extrapolates from a regime where route A is purely harmful (16 cores,
# 51 profiles) to one where it is supposed to help (48). A monotone trend
# toward 1.0 is suggestive, not proof -- but right now there is NO evidence at
# all, and a measured trend beats a projection.
#
# ICCAD_ADAPTIVE_CORES stays 48 in every run: the CONFIGURATION under test is
# the shipped one. Only the affinity changes.
$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$REPO = 'C:\ICCAD_ml\ship_final'
$env:ICCAD_ADAPTIVE_CORES = '48'
$env:ICCAD_SHAPE_LP = '0'
$self = Get-Process -Id $PID
$orig = $self.ProcessorAffinity
Set-Location "$REPO\iccad2026contest"
'{0,-6} {1,-6} {2,-10} {3,-10}' -f 'cores','case','routeA_ON','routeA_OFF'
foreach ($pair in @(@(4,0xF), @(8,0xFF), @(16,0xFFFF))) {
  $C = $pair[0]; $mask = $pair[1]
  $self.ProcessorAffinity = [IntPtr]$mask
  foreach ($case in 99, 93) {
    $t = @{}
    foreach ($ra in '1', '0') {
      if ($ra -eq '0') { $env:ICCAD_ROUTE_A = '0' } else { $env:ICCAD_ROUTE_A = $null }
      $out = "$REPO\_l197_${C}_${case}_${ra}.json"
      & $PY -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py `
          --test-id $case -o $out *> "$REPO\_l197_${C}_${case}_${ra}.log"
      if (Test-Path $out) {
        $j = Get-Content $out -Raw | ConvertFrom-Json
        $t[$ra] = $j.test_results[0].runtime_seconds
      } else { $t[$ra] = [double]::NaN }
    }
    $env:ICCAD_ROUTE_A = $null
    '{0,-6} {1,-6} {2,-10:N3} {3,-10:N3}  ratio {4:N3}' -f `
        $C, $case, $t['1'], $t['0'], ($t['1'] / $t['0'])
  }
}
$self.ProcessorAffinity = $orig
'L197_ROUTEA_DONE'
