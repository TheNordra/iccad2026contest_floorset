# L181 -- redo the pool attribution with ROUTE A OFF.
#
# WHY EVERYTHING BEFORE THIS IS VOID. Route A is ON by default at >=40 DETECTED
# cores (_route_a_default), and every L176/L179 arm ran with ICCAD_ADAPTIVE_CORES=48,
# so route A was live in all of them. It converts each profile into frame tasks on
# a global queue -- designed for 48 REAL cores, where L110/L111 measured wall
# -32.2%. This box has 16 physical cores, so there it is pure oversubscription:
#
#   n= 60   route A on 3.589s   off 1.233s   M73 1.277s
#   n=120   route A on 4.011s   off 2.757s   M73 2.780s     (cost bit-identical)
#
# With route A off the current wrapper matches M73 to within noise. So the
# "1.89x unattributed pool regression" was route A on a small box, and the
# M80 / L124-twin / L137 wall costs measured with it live are inflated too --
# the oversubscription scales with the number of concurrent tasks, so it
# penalises the bigger pools hardest.
#
# These arms isolate the POOL. They are NOT the shipped configuration's runtime:
# nothing on a 16-core box can measure that, because route A's sign flips.
$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$REPO = 'C:\ICCAD_ml\ship_final'
$lock = "$REPO\.l181.lock"
if (Test-Path $lock) { 'ABORT: .l181.lock exists'; exit 1 }
New-Item -ItemType Directory -Path $lock | Out-Null
try {
  Set-Location "$REPO\iccad2026contest"
  $arms = @(
    @{tag='cur';    opt='../optimizer_constructive.py';       env=@{}},
    @{tag='nom80';  opt='../optimizer_constructive.py';       env=@{ICCAD_M80_TIER='0'}},
    @{tag='notwin'; opt='../optimizer_constructive.py';       env=@{ICCAD_M80_TIER='0'; ICCAD_M124_TWIN='0'}},
    @{tag='nohint'; opt='../optimizer_constructive.py';       env=@{ICCAD_M80_TIER='0'; ICCAD_M124_TWIN='0'; ICCAD_HINT_MODE='0'}},
    @{tag='m73';    opt='../_m73win/optimizer_constructive.py'; env=@{}}
  )
  '{0,-8} {1,-10} {2,-12} {3}' -f 'arm','scored_s','weighted','SAfb'
  foreach ($a in $arms) {
    Get-ChildItem env:ICCAD_* -ErrorAction SilentlyContinue | Remove-Item -ErrorAction SilentlyContinue
    $env:ICCAD_ADAPTIVE_CORES = '48'
    $env:ICCAD_SHAPE_LP = '0'
    $env:ICCAD_ROUTE_A = '0'
    foreach ($k in $a.env.Keys) { Set-Item -Path "env:$k" -Value $a.env[$k] }
    $out = "$REPO\_l181_$($a.tag).json"
    & $PY -u iccad2026_evaluate.py --evaluate $a.opt -o $out *> "$REPO\_l181_$($a.tag).log"
    if (Test-Path $out) {
      $j = Get-Content $out -Raw | ConvertFrom-Json
      $sum = ($j.test_results | Measure-Object -Property runtime_seconds -Sum).Sum
      $num = 0.0; $den = 0.0
      foreach ($r in $j.test_results) {
        $w = [Math]::Exp($r.block_count / 12.0); $num += $w * $r.cost; $den += $w
      }
      $fb = (Select-String -Path "$REPO\_l181_$($a.tag).log" -Pattern 'SA fallback' -AllMatches).Matches.Count
      '{0,-8} {1,-10:N2} {2,-12:N6} {3}' -f $a.tag, $sum, ($num/$den), $fb
    } else { '{0,-8} MISSING' -f $a.tag }
  }
  Get-ChildItem env:ICCAD_* -ErrorAction SilentlyContinue | Remove-Item -ErrorAction SilentlyContinue
  'L181_POOLONLY_DONE'
} finally { Remove-Item -Recurse -Force $lock -ErrorAction SilentlyContinue }
