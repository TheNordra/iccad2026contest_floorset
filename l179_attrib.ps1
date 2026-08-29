# L179 -- attribute the 3.95x pool-wall regression to individual mechanisms.
#
# L176: current tree 437.59s vs M73 110.73s on this box, LP off, 100 cases.
# Turning M80 off gets to 237.18s, still 2.14x M73. So M80 is not all of it.
#
# Each arm removes one more post-M73 pool mechanism. LP OFF throughout, so this
# is purely the pool; the LP is priced separately and is only 7.7 grader-seconds.
#
#   cur        (already have, 437.59s)          51 profiles
#   nom80      ICCAD_M80_TIER=0                 43            (already have, 237.18s)
#   notwin     + ICCAD_M124_TWIN=0              the L124 R3 MIB twins
#   nohint     + ICCAD_HINT_MODE=0              L137's GORDIAN hint overlay
#   nom71      + ICCAD_M71=0                    the M71 cluster knobs
#   m73        (already have, 110.73s)          35 profiles
#
# WHY L137 IS A PRIME SUSPECT. `_l137_env()` injects ICCAD_HINT_MODE=1 and
# ICCAD_HINT_REFINE=4 into EVERY profile at >=40 cores, and its own comment says
# the +0.46% wall was measured at 48c and that "below the gate the pool is
# sum-bound, where the hint's extra refine passes are NOT absorbed by a
# max-setter, and nothing has measured it there". L173 measured the pool as
# sum-bound AT 48 forced cores, so that is the regime nothing had measured.
#
# Quality is reported alongside the wall for every arm, because the decision is
# a trade and a wall number alone cannot make it.
$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$REPO = 'C:\ICCAD_ml\ship_final'
$lock = "$REPO\.l179.lock"
if (Test-Path $lock) { 'ABORT: .l179.lock exists'; exit 1 }
New-Item -ItemType Directory -Path $lock | Out-Null
try {
  Set-Location "$REPO\iccad2026contest"
  $arms = @(
    @{tag='notwin'; env=@{ICCAD_M80_TIER='0'; ICCAD_M124_TWIN='0'}},
    @{tag='nohint'; env=@{ICCAD_M80_TIER='0'; ICCAD_M124_TWIN='0'; ICCAD_HINT_MODE='0'}},
    @{tag='nom71';  env=@{ICCAD_M80_TIER='0'; ICCAD_M124_TWIN='0'; ICCAD_HINT_MODE='0'; ICCAD_M71='0'}}
  )
  '{0,-8} {1,-10} {2,-12} {3}' -f 'arm','scored_s','weighted','SAfb'
  foreach ($a in $arms) {
    Get-ChildItem env:ICCAD_* -ErrorAction SilentlyContinue | Remove-Item -ErrorAction SilentlyContinue
    $env:ICCAD_ADAPTIVE_CORES = '48'
    $env:ICCAD_SHAPE_LP = '0'
    foreach ($k in $a.env.Keys) { Set-Item -Path "env:$k" -Value $a.env[$k] }
    $out = "$REPO\_l179_$($a.tag).json"
    & $PY -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py `
        -o $out *> "$REPO\_l179_$($a.tag).log"
    if (Test-Path $out) {
      $j = Get-Content $out -Raw | ConvertFrom-Json
      $sum = ($j.test_results | Measure-Object -Property runtime_seconds -Sum).Sum
      $num = 0.0; $den = 0.0
      foreach ($r in $j.test_results) {
        $w = [Math]::Exp($r.block_count / 12.0)
        $num += $w * $r.cost; $den += $w
      }
      $fb = (Select-String -Path "$REPO\_l179_$($a.tag).log" -Pattern 'SA fallback' -AllMatches).Matches.Count
      '{0,-8} {1,-10:N2} {2,-12:N6} {3}' -f $a.tag, $sum, ($num/$den), $fb
    } else { '{0,-8} MISSING' -f $a.tag }
  }
  Get-ChildItem env:ICCAD_* -ErrorAction SilentlyContinue | Remove-Item -ErrorAction SilentlyContinue
  'L179_ATTRIB_DONE'
} finally {
  Remove-Item -Recurse -Force $lock -ErrorAction SilentlyContinue
}
