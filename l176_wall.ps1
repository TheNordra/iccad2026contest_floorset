# L176 -- the clean, calibration-free pool-wall measurement. Box is exclusive.
#
# Supersedes the L173 paired runs, which were quarantined: a second agent's
# l171_gates.sh was running 100-case evaluations from 11:05 onward and those
# arms were not affinity-confined, so they shared all 16 physical cores with it.
#
# Three arms, sequential, LP OFF in all three so this is a POOL measurement:
#   cur     current tree, 51 profiles at n>100
#   m73     git 7f38893's wrapper, 35 profiles   -- the package the graders ran
#   nom80   current tree with ICCAD_M80_TIER=0, 43 profiles
#
# f never appears: the transfer is a RATIO on one box,
#     t_current_grader(n) ~= t_beta(n) * w_cur(n)/w_m73(n) * k
# and (w_cur - w_nom80)/8 is the marginal wall of ONE M80 knob profile, which
# turns K into a priceable knob against the OOS quality curve already committed
# in results_M80_oos_s{1,2}_c48.json.
#
# The depth map and L171's hb predictor are both LP-side, so ICCAD_SHAPE_LP=0
# makes all three arms independent of which of them is in the tree.
$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$REPO = 'C:\ICCAD_ml\ship_final'
$lock = "$REPO\.l176.lock"
if (Test-Path $lock) { 'ABORT: .l176.lock exists'; exit 1 }
New-Item -ItemType Directory -Path $lock | Out-Null
try {
  $env:ICCAD_ADAPTIVE_CORES = '48'
  $env:ICCAD_SHAPE_LP = '0'
  Set-Location "$REPO\iccad2026contest"
  foreach ($arm in @(@('cur','../optimizer_constructive.py',$null),
                     @('m73','../_m73win/optimizer_constructive.py',$null),
                     @('nom80','../optimizer_constructive.py','0'))) {
    $tag = $arm[0]; $opt = $arm[1]
    if ($arm[2]) { $env:ICCAD_M80_TIER = $arm[2] } else { $env:ICCAD_M80_TIER = $null }
    $t0 = Get-Date
    & $PY -u iccad2026_evaluate.py --evaluate $opt -o "$REPO\_l176_$tag.json" `
        *> "$REPO\_l176_$tag.log"
    $el = (Get-Date) - $t0
    if (Test-Path "$REPO\_l176_$tag.json") {
      $j = Get-Content "$REPO\_l176_$tag.json" -Raw | ConvertFrom-Json
      $sum = ($j.test_results | Measure-Object -Property runtime_seconds -Sum).Sum
      $fb = (Select-String -Path "$REPO\_l176_$tag.log" -Pattern 'SA fallback' -AllMatches).Matches.Count
      '{0,-6} {1} cases  scored {2,8:N2}s  elapsed {3,4:N0}s  SAfallback={4}' -f `
          $tag, $j.test_results.Count, $sum, $el.TotalSeconds, $fb
    } else { '{0,-6} MISSING' -f $tag }
  }
  $env:ICCAD_M80_TIER = $null
  'L176_WALL_DONE'
} finally {
  Remove-Item -Recurse -Force $lock -ErrorAction SilentlyContinue
}
