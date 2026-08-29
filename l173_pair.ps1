# L173f -- the current tree and M73, FULL 100 cases, SAME box, SAME flags.
#
# This is the calibration-free comparison. `f` cancels in the ratio, so no
# cross-box constant is used anywhere:
#
#     t_current_grader(n)  ~=  t_beta(n) * w_current(n) / w_M73(n)
#
# t_beta(n) is M73's runtime as measured BY THE GRADER, so the ratio transports
# our box's arithmetic onto the grader's clock without ever needing to know how
# fast either machine is.
#
# LP off in both: M73 has no shape LP at all (`_shape_lp` appears zero times in
# 7f38893), so leaving it on in the current tree would price the LP into the
# pool ratio. The LP is added back separately, from its own measured seconds.
#
# Cores unrestricted (16 physical + SMT). The 4/8/16 scans in l173_cores.ps1
# supply the a + b/C split needed to carry this ratio from 16 real cores to the
# grader's 48.
$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$REPO = 'C:\ICCAD_ml\ship_final'
$env:ICCAD_ADAPTIVE_CORES = '48'
$env:ICCAD_SHAPE_LP = '0'
Set-Location "$REPO\iccad2026contest"
foreach ($arm in @(@('cur','../optimizer_constructive.py'),
                   @('m73','../_m73win/optimizer_constructive.py'))) {
  $tag = $arm[0]; $opt = $arm[1]
  $t0 = Get-Date
  & $PY -u iccad2026_evaluate.py --evaluate $opt -o "$REPO\_l173p_$tag.json" `
      *> "$REPO\_l173p_$tag.log"
  $el = (Get-Date) - $t0
  if (Test-Path "$REPO\_l173p_$tag.json") {
    $j = Get-Content "$REPO\_l173p_$tag.json" -Raw | ConvertFrom-Json
    $sum = ($j.test_results | Measure-Object -Property runtime_seconds -Sum).Sum
    '{0}: {1} cases, scored wall {2:N2}s, elapsed {3:N0}s' -f $tag, $j.test_results.Count, $sum, $el.TotalSeconds
  } else {
    '{0}: MISSING' -f $tag
  }
}
'L173_PAIR_DONE'
