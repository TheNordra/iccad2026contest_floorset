# L175 -- the third full-corpus arm: M80 tier OFF.
#
# With w_cur (51 profiles) and w_nom80 (43) over all 100 cases on one box, the
# marginal wall of ONE M80 profile is (w_cur - w_nom80)/8 PER CASE. That turns
# K -- the number of knob-cloud profiles -- into a priceable knob instead of a
# constant, and results_M80_oos_s{1,2}_c48.json already carry the measured OOS
# quality for every K from 0 to 12:
#
#     K        0      1      2      3      4      5      6      7      8
#     s1    0.000  0.299  1.219  1.453  1.626  1.712  1.775  1.878  2.073
#     s2    0.000  0.101  1.096  1.230  1.465  1.583  1.599  1.671  1.920
#
# K=2 is already 57-59% of the quality K=8 buys, for 25% of the profiles. That
# shape only matters if profiles cost wall, which L173 shows they now do.
#
# Same flags as l173_pair.ps1 so the three arms are directly comparable.
$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$REPO = 'C:\ICCAD_ml\ship_final'
$env:ICCAD_ADAPTIVE_CORES = '48'
$env:ICCAD_SHAPE_LP = '0'
$env:ICCAD_M80_TIER = '0'
Set-Location "$REPO\iccad2026contest"
$t0 = Get-Date
& $PY -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py `
    -o "$REPO\_l173p_nom80.json" *> "$REPO\_l173p_nom80.log"
$el = (Get-Date) - $t0
if (Test-Path "$REPO\_l173p_nom80.json") {
  $j = Get-Content "$REPO\_l173p_nom80.json" -Raw | ConvertFrom-Json
  $sum = ($j.test_results | Measure-Object -Property runtime_seconds -Sum).Sum
  'nom80: {0} cases, scored wall {1:N2}s, elapsed {2:N0}s' -f $j.test_results.Count, $sum, $el.TotalSeconds
} else { 'nom80: MISSING' }
$env:ICCAD_M80_TIER = $null
'L175_NOM80_DONE'
