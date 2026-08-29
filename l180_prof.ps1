# L180b -- profile ONE case where the two wrappers ask the placer for identical
# work, and find the ~1.0s/case of Python that M73 does not spend.
#
# n=60 is test_id 39. At n=60 l180_diff.py shows the current tree and M73 select
# the SAME 35 profile indices with IDENTICAL effective env on every one of them,
# yet the case costs 2.429s against M73's 1.061s. The gap is ~1.0s per case and
# almost flat in n (n=21-40 is 3.56x, n=101-120 is 1.54x -- a fixed cost looks
# exactly like that), so over 100 cases it is ~100s, which is essentially the
# whole 98.1s that M80 / the L124 twins / L137 / M71 do not explain.
#
# cProfile adds overhead to both arms equally, so the DIFFERENCE of the two
# tottime tables is the answer even though neither absolute number is clean.
$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\.01\anaconda3\envs\floorset\python.exe'
$REPO = 'C:\ICCAD_ml\ship_final'
$env:ICCAD_ADAPTIVE_CORES = '48'
$env:ICCAD_SHAPE_LP = '0'
$env:ICCAD_M80_TIER = '0'
$env:ICCAD_M124_TWIN = '0'
$env:ICCAD_HINT_MODE = '0'
Set-Location "$REPO\iccad2026contest"
foreach ($arm in @(@('cur','../optimizer_constructive.py'),
                   @('m73','../_m73win/optimizer_constructive.py'))) {
  $tag = $arm[0]
  & $PY -u -m cProfile -o "$REPO\_l180_$tag.prof" iccad2026_evaluate.py `
      --evaluate $arm[1] --test-id 39 -o "$REPO\_l180_$tag.json" `
      *> "$REPO\_l180_$tag.log"
  "profiled $tag"
}
Get-ChildItem env:ICCAD_* | Remove-Item -ErrorAction SilentlyContinue
'L180_PROF_DONE'
