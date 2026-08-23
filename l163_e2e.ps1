$sp = "C:\Users\.01\anaconda3\envs\floorset\Lib\site-packages"
$orig = "$sp\scipy"; $hid = "$sp\scipy__HIDDEN_L163"
try {
  Rename-Item $orig $hid -ErrorAction Stop
  "scipy hidden"
  Set-Location C:\ICCAD_ml\ship_final\iccad2026contest
  $env:ICCAD_ADAPTIVE_CORES = "48"
  & "C:\Users\.01\anaconda3\envs\floorset\python.exe" -u iccad2026_evaluate.py --evaluate ../optimizer_constructive.py -o ../results_L163_vendor.json > ../l163_vendor.log 2>&1
  "eval exit=$LASTEXITCODE"
} finally {
  if (Test-Path $hid) { Rename-Item $hid $orig; "scipy RESTORED" }
}
