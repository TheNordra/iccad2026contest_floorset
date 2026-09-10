# 跑一個 arm 的官方 eval。刻意用 Start-Process 分開導 stdout/stderr，
# 避免 PS 5.1 把 native exe 的 stderr 包成 NativeCommandError。
param(
  [Parameter(Mandatory = $true)][string]$Arm,
  [string]$Tag = "",
  [string]$Repo = "C:\Users\Nordra\Downloads\ICCAD2026_FloorSet\FloorSet",
  [string]$Python = "C:\Users\Nordra\.conda\envs\iccadv\python.exe",
  # 量 LP 成本一定要用 CPU time（ICCAD_LP_TIMING），不可用整輪 wall 差：
  # 本機 wall 噪聲 >=20%，曾把「做更多工作」的 k=2 量得比 k=1 還快。
  [switch]$Timing,
  # LP 開在全部 100 案（pre-L196 行為）。用來量每一個 block count 的 LP 成本。
  [switch]$GateOff
)

if ($Tag -eq "") { $Tag = $Arm }
$dir = Join-Path $Repo "vk\$Arm\cadc1075"
if (-not (Test-Path $dir)) { Write-Output "MISSING ARM DIR: $dir"; exit 2 }

# 🚨 剝掉所有 ICCAD_*。殘留一個就會量到不是出貨組態的東西——
#    這是本專案記錄最多次的無聲失效。
Get-ChildItem Env: | Where-Object { $_.Name -like 'ICCAD_*' } |
  ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:ICCAD_ADAPTIVE_CORES = '48'
$stats = Join-Path $dir "lp_stats_$Tag.txt"
if (Test-Path $stats) { Remove-Item $stats }
$env:ICCAD_SHAPE_LP_STATS = $stats
if ($Timing) { $env:ICCAD_LP_TIMING = '1' }
if ($GateOff) { $env:ICCAD_LP_GATE = '0' }

$out = Join-Path $Repo "vk\$Tag.out.log"
$err = Join-Path $Repo "vk\$Tag.err.log"
$res = "results_$Tag.json"

Write-Output "ARM=$Arm TAG=$Tag"
Write-Output "  dir    : $dir"
Write-Output "  cores  : forced 48 (pool shape); local harness forces RF=1.0"
Write-Output "  started: $(Get-Date -Format 'HH:mm:ss')"

$p = Start-Process -FilePath $Python `
  -ArgumentList @("-u", "iccad2026_evaluate.py", "--evaluate", "op_wrapper.py",
                  "-d", $Repo, "-o", $res) `
  -WorkingDirectory $dir -NoNewWindow -Wait -PassThru `
  -RedirectStandardOutput $out -RedirectStandardError $err

$n = 0
if (Test-Path $stats) { $n = (Get-Content $stats | Measure-Object -Line).Lines }
Write-Output "  ended  : $(Get-Date -Format 'HH:mm:ss')"
Write-Output "  exit   : $($p.ExitCode)"
Write-Output "  lp_stats lines: $n"
Write-Output "  results: $dir\$res"
exit $p.ExitCode
