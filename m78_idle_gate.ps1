# M78 idle hard gate -- Route A step 0 (NEVER ship).
#
# HANDOFF_2026-08-02 section 3-A says: prove the box is idle BEFORE measuring
# runtime, and do NOT use Get-Counter for total CPU (instantaneous value +
# the sampler's own activity => badly misleading).  The prescribed method is
# per-process TotalProcessorTime differencing over a 30-60s window.
#
# PRE-REGISTERED CRITERIA (written before the first run, do not move them):
#   window          = 60 s
#   threshold       = 2% of ONE core over the window = 1.20 CPU-seconds
#   PASS  iff  no process other than this script's own PID accumulates more
#              than the threshold in the window
#   codex / python / node / constructive are called out explicitly because
#   those are the ones that would silently poison an RF measurement.
#   A FAIL means REPORT "box is not idle" -- it does NOT mean lower the bar
#   and measure anyway.  (M64/M65 discipline: never retro-move a threshold.)
#
# Exit code 0 = PASS, 1 = FAIL.  stdout is the evidence record.

$ErrorActionPreference = "Stop"
$WINDOW   = 60
$PCT      = 0.02
$THRESH   = $WINDOW * $PCT          # CPU-seconds = 2% of one core
$SELF     = $PID

function Snap {
    $h = @{}
    foreach ($p in Get-Process -ErrorAction SilentlyContinue) {
        try { $c = $p.CPU } catch { $c = $null }
        if ($null -ne $c) { $h["$($p.Id)|$($p.ProcessName)"] = [double]$c }
    }
    return $h
}

$cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Output "M78 idle gate  |  logical cores $cores  |  window ${WINDOW}s  |  threshold $THRESH CPU-s (= $($PCT*100)% of one core)"
Write-Output "self PID $SELF (excluded)"
Write-Output ""

$t0 = Get-Date
$a  = Snap
Start-Sleep -Seconds $WINDOW
$b  = Snap
$elapsed = ((Get-Date) - $t0).TotalSeconds

$rows = @()
foreach ($k in $b.Keys) {
    $id, $name = $k.Split("|")
    # a process absent at t0 is new: charge its whole CPU time (conservative)
    $d = if ($a.ContainsKey($k)) { $b[$k] - $a[$k] } else { $b[$k] }
    if ($d -gt 0.005) {
        $rows += [pscustomobject]@{
            Pid = [int]$id; Name = $name; CpuSec = [math]::Round($d, 3)
            PctCore = [math]::Round(100.0 * $d / $elapsed, 2)
            Self = ([int]$id -eq $SELF)
        }
    }
}
$rows = $rows | Sort-Object CpuSec -Descending

Write-Output ("actual window {0:N1}s   processes accumulating CPU: {1}" -f $elapsed, $rows.Count)
Write-Output ""
Write-Output ("{0,-8} {1,-32} {2,10} {3,9}  {4}" -f "PID", "Name", "CPU-s", "%1core", "note")
foreach ($r in ($rows | Select-Object -First 20)) {
    $note = if ($r.Self) { "<- this gate (excluded)" }
            elseif ($r.CpuSec -gt $THRESH) { "*** OVER THRESHOLD ***" } else { "" }
    Write-Output ("{0,-8} {1,-32} {2,10} {3,8}%  {4}" -f $r.Pid, $r.Name, $r.CpuSec, $r.PctCore, $note)
}

$total = ($rows | Measure-Object CpuSec -Sum).Sum
Write-Output ""
Write-Output ("total CPU consumed by all processes: {0:N2} CPU-s of {1:N0} available ({2:N2}% of the box)" -f `
    $total, ($elapsed * $cores), (100.0 * $total / ($elapsed * $cores)))

# explicit named check, per the handoff
$watch = $rows | Where-Object { -not $_.Self -and $_.Name -match "codex|python|node|constructive|g\+\+" }
Write-Output ""
if ($watch) {
    Write-Output "named-watch processes (codex/python/node/constructive/g++) that moved:"
    foreach ($r in $watch) { Write-Output ("  {0,-24} pid {1,-8} {2} CPU-s" -f $r.Name, $r.Pid, $r.CpuSec) }
} else {
    Write-Output "named-watch processes (codex/python/node/constructive/g++): NONE moved"
}

$offenders = $rows | Where-Object { -not $_.Self -and $_.CpuSec -gt $THRESH }
Write-Output ""
if ($offenders) {
    Write-Output "VERDICT: FAIL -- box is NOT idle. Offenders:"
    foreach ($r in $offenders) { Write-Output ("  {0,-24} pid {1,-8} {2} CPU-s ({3}% of one core)" -f $r.Name, $r.Pid, $r.CpuSec, $r.PctCore) }
    Write-Output ""
    Write-Output "Do NOT measure runtime. Report 'box is not idle' and stop."
    exit 1
} else {
    Write-Output "VERDICT: PASS -- no non-self process exceeded $THRESH CPU-s in the window."
    Write-Output "RF runtime measurement is authorised."
    exit 0
}
