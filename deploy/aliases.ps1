# Short commands for the hub counter, Watchtower and the fuel models, in
# PowerShell (Windows). The same names as deploy/aliases.sh.
#
#   . .\deploy\aliases.ps1                  # this window only
#   python run.py aliases --install         # every new window (adds one line to your $PROFILE)
#
# Then `frc` lists them.

$script:FrcRoot = if ($env:FRC_HOME) { $env:FRC_HOME } else { Split-Path -Parent $PSScriptRoot }

function script:FrcPy {
  foreach ($p in @("$FrcRoot\.venv-train\Scripts\python.exe", "$FrcRoot\.venv\Scripts\python.exe")) {
    if (Test-Path $p) { return $p }
  }
  return "python"
}
function script:FrcRun { Push-Location $FrcRoot; try { & (FrcPy) run.py @args } finally { Pop-Location } }
function script:FrcScript($s) { Push-Location $FrcRoot; try { & (FrcPy) $s @args } finally { Pop-Location } }
function script:Setup($a) { if ($a.Count -and $a[0] -notlike '-*') { $a[0] } else { Join-Path $PWD 'cams.json' } }
function script:Rest($a) { if ($a.Count -and $a[0] -notlike '-*') { $a | Select-Object -Skip 1 } else { $a } }

# hub counter
function hc        { FrcRun hubgui --setup (Setup $args) @(Rest $args) }
function hc-share  { FrcRun hubgui --share --setup (Setup $args) @(Rest $args) }
function hc-feed   { FrcRun hubfeed --setup (Setup $args) @(Rest $args) }
function hc-count  { FrcRun hubcount @args }
function hc-listen { FrcRun hubfeed-listen @args }
function hc-board  { $p = if ($args.Count) { $args[0] } else { 8790 }; Start-Process "http://127.0.0.1:$p/board" }

# Watchtower: the installed app if there is one, else this checkout
function wt {
  $app = "$env:LOCALAPPDATA\Programs\Watchtower\Watchtower.exe"
  if (Test-Path $app) { & $app @args }
  else { $env:PYTHONPATH = $FrcRoot; FrcScript apps\watchtower\main.py @args }
}
function wt-cli  { wt --no-window @args }
function wt-open { $p = if ($args.Count) { $args[0] } else { 8000 }; $pg = if ($args.Count -gt 1) { $args[1] } else { "control" }; Start-Process "http://localhost:$p/$pg" }
function wt-logs { Get-Content -Wait -Tail 50 "$HOME\Documents\Watchtower\watchtower.log" }
function wt-pins { Select-String -Path "$HOME\Documents\Watchtower\config\event.yaml" -Pattern 'pins:' -Context 0,5 }

# fuel models
function fuel-export  { FrcScript train\export_onnx.py @args }
function fuel-compare { FrcScript train\compare_detectors.py @args }
function fuel-count   { FrcScript train\count_einstein.py @args }

# this repo
function frc-test   { FrcScript tests\test_pipeline.py @args }
function frc-update { Push-Location $FrcRoot; try { git pull --ff-only } finally { Pop-Location } }

function frc {
@"
hub counter
  hc [cams.json]           the set-up page (cameras, outlines, drop test, Start)
  hc-share [cams.json]     the same, also on phones over Wi-Fi behind a PIN
  hc-feed [cams.json] ...  count and send to the field with no page (--target HOST:PORT)
  hc-count VIDEO --setup cams.json   count a recording
  hc-listen                stand in for bioarena and print what arrives
  hc-board [port]          the full-screen scoreboard
Watchtower
  wt / wt-cli              start it (app window / browser), --no-vision to score by hand
  wt-open [port] [page]    open control, ref, emcee, queue, display or overlay
  wt-pins / wt-logs        the PINs, and the live log
fuel models
  fuel-export MODEL.pt     .onnx for the apps (refused unless boxes match)
  fuel-compare / fuel-count   score tables / Einstein 1 vs the scoreboard
repo
  frc-test / frc-update    run the tests / pull the latest
"@
}
