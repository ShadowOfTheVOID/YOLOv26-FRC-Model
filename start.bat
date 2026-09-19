@echo off
rem
rem  TBACroppedOutVid -- double-click this file to harvest matches.
rem
rem  Everything you would normally want to change is in the block below. Open
rem  this file in Notepad, edit, save, double-click again. Lines starting with
rem  "rem" are notes and are ignored.
rem
rem  First run only: it makes a .venv, installs the Python packages, and asks
rem  for your TBA key once. After that it just runs.
rem
rem --------------------------------------------------------------- settings --

rem What to do. "pull" is the full run: pick matches, download, crop, export
rem frames. Other useful values:
rem   menu    ask each time instead of deciding here
rem   fetch   download and crop, but do not export frames
rem   status  ledger and dataset summary, downloads nothing
rem   verify  check the OCR'd fuel totals against TBA's official scores
rem   serve   the read-only scouting API on http://127.0.0.1:8781
rem   doctor  check the install and print what is where, run nothing
set "TBAVID_MODE=pull"

rem How many never-pulled matches to fetch. Each one costs roughly 330 MB of
rem disk and a few minutes. 5 is a comfortable first run; 20 is an evening.
set "TBAVID_COUNT=5"

rem Most videos from any one event. 0 means no limit. A cap of 2 spreads the
rem harvest across events, which is better training data than 20 matches from
rem one field.
set "TBAVID_PER_EVENT_CAP=0"

rem Splitting the work with teammates: you take slice N of M, e.g. "2/4". No
rem two people with different N can ever pull the same match. Empty = anything.
set "TBAVID_SHARD="

rem 1 = show what would be pulled and download nothing. A sanity check.
set "TBAVID_DRY_RUN=0"

rem 1 = open the shot review UI in a browser before frames are exported.
set "TBAVID_REVIEW=0"

rem ----------------------------------------------------------- if it breaks --

rem 1 = update yt-dlp before running. Try this first when downloads start
rem failing for no apparent reason -- YouTube changes and a stale yt-dlp stops
rem working. It is the single most common cause of "it used to work".
set "TBAVID_UPDATE_YTDLP=0"

rem 1 = create .venv and install missing Python packages automatically.
rem 0 = never touch the disk; complain instead.
set "TBAVID_AUTO_INSTALL=1"

rem Anything else, appended to the command verbatim. Examples:
rem   --retry-failed             reconsider matches that failed before
rem   --seed 7                   reproducible picks
rem   --include-noncompetitive   also offseason events and practice matches
set "TBAVID_EXTRA="

rem 1 = stop Windows sleeping while a long harvest runs. This changes a power
rem setting for the whole machine and puts it back afterwards.
set "KEEP_AWAKE=1"

rem 1 = leave this window open at the end so you can read the output.
set "KEEP_OPEN=1"

rem ------------------------------------------------------ nothing below here --

setlocal enabledelayedexpansion
cd /d "%~dp0"

rem The py launcher is what a python.org install always provides; a Microsoft
rem Store install may only have python.exe on PATH.
set "PY="
py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
if not defined PY (
  python -c "import sys" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo No Python found.
  echo   Install it from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" during setup.
  if "%KEEP_OPEN%"=="1" pause
  exit /b 1
)

rem Keep the display off but the machine awake, then restore. powercfg is not
rem available on every edition, so a failure here is not fatal.
set "SLEPT="
if "%KEEP_AWAKE%"=="1" (
  powercfg /change standby-timeout-ac 0 >nul 2>&1 && set "SLEPT=1"
)

%PY% deploy\launcher.py %*
set "STATUS=%ERRORLEVEL%"

if defined SLEPT powercfg /change standby-timeout-ac 30 >nul 2>&1

if "%KEEP_OPEN%"=="1" (
  echo.
  echo Done ^(exit %STATUS%^).
  pause
)
exit /b %STATUS%
