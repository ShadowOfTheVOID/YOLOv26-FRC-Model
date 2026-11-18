# Short commands for the hub counter, Watchtower and the fuel models, for
# people who would rather type than click. bash or zsh (macOS, Linux, a Pi).
#
#   source deploy/aliases.sh                 # this shell only
#   python3 run.py aliases --install         # every new shell (adds one line to ~/.zshrc or ~/.bashrc)
#
# Then `frc` lists them. Every command takes the same extra options as the
# run.py / train/ script it wraps; `hc-feed --help` and so on show them.
# They run from this checkout and use .venv-train's Python when it exists
# (it has OpenCV and the model runtime), else .venv's, else python3.

_frc_root() {
  if [ -n "${FRC_HOME:-}" ]; then printf '%s' "$FRC_HOME"; return; fi
  if [ -n "${BASH_SOURCE:-}" ]; then (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); return; fi
  printf '%s' "$_FRC_ROOT_ZSH"
}
# zsh has no BASH_SOURCE: remember where this file was sourced from now
[ -n "${ZSH_VERSION:-}" ] && _FRC_ROOT_ZSH="$(cd "$(dirname "${(%):-%x}")/.." && pwd)"

_frc_py() {
  local r; r="$(_frc_root)"
  for p in "$r/.venv-train/bin/python" "$r/.venv/bin/python"; do
    [ -x "$p" ] && { printf '%s' "$p"; return; }
  done
  printf 'python3'
}
_frc_run() { (cd "$(_frc_root)" && "$(_frc_py)" run.py "$@"); }
_frc_open() {
  case "$(uname -s)" in Darwin) open "$1" ;; *) xdg-open "$1" >/dev/null 2>&1 || echo "open $1" ;; esac
}

# ---- hub counter ------------------------------------------------------------
# The setup file defaults to cams.json in the folder you are in.
hc()        { _frc_run hubgui --setup "${1:-$PWD/cams.json}" "${@:2}"; }                 # the set-up page
hc-share()  { _frc_run hubgui --share --setup "${1:-$PWD/cams.json}" "${@:2}"; }         # ... and on Wi-Fi with a PIN
hc-feed()   { _frc_run hubfeed --setup "${1:-$PWD/cams.json}" "${@:2}"; }                # count and send to the field, no page
hc-count()  { _frc_run hubcount "$@"; }                                                   # count a recording: hc-count match.mp4 --setup cams.json
hc-listen() { _frc_run hubfeed-listen "$@"; }                                             # stand in for bioarena: print the feed
hc-board()  { _frc_open "http://127.0.0.1:${1:-8790}/board"; }                           # the full-screen scoreboard

# ---- Watchtower -------------------------------------------------------------
# The installed app if there is one, else this checkout with a clone of
# watchtower-fms in watchtower-src/ (WATCHTOWER_SRC to point elsewhere).
wt() {
  local app=""
  for a in "/Applications/Watchtower.app/Contents/MacOS/Watchtower" "$HOME/Watchtower/Watchtower" "$HOME/Applications/Watchtower/Watchtower"; do
    [ -x "$a" ] && { app="$a"; break; }
  done
  if [ -n "$app" ]; then "$app" "$@"
  else (cd "$(_frc_root)" && PYTHONPATH="$(_frc_root)" "$(_frc_py)" apps/watchtower/main.py "$@"); fi
}
wt-cli()    { wt --no-window "$@"; }                                                      # Control in the browser, logs here
wt-open()   { _frc_open "http://localhost:${1:-8000}/${2:-control}"; }                    # wt-open 8000 ref / emcee / queue
wt-logs()   { tail -n 50 -f "$HOME/Documents/Watchtower/watchtower.log" "$HOME/Documents/Watchtower/vision.log"; }
wt-pins()   { grep -A5 'pins:' "$HOME/Documents/Watchtower/config/event.yaml"; }

# ---- fuel models ------------------------------------------------------------
fuel-export()  { (cd "$(_frc_root)" && "$(_frc_py)" train/export_onnx.py "$@"); }        # .pt -> .onnx, checked box for box
fuel-compare() { (cd "$(_frc_root)" && "$(_frc_py)" train/compare_detectors.py "$@"); }  # score tables on a val set
fuel-count()   { (cd "$(_frc_root)" && "$(_frc_py)" train/count_einstein.py "$@"); }     # count Einstein 1 vs the scoreboard

# ---- this repo --------------------------------------------------------------
frc-test()   { (cd "$(_frc_root)" && "$(_frc_py)" tests/test_pipeline.py "$@"); }
frc-update() { (cd "$(_frc_root)" && git pull --ff-only); }

frc() {
  cat <<'EOF'
hub counter
  hc [cams.json]           the set-up page (cameras, outlines, drop test, Start)
  hc-share [cams.json]     the same, also on phones over Wi-Fi behind a PIN
  hc-feed [cams.json] ...  count and send to the field with no page (--target HOST:PORT)
  hc-count VIDEO --setup cams.json   count a recording; --hand red=N blue=M to check it
  hc-listen                stand in for bioarena and print what arrives
  hc-board [port]          the full-screen scoreboard
Watchtower
  wt / wt-cli              start it (app window / browser), --no-vision to score by hand
  wt-open [port] [page]    open control, ref, emcee, queue, display or overlay
  wt-pins / wt-logs        the PINs, and the live logs
fuel models
  fuel-export MODEL.pt     .onnx for the apps (refused unless boxes match)
  fuel-compare --data D --yolo NAME=W.pt --rfdetr NAME=CKPT   score tables
  fuel-count --video e1.mp4 --model NAME=W.pt                  Einstein 1 vs the scoreboard
repo
  frc-test / frc-update    run the tests / pull the latest
EOF
}
