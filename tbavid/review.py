"""Optional human pass over the automatic camera classification.

Serves a local page of every shot in the batch, tinted by the keep/drop the
clusterer chose, and writes any overrides back to the manifest. Quarantined
videos sort to the top because those are the ones the classifier already
admitted it wasn't sure about.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List

from .config import MANIFEST_PATH, REVIEW_DIR

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shot review</title>
<style>
  :root { color-scheme: dark; --bg:#14161a; --card:#1e2127; --edge:#2e333c;
          --keep:#2f9e5f; --drop:#b3453c; --ink:#e6e8ec; --dim:#98a0ad; }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { color:var(--dim); margin-bottom:20px; }
  .vid { background:var(--card); border:1px solid var(--edge); border-radius:10px;
         padding:16px; margin-bottom:18px; }
  .vid.quar { border-color:var(--drop); }
  .vh { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:12px; }
  .vh b { font-size:15px; }
  .vh span { color:var(--dim); font-size:12px; }
  .badge { background:var(--drop); color:#fff; border-radius:4px; padding:1px 7px;
           font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:10px; }
  .shot { border:3px solid var(--drop); border-radius:8px; overflow:hidden;
          cursor:pointer; background:#000; opacity:.5; transition:.12s; }
  .shot.keep { border-color:var(--keep); opacity:1; }
  .shot:hover { transform:translateY(-2px); }
  .shot img { display:block; width:100%; aspect-ratio:16/9; object-fit:cover; }
  .shot .meta { padding:5px 7px; font-size:11px; color:var(--dim);
                display:flex; justify-content:space-between; gap:6px; }
  .noimg { display:grid; place-items:center; aspect-ratio:16/9; color:var(--dim); font-size:11px; }
  button { background:#2b3039; color:var(--ink); border:1px solid var(--edge);
           border-radius:6px; padding:5px 11px; cursor:pointer; font-size:12px; }
  button:hover { background:#353b46; }
  .bar { position:sticky; top:0; z-index:5; background:var(--bg); padding:10px 0 14px;
         display:flex; gap:10px; align-items:center; border-bottom:1px solid var(--edge);
         margin-bottom:18px; }
  .save { background:var(--keep); border-color:var(--keep); color:#fff; font-weight:600;
          padding:7px 18px; }
  #status { color:var(--dim); }
</style></head><body>
<h1>Shot review</h1>
<div class="sub">Green = kept as main camera. Click any shot to flip it. Saving re-renders
only the videos you changed, then exports frames.</div>
<div class="bar">
  <button class="save" onclick="save()">Save &amp; continue</button>
  <button onclick="window.close()">Close without saving</button>
  <span id="status"></span>
</div>
<div id="app"></div>
<script>
const DATA = __DATA__;
function render() {
  const app = document.getElementById('app');
  app.innerHTML = '';
  const ids = Object.keys(DATA).sort((a,b) =>
    (DATA[b].quarantined?1:0) - (DATA[a].quarantined?1:0) || a.localeCompare(b));
  for (const id of ids) {
    const v = DATA[id];
    const el = document.createElement('div');
    el.className = 'vid' + (v.quarantined ? ' quar' : '');
    const kept = v.shots.filter(s => s.keep).length;
    el.innerHTML = `<div class="vh">
        <b>${id}</b>
        ${v.quarantined ? '<span class="badge">quarantined</span>' : ''}
        <span>${v.shots.length} shots &middot; ${kept} kept &middot;
              ${(v.coverage*100).toFixed(0)}% coverage &middot;
              ${v.n_clusters} clusters</span>
        <span style="margin-left:auto"></span>
        <button data-all="1">Keep all</button>
        <button data-all="0">Drop all</button>
      </div><div class="grid"></div>`;
    el.querySelectorAll('button[data-all]').forEach(b => b.onclick = () => {
      const on = b.dataset.all === '1';
      v.shots.forEach(s => s.keep = on);
      render();
    });
    const grid = el.querySelector('.grid');
    v.shots.forEach((s, i) => {
      const c = document.createElement('div');
      c.className = 'shot' + (s.keep ? ' keep' : '');
      const img = s.thumb
        ? `<img loading="lazy" src="thumbs/${s.thumb}" alt="">`
        : `<div class="noimg">no thumbnail</div>`;
      c.innerHTML = img + `<div class="meta">
          <span>${s.start.toFixed(1)}&ndash;${s.end.toFixed(1)}s</span>
          <span>c${s.cluster} &middot; ${s.duration.toFixed(1)}s</span></div>`;
      c.onclick = () => { s.keep = !s.keep; render(); };
      grid.appendChild(c);
    });
    app.appendChild(el);
  }
}
async function save() {
  const out = {};
  for (const [id, v] of Object.entries(DATA)) out[id] = v.shots.map(s => !!s.keep);
  document.getElementById('status').textContent = 'saving...';
  const r = await fetch('/save', {method:'POST', body: JSON.stringify(out)});
  const j = await r.json();
  document.getElementById('status').textContent =
    j.changed.length ? `saved - ${j.changed.length} video(s) changed. You can close this tab.`
                     : 'saved - no changes. You can close this tab.';
}
render();
</script></body></html>
"""


def _payload(manifest: dict) -> Dict:
    out = {}
    for vid, entry in manifest.get("videos", {}).items():
        analysis = entry.get("analysis") or {}
        if not analysis.get("shots"):
            continue
        out[vid] = {
            "quarantined": entry.get("status") == "quarantined",
            "coverage": analysis.get("coverage", 0.0),
            "n_clusters": analysis.get("n_clusters", 0),
            "shots": [
                {"start": s["start"], "end": s["end"], "duration": s["duration"],
                 "cluster": s.get("cluster", 0), "keep": bool(s.get("keep")),
                 "thumb": s.get("thumb")}
                for s in analysis["shots"]
            ],
        }
    return out


def serve(manifest: dict, port: int = 8731) -> List[str]:
    """Block on a local review page. Returns the ids whose decisions changed."""
    payload = _payload(manifest)
    if not payload:
        print("  nothing to review (no analysed shots in the manifest)")
        return []

    page = PAGE.replace("__DATA__", json.dumps(payload))
    done = threading.Event()
    changed: List[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(REVIEW_DIR), **kw)

        def log_message(self, *a):  # keep the console readable
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            decisions = json.loads(self.rfile.read(length) or b"{}")
            changed.extend(apply_decisions(manifest, decisions))
            body = json.dumps({"changed": changed}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"  review UI: {url}   (Ctrl-C to skip)")
    webbrowser.open(url)
    try:
        done.wait()
    except KeyboardInterrupt:
        print("\n  review skipped")
    finally:
        server.shutdown()
    return changed


def apply_decisions(manifest: dict, decisions: Dict[str, List[bool]]) -> List[str]:
    """Write keep flags back and re-derive keep_ranges. Returns changed ids."""
    from . import shots as shots_mod
    from .config import load_config

    cfg = load_config()
    changed = []
    for vid, keeps in decisions.items():
        entry = manifest.get("videos", {}).get(vid)
        if not entry:
            continue
        analysis = entry.get("analysis") or {}
        shot_list = analysis.get("shots") or []
        if len(keeps) != len(shot_list):
            continue
        before = [bool(s.get("keep")) for s in shot_list]
        after = [bool(k) for k in keeps]
        if before == after:
            continue
        for s, k in zip(shot_list, after):
            s["keep"] = k
        info = {"width": analysis["width"], "height": analysis["height"],
                "fps": analysis["fps"]}
        entry["analysis"] = shots_mod.finalize(
            shot_list, analysis["duration"], cfg, info,
            analysis.get("n_cuts", 0), analysis.get("n_clusters", 0),
            analysis.get("main_cluster", 0))
        # A human looked at it, so the automatic quarantine no longer applies.
        entry["status"] = "ok" if entry["analysis"]["keep_ranges"] else "empty"
        entry["reviewed"] = True
        entry["exported"] = None
        changed.append(vid)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return changed
