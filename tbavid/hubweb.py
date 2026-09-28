"""The hub counter as a web page: standard library only, any browser.

    python3 run.py hubgui --ui web          # opens http://127.0.0.1:8790

No toolkit to install -- the Tk window needed `brew install python-tk` on a
Homebrew Mac -- and it looks the same on every machine. All the logic is in
`hubapp.HubController`; this file only turns HTTP into calls on it.

It listens on 127.0.0.1 and controls cameras and reads the disk, so it also
refuses requests whose Host header is not local (DNS rebinding) and POSTs
that are not JSON (a cross-site form cannot send application/json without a
preflight this server never answers). `--bind` can widen it for a trusted
field network; the page is then an open control panel to anyone on it.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .hubapp import HubController, encode, fit_scale, list_dir, probe_cameras

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def make_handler(ctl: HubController, allow_remote: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):     # quiet; the page has its log
            pass

        # -- plumbing ---------------------------------------------------------
        def _host_ok(self) -> bool:
            if allow_remote:
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in LOCAL_HOSTS

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        # -- GET ----------------------------------------------------------------
        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"local only", "text/plain")
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._json(ctl.state(int(q.get("log", 0) or 0)))
            if u.path == "/api/ls":
                return self._json(list_dir(q.get("path", "")))
            if u.path == "/frame.jpg":
                frame = ctl.picture(q.get("cam", ""))
                if frame is None:
                    return self._send(404, b"no picture", "text/plain")
                h, w = frame.shape[:2]
                mw = int(q.get("w", 960) or 960)
                return self._send(200, encode(frame, fit_scale(w, h, mw, mw)),
                                  "image/jpeg")
            self._send(404, b"not found", "text/plain")

        # -- POST ---------------------------------------------------------------
        def do_POST(self):
            if not self._host_ok():
                return self._send(403, b"local only", "text/plain")
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._send(415, b"json only", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            action = urlparse(self.path).path.rsplit("/", 1)[-1]
            try:
                result = ACTIONS[action](ctl, body)
            except KeyError as e:
                return self._json({"error": f"unknown: {e}"}, 404)
            except (ValueError, OSError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True, "result": result})

    return Handler


def _at(body) -> Optional[float]:
    v = body.get("at")
    return float(v) if v not in (None, "") else None


ACTIONS = {
    "add_camera": lambda c, b: c.add_camera(b.get("source", ""), b.get("name", "")),
    "remove_camera": lambda c, b: c.remove_camera(b["name"]),
    "update_camera": lambda c, b: c.update_camera(b["name"], b.get("fields", {})),
    "add_zone": lambda c, b: c.add_zone(b["camera"], b["hub"], b["points"]),
    "delete_zone": lambda c, b: c.delete_zone(b["camera"], b["zone"]),
    "combine": lambda c, b: c.set_combine(b["hub"], b["how"]),
    "grab": lambda c, b: c.job("picture", lambda: (c.grab(b["camera"], _at(b)), None)[1]),
    "measure": lambda c, b: c.job("measure", lambda: c.measure(b["camera"])),
    "find_cameras": lambda c, b: c.job("find cameras", probe_cameras),
    "calibrate": lambda c, b: c.job("calibrate", lambda: c.calibrate(
        b["camera"], b["video"], {k: int(v) for k, v in b["hand"].items()})),
    "apply_calibration": lambda c, b: c.update_camera(
        b["camera"], {"blur": b["blur"], "remove_static": b["remove_static"]}),
    "start": lambda c, b: c.start(b.get("target", ""), bool(b.get("practice")),
                                  bool(b.get("realtime", True)),
                                  bool(b.get("log", True))),
    "stop": lambda c, b: c.stop(),
    "save": lambda c, b: c.save(b.get("path") or None),
    "load": lambda c, b: c.load(b["path"]),
}


def serve(ctl: HubController, port: int = 8790, bind: str = "127.0.0.1",
          open_browser: bool = True) -> None:
    remote = bind not in ("127.0.0.1", "localhost", "::1")
    httpd = ThreadingHTTPServer((bind, port), make_handler(ctl, remote))
    url = f"http://{'127.0.0.1' if bind in ('0.0.0.0', '') else bind}:{port}/"
    print(f"hub counter at {url}  (Ctrl-C to quit)")
    if remote:
        print("! listening beyond this machine: anyone who can reach it can "
              "start, stop and reconfigure the counter")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop()
        httpd.server_close()


def main(setup_path: Optional[str] = None, port: int = 8790,
         bind: str = "127.0.0.1", open_browser: bool = True) -> int:
    serve(HubController(setup_path), port, bind, open_browser)
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hub FUEL counter</title>
<style>
:root{--bg:#f4f5f7;--panel:#fff;--ink:#1d2330;--muted:#667085;--line:#d9dde5;
--red:#e53935;--blue:#1e88e5;--ok:#2e7d32;--bad:#c62828;--accent:#1f6feb}
@media (prefers-color-scheme:dark){:root{--bg:#14161b;--panel:#1d2027;--ink:#e8eaf0;
--muted:#98a2b3;--line:#323743}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:16px;margin:0 12px 0 0}header .path{color:var(--muted);font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px;padding:14px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.score{display:flex;gap:8px;align-items:stretch;margin-bottom:8px}
.big{color:#fff;font-weight:800;font-size:34px;padding:4px 18px;border-radius:8px;min-width:170px;text-align:center}
.big.red{background:var(--red)}.big.blue{background:var(--blue)}
.link{flex:1;padding:4px 10px;font-size:14px;border-radius:8px;background:var(--panel);border:1px solid var(--line)}
.link b.ok{color:var(--ok)}.link b.bad{color:var(--bad)}
.view{position:relative;background:#111;border-radius:8px;overflow:hidden;min-height:200px;display:flex;align-items:center;justify-content:center}
.stage{position:relative;display:inline-block;line-height:0}
.stage img{display:block;max-width:100%}.stage canvas{position:absolute;left:0;top:0;cursor:crosshair}
.empty{color:#aaa;padding:60px 20px;text-align:center}
.hint{margin:8px 0;font-size:15px;min-height:22px}
#log{height:150px;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font:12px ui-monospace,Menlo,monospace;white-space:pre-wrap}
aside section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:10px}
aside h2{font-size:13px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.list{border:1px solid var(--line);border-radius:6px;max-height:120px;overflow:auto;margin-bottom:6px}
.list div{padding:5px 8px;cursor:pointer;display:flex;justify-content:space-between;gap:6px}
.list div.sel{background:color-mix(in srgb,var(--accent) 18%,transparent)}
.list .warn{color:var(--bad);font-size:12px}
.row{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
button{font:inherit;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}button:disabled{opacity:.45;cursor:default}
button.red{color:var(--red);font-weight:600}button.blue{color:var(--blue);font-weight:600}
button.go{background:#43a047;color:#fff;border:0;font-size:18px;font-weight:700;padding:8px 22px}
button.stop{background:var(--bad);color:#fff;border:0;font-size:18px;font-weight:700;padding:8px 22px}
label{display:block;margin:3px 0}.grid{display:grid;grid-template-columns:110px 1fr;gap:4px 6px;align-items:center}
input[type=text],input[type=number],select{font:inherit;width:100%;padding:4px 6px;border-radius:5px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.muted{color:var(--muted);font-size:12px}
dialog{border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--ink);min-width:420px;max-width:90vw}
dialog .files{max-height:50vh;overflow:auto;border:1px solid var(--line);border-radius:6px;margin:8px 0}
dialog .files div{padding:5px 8px;cursor:pointer}dialog .files div:hover{background:var(--bg)}
</style></head><body>
<header><h1>Hub FUEL counter</h1><span class="path" id="path">(unsaved setup)</span>
<button onclick="save()">Save</button><button onclick="openSetup()">Open…</button></header>
<main>
<div>
  <div class="score"><div class="big red" id="red">RED 0</div><div class="big blue" id="blue">BLUE 0</div>
  <div class="link" id="link">stopped</div></div>
  <div class="view" id="view"><div class="empty" id="empty">Add a camera or a recording to begin.</div>
  <div class="stage"><img id="img" alt=""><canvas id="cv"></canvas></div></div>
  <div class="hint" id="hint"></div>
  <div id="log"></div>
</div>
<aside>
<section><h2>1. Cameras</h2>
  <div class="list" id="cams"></div>
  <div class="row"><button onclick="findCams()">Find cameras</button><button onclick="pickFile('add')">Add recording…</button><button onclick="removeCam()">Remove</button></div>
  <div class="grid" id="form">
    <span>Name</span><input type="text" id="f_name">
    <span>Source</span><input type="text" id="f_source">
    <span>Frame rate</span><input type="number" id="f_fps" step="1" min="0">
    <span>Size (WxH)</span><input type="text" id="f_size" placeholder="1280x720">
    <span>One ball (px)</span><input type="number" id="f_ball_area" min="0">
    <span>Blur correction</span><span><input type="range" id="f_blur" min="0" max="1" step="0.1"> <b id="blurv">0</b></span>
  </div>
  <label><input type="checkbox" id="f_remove_static"> Ignore yellow that stays still</label>
  <div class="row"><button onclick="grab()">Show picture</button><button onclick="measure()">Measure ball</button>
  <span class="muted">recording at <input type="number" id="seek" style="width:60px"> s</span></div>
</section>
<section><h2>2. Hub outlines</h2>
  <div class="list" id="zones"></div>
  <div class="row"><button class="red" onclick="draw('red')">Draw RED</button><button class="blue" onclick="draw('blue')">Draw BLUE</button><button onclick="delZone()">Delete</button></div>
  <div class="grid"><span>Red zones</span><select id="c_red"><option>sum</option><option>max</option><option>median</option></select>
  <span>Blue zones</span><select id="c_blue"><option>sum</option><option>max</option><option>median</option></select></div>
  <p class="muted">sum: cameras see different balls. max / median: several cameras watch the same balls.</p>
</section>
<section><h2>3. Calibrate (optional)</h2>
  <p class="muted">Record this camera while balls go in, count them by hand, then pick the recording.</p>
  <button onclick="pickFile('calibrate')">Calibrate from recording…</button>
</section>
<section><h2>4. Run</h2>
  <div class="grid"><span>bioarena</span><input type="text" id="target" value="10.0.100.5:8411"></div>
  <label><input type="checkbox" id="practice"> Practice: send to a test receiver here</label>
  <label><input type="checkbox" id="realtime" checked> Play recordings at real speed</label>
  <label><input type="checkbox" id="csv" checked> Save a CSV log of every count</label>
  <div class="row"><button class="go" id="start" onclick="start()">START</button><button class="stop" id="stop" onclick="stop()" disabled>STOP</button></div>
</section>
</aside></main>
<dialog id="dlg"><b id="dlgTitle">Pick a recording</b><div class="muted" id="dlgPath"></div>
<div class="files" id="files"></div><div class="row"><button onclick="dlg.close()">Cancel</button></div></dialog>
<script>
const $=id=>document.getElementById(id);
let S=null, cur=null, drawing=null, logId=0, lastPic="", jobSeen=0, pickMode=null;
const COL={red:"#e53935",blue:"#1e88e5"};
async function api(action,body={}){
  const r=await fetch("/api/"+action,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const j=await r.json(); if(!r.ok||j.error){alert(j.error||"failed");throw j;} return j.result;}
function cam(){return S&&S.cfg.cameras.find(c=>c.name===cur);}
function hint(t){$("hint").textContent=t;}
async function refresh(){
  let r; try{r=await fetch("/api/state?log="+logId);}catch(e){$("link").innerHTML='<b class="bad">window lost the counter -- is run.py still running?</b>';return;}
  S=await r.json();
  for(const [id,t,m] of S.log){logId=id;const d=$("log");d.textContent+=t+"  "+m+"\n";d.scrollTop=d.scrollHeight;}
  $("path").textContent=S.path||"(unsaved setup)";
  if(!cam()&&S.cfg.cameras.length) cur=S.cfg.cameras[0].name;
  renderCams(); renderZones(); renderLive(); handleJob();
  const pic=cur&&S.pictures[cur]; const key=cur+":"+(pic||"")+":"+(S.running?Date.now():S.job.id);
  if(pic&&(S.running||key!==lastPic)){lastPic=key;loadImg();}
  if(!pic){$("img").style.display="none";$("cv").style.display="none";$("empty").style.display="";}
  if(!S.cfg.cameras.length) hint("Add a camera (Find cameras) or a recording to begin.");
  if($("cv").style.display==="block") overlay();
}
function renderCams(){
  const L=$("cams"); L.innerHTML="";
  for(const c of S.cfg.cameras){const d=document.createElement("div");d.className=c.name===cur?"sel":"";
    d.innerHTML=`<span>${esc(c.name)}</span><span class="${c.ball_area?'muted':'warn'}">${c.ball_area?Math.round(c.ball_area)+' px':'measure ball'}</span>`;
    d.onclick=()=>{cur=c.name;lastPic="";fillForm();refresh();if(!S.pictures[c.name])grab();};L.appendChild(d);}
  if(document.activeElement.closest&&!document.activeElement.closest("#form")) fillForm();
  $("c_red").value=S.cfg.combine.red||"sum";$("c_blue").value=S.cfg.combine.blue||"sum";
  $("start").disabled=S.running;$("stop").disabled=!S.running;
}
function fillForm(){const c=cam();if(!c)return;
  for(const k of ["name","source","fps","size","ball_area"]) $("f_"+k).value=c[k]||"";
  $("f_blur").value=c.blur||0;$("blurv").textContent=c.blur||0;$("f_remove_static").checked=!!c.remove_static;}
function renderZones(){const L=$("zones");L.innerHTML="";const c=cam();if(!c)return;
  for(const z of c.zones){const d=document.createElement("div");d.dataset.z=z.name;
    const n=S.live.zones[z.name];d.innerHTML=`<span style="color:${COL[z.hub]}"><b>${z.hub.toUpperCase()}</b> ${esc(z.name)}</span><span class="muted">${n!==undefined?n+' in':z.outline.length+' corners'}</span>`;
    d.onclick=()=>{[...L.children].forEach(x=>x.className="");d.className="sel";};L.appendChild(d);}}
function renderLive(){const v=S.live;$("red").textContent="RED "+v.counts.red;$("blue").textContent="BLUE "+v.counts.blue;
  let h;if(!S.running) h="stopped";
  else if(v.stale.length) h=`<b class="bad">NO PICTURE from ${esc(v.stale.join(", "))} — bioarena shows OFFLINE</b>`;
  else if(v.linked) h=`<b class="ok">bioarena OK</b> ${esc((v.reply||{}).match_state||"")} ${esc((v.reply||{}).shift||"")} · ${v.rtt_ms} ms`;
  else h=`<b class="bad">no reply from bioarena</b> (${esc(v.target)})`;
  const cams=Object.entries(v.cameras).map(([n,c])=>`${esc(n)} ${c.fps} fps`).join(" · ");
  $("link").innerHTML=h+(cams?"<br><span class='muted'>"+cams+"</span>":"");}
function handleJob(){const j=S.job;if(j.id===jobSeen||j.running){if(j.running)hint(j.name+"…"+(j.progress!=null?" "+Math.round(j.progress*100)+"%":""));return;}
  jobSeen=j.id; if(j.error){hint("");return;}
  if(j.name==="find cameras"){const f=j.result||[];if(!f.length){alert("No camera answered. Check it is plugged in, and on macOS that this terminal may use the camera (System Settings > Privacy > Camera).");return;}
    const pick=prompt("Cameras found:\n"+f.map(c=>`  ${c.index}  (${c.size})`).join("\n")+"\n\nType the number to add (numbers can change when re-plugged -- check the picture):",f[0].index);
    if(pick!==null) api("add_camera",{source:String(pick)}).then(c=>{cur=c.name;grab();});}
  else if(j.name==="measure") hint(j.result?`One ball = ${Math.round(j.result)} px. Now START, or calibrate first.`:"No isolated ball found near the outline -- put a few balls apart near the hub.");
  else if(j.name==="calibrate"&&j.result){const r=j.result;
    const lines=Object.entries(r.hand).map(([h,n])=>`${h}: ${r.counts[h]||0} counted, ${n} by hand (uncorrected ${r.uncorrected[h]||0})`).join("\n");
    if(confirm(`Best: blur ${r.blur}, ignore still yellow ${r.remove_static?"on":"off"}\n${lines}\n\nFitted to one recording -- check it on a second one before trusting it.\n\nUse these settings?`))
      api("apply_calibration",{camera:r.camera,blur:r.blur,remove_static:r.remove_static});}
  else if(j.name==="picture") hint(S.cfg.cameras.length&&cam()&&!cam().zones.length?"Now draw each hub's outline: Draw RED or Draw BLUE, then click the corners.":"");
}
// The overlay is sized from the picture's own load event: sized when the
// address changed, it measured 0x0 (the picture was not laid out yet) and
// every click fell on nothing.
$("img").addEventListener("load",sizeCanvas);
function loadImg(){const img=$("img");const n=new Image();
  n.onload=()=>{img.style.display="block";$("empty").style.display="none";img.src=n.src;};
  n.src="/frame.jpg?cam="+encodeURIComponent(cur)+"&t="+Date.now();}
function sizeCanvas(){const img=$("img"),cv=$("cv");if(!img.clientWidth)return;
  if(cv.width!==img.clientWidth||cv.height!==img.clientHeight){cv.width=img.clientWidth;cv.height=img.clientHeight;}
  cv.style.display="block";overlay();}
function scale(){const p=S.pictures[cur];return p?$("img").clientWidth/p[0]:1;}
function overlay(){const cv=$("cv"),g=cv.getContext("2d");g.clearRect(0,0,cv.width,cv.height);const c=cam();if(!c)return;const k=scale();
  for(const z of c.zones){g.strokeStyle=COL[z.hub];g.lineWidth=3;g.beginPath();z.outline.forEach(([x,y],i)=>i?g.lineTo(x*k,y*k):g.moveTo(x*k,y*k));g.closePath();g.stroke();
    const n=S.live.zones[z.name];g.fillStyle=COL[z.hub];g.font="bold 14px sans-serif";g.fillText(z.name+(n!==undefined?": "+n:""),z.outline[0][0]*k,z.outline[0][1]*k-6);}
  if(drawing){g.strokeStyle=COL[drawing.hub];g.setLineDash([5,3]);g.beginPath();drawing.pts.forEach(([x,y],i)=>i?g.lineTo(x*k,y*k):g.moveTo(x*k,y*k));g.stroke();g.setLineDash([]);
    for(const [x,y] of drawing.pts){g.fillStyle=COL[drawing.hub];g.beginPath();g.arc(x*k,y*k,5,0,7);g.fill();}}}
$("cv").addEventListener("click",e=>{if(!drawing)return;const r=e.target.getBoundingClientRect(),k=scale();
  drawing.pts.push([(e.clientX-r.left)/k,(e.clientY-r.top)/k]);overlay();});
$("cv").addEventListener("dblclick",()=>finish());
$("cv").addEventListener("contextmenu",e=>{e.preventDefault();finish();});
document.addEventListener("keydown",e=>{if(!drawing)return;if(e.key==="Enter")finish();if(e.key==="Escape"){drawing=null;overlay();hint("");}});
window.addEventListener("resize",()=>{if($("img").style.display==="block")sizeCanvas();});
function draw(hub){if(!cam()||!S.pictures[cur]){alert("Pick a camera and show its picture first.");return;}
  drawing={hub,pts:[]};hint(`Click the corners of the ${hub.toUpperCase()} hub's opening. Double-click, right-click or Enter to finish; Esc to cancel.`);overlay();}
async function finish(){if(!drawing)return;const d=drawing;
  try{await api("add_zone",{camera:cur,hub:d.hub,points:d.pts});drawing=null;hint("Outline saved. Put a few balls near the hub and press Measure ball.");}
  catch(e){} refresh().then(overlay);}
function delZone(){const s=document.querySelector("#zones .sel");if(s)api("delete_zone",{camera:cur,zone:s.dataset.z}).then(refresh);}
async function storeForm(){const c=cam();if(!c)return;const f={};
  for(const k of ["name","source","fps","size","ball_area"]) f[k]=$("f_"+k).value;
  f.blur=$("f_blur").value;f.remove_static=$("f_remove_static").checked;
  try{const r=await api("update_camera",{name:cur,fields:f});cur=r.name;}catch(e){} refresh();}
for(const k of ["name","source","fps","size","ball_area"]) $("f_"+k).addEventListener("change",storeForm);
$("f_blur").addEventListener("input",()=>$("blurv").textContent=$("f_blur").value);
$("f_blur").addEventListener("change",storeForm);$("f_remove_static").addEventListener("change",storeForm);
for(const h of ["red","blue"]) $("c_"+h).addEventListener("change",e=>api("combine",{hub:h,how:e.target.value}));
function grab(){if(cur)api("grab",{camera:cur,at:$("seek").value}).then(()=>{lastPic="";});}
function measure(){if(cur){hint("Measuring for 5 s -- balls should be sitting apart near the hub…");api("measure",{camera:cur});}}
function findCams(){hint("Looking for cameras…");api("find_cameras");}
function removeCam(){if(cur&&confirm("Remove "+cur+"?"))api("remove_camera",{name:cur}).then(()=>{cur=null;refresh();});}
function start(){api("start",{target:$("target").value,practice:$("practice").checked,realtime:$("realtime").checked,log:$("csv").checked})
  .then(()=>hint("Running. Leave it on across matches; bioarena takes each match's start itself."));}
function stop(){api("stop");}
function save(){const p=S&&S.path?null:prompt("Save setup as:","cams.json");if(S&&!S.path&&p===null)return;api("save",{path:p}).then(refresh);}
function openSetup(){const p=prompt("Setup file to open:",S&&S.path||"cams.json");if(p)api("load",{path:p}).then(()=>{cur=null;lastPic="";refresh();});}
async function pickFile(mode,path=""){pickMode=mode;const L=await (await fetch("/api/ls?path="+encodeURIComponent(path))).json();
  $("dlgTitle").textContent=mode==="add"?"Add a recording":"Recording for calibration (same camera position and settings)";
  $("dlgPath").textContent=L.path+(L.error?"  -- "+L.error:"");const F=$("files");F.innerHTML="";
  const add=(t,fn)=>{const d=document.createElement("div");d.textContent=t;d.onclick=fn;F.appendChild(d);};
  add("⬆ ..",()=>pickFile(mode,L.parent));for(const d of L.dirs)add("📁 "+d,()=>pickFile(mode,L.path+"/"+d));
  for(const v of L.videos)add("🎞 "+v,()=>chose(L.path+"/"+v));if(!$("dlg").open)$("dlg").showModal();}
async function chose(p){$("dlg").close();
  if(pickMode==="add"){const c=await api("add_camera",{source:p});cur=c.name;lastPic="";grab();return;}
  const c=cam();if(!c)return;const hand={};
  for(const h of [...new Set(c.zones.map(z=>z.hub))]){const n=prompt(`Balls that went into the ${h.toUpperCase()} hub in this recording:`);if(n===null)return;hand[h]=parseInt(n,10)||0;}
  api("calibrate",{camera:cur,video:p,hand});}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
(function loop(){refresh().finally(()=>setTimeout(loop,S&&S.running?250:700));})();
</script></body></html>
"""
