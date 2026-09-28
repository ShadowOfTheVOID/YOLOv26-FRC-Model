"""The hub counter as a native window, in Qt (PySide6).

    pip install PySide6
    python3 run.py hubgui --ui qt

The same controls as the web page (`hubweb.py`), in a desktop window. All the
logic is `hubapp.HubController`; this file only lays out widgets, turns
clicks into calls on it, and repaints from `controller.state()` four times a
second -- the same state the web page polls, so the two cannot disagree.

PySide6 is imported at the top of this module, so only `run.py hubgui --ui
qt` imports it; nothing else in the repository depends on Qt.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from .hubapp import (DEFAULT_TARGET, HubController, encode, fit_scale,
                     probe_cameras)

COL = {"red": QColor("#e53935"), "blue": QColor("#1e88e5")}
VIDEO_FILTER = "Video (*.mp4 *.mov *.mkv *.avi *.m4v);;All files (*)"


class PictureView(QWidget):
    """The camera picture, the hub outlines over it, and outline drawing."""

    def __init__(self, win: "HubWindow"):
        super().__init__()
        self.win = win
        self.pix: Optional[QPixmap] = None
        self.frame_w = 0
        self.scale = 1.0
        self.drawing: Optional[Dict] = None
        self.setMinimumSize(640, 360)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame(self, frame) -> None:
        h, w = frame.shape[:2]
        self.frame_w = w
        self.scale = fit_scale(w, h, max(self.width(), 320), max(self.height(), 180))
        img = QImage.fromData(encode(frame, self.scale))
        self.pix = QPixmap.fromImage(img)
        self.update()

    def clear(self) -> None:
        self.pix = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        if self.pix is None:
            p.setPen(QColor("#aaa"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Add a camera or a recording to begin.")
            return
        p.drawPixmap(0, 0, self.pix)
        cam = self.win.cam()
        k = self.scale
        live = self.win.last_state.get("live", {}).get("zones", {}) \
            if self.win.last_state else {}
        p.setFont(QFont("Helvetica", 12, QFont.Bold))
        for z in (cam or {}).get("zones", []):
            poly = QPolygonF([QPointF(x * k, y * k) for x, y in z["outline"]])
            p.setPen(QPen(COL[z["hub"]], 3))
            p.drawPolygon(poly)
            label = z["name"] + (f": {live[z['name']]}" if z["name"] in live else "")
            p.drawText(QPointF(z["outline"][0][0] * k, z["outline"][0][1] * k - 6), label)
        if self.drawing:
            pen = QPen(COL[self.drawing["hub"]], 2, Qt.DashLine)
            p.setPen(pen)
            pts = [QPointF(x * k, y * k) for x, y in self.drawing["pts"]]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            p.setBrush(COL[self.drawing["hub"]])
            for q in pts:
                p.drawEllipse(q, 5, 5)

    def mousePressEvent(self, e):
        if not self.drawing:
            return
        if e.button() == Qt.RightButton:
            self.win.finish_outline()
            return
        pos = e.position()
        self.drawing["pts"].append([pos.x() / self.scale, pos.y() / self.scale])
        self.update()

    def mouseDoubleClickEvent(self, e):
        if self.drawing:
            self.win.finish_outline()

    def keyPressEvent(self, e):
        if not self.drawing:
            return
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.win.finish_outline()
        elif e.key() == Qt.Key_Escape:
            self.drawing = None
            self.win.hint.setText("")
            self.update()


class HubWindow(QMainWindow):
    def __init__(self, ctl: HubController):
        super().__init__()
        self.ctl = ctl
        self.current: Optional[str] = None
        self.last_state: Dict = {}
        self.log_id = 0
        self.job_seen = 0
        self.pic_key = ""
        self.filling = False
        self.setWindowTitle("Hub FUEL counter")
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)
        self.refresh()

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        left = QVBoxLayout()
        outer.addLayout(left, 1)
        right = QVBoxLayout()
        outer.addLayout(right)

        bar = QHBoxLayout()
        for text, fn in (("Open setup…", self.open_setup), ("Save", self.save),
                         ("Save as…", self.save_as)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        self.path_lbl = QLabel("(unsaved setup)")
        bar.addWidget(self.path_lbl, 1)
        left.addLayout(bar)

        score = QHBoxLayout()
        self.red = QLabel("RED 0")
        self.blue = QLabel("BLUE 0")
        for lbl, col in ((self.red, "#e53935"), (self.blue, "#1e88e5")):
            lbl.setStyleSheet(f"background:{col};color:white;font:bold 30px;"
                              f"padding:4px 16px;border-radius:8px")
            lbl.setMinimumWidth(170)
            lbl.setAlignment(Qt.AlignCenter)
            score.addWidget(lbl)
        self.link = QLabel("stopped")
        self.link.setWordWrap(True)
        score.addWidget(self.link, 1)
        left.addLayout(score)

        self.view = PictureView(self)
        left.addWidget(self.view, 1)
        self.hint = QLabel("")
        self.hint.setStyleSheet("font-size:15px")
        left.addWidget(self.hint)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.setStyleSheet("font-family:Menlo,monospace;font-size:11px")
        left.addWidget(self.log)

        # 1. cameras
        g = QGroupBox("1. Cameras")
        v = QVBoxLayout(g)
        self.cams = QListWidget()
        self.cams.setMaximumHeight(90)
        self.cams.currentTextChanged.connect(self._pick_camera)
        v.addWidget(self.cams)
        row = QHBoxLayout()
        for text, fn in (("Find cameras", self.find_cameras),
                         ("Add recording…", self.add_recording),
                         ("Remove", self.remove_camera)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        form = QFormLayout()
        self.f = {}
        for key, label in (("name", "Name"), ("source", "Source"),
                           ("fps", "Frame rate"), ("size", "Size (WxH)"),
                           ("ball_area", "One ball (px)")):
            e = QLineEdit()
            e.editingFinished.connect(self.store_camera)
            form.addRow(label, e)
            self.f[key] = e
        self.blur = QSlider(Qt.Horizontal)
        self.blur.setRange(0, 10)
        self.blur_lbl = QLabel("0.0")
        self.blur.valueChanged.connect(lambda n: self.blur_lbl.setText(f"{n / 10:.1f}"))
        self.blur.sliderReleased.connect(self.store_camera)
        br = QHBoxLayout()
        br.addWidget(self.blur)
        br.addWidget(self.blur_lbl)
        form.addRow("Blur correction", br)
        v.addLayout(form)
        self.static = QCheckBox("Ignore yellow that stays still")
        self.static.toggled.connect(lambda _: self.store_camera())
        v.addWidget(self.static)
        row = QHBoxLayout()
        b = QPushButton("Show picture")
        b.clicked.connect(self.grab)
        row.addWidget(b)
        b = QPushButton("Measure ball")
        b.clicked.connect(self.measure)
        row.addWidget(b)
        self.seek = QLineEdit()
        self.seek.setPlaceholderText("at s")
        self.seek.setMaximumWidth(60)
        row.addWidget(self.seek)
        v.addLayout(row)
        right.addWidget(g)

        # 2. outlines
        g = QGroupBox("2. Hub outlines")
        v = QVBoxLayout(g)
        self.zones = QListWidget()
        self.zones.setMaximumHeight(80)
        v.addWidget(self.zones)
        row = QHBoxLayout()
        for hub in ("red", "blue"):
            b = QPushButton(f"Draw {hub.upper()}")
            b.setStyleSheet(f"color:{COL[hub].name()};font-weight:bold")
            b.clicked.connect(lambda _=False, h=hub: self.start_outline(h))
            row.addWidget(b)
        b = QPushButton("Delete")
        b.clicked.connect(self.delete_zone)
        row.addWidget(b)
        v.addLayout(row)
        form = QFormLayout()
        self.combine = {}
        for hub in ("red", "blue"):
            cb = QComboBox()
            cb.addItems(["sum", "max", "median"])
            cb.currentTextChanged.connect(
                lambda how, h=hub: None if self.filling else self._call(
                    lambda: self.ctl.set_combine(h, how)))
            form.addRow(f"{hub.capitalize()} zones", cb)
            self.combine[hub] = cb
        v.addLayout(form)
        note = QLabel("sum: cameras see different balls.  max / median: "
                      "several cameras watch the same balls.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#667085;font-size:11px")
        v.addWidget(note)
        right.addWidget(g)

        # 3. calibrate
        g = QGroupBox("3. Calibrate (optional)")
        v = QVBoxLayout(g)
        note = QLabel("Record this camera while balls go in, count them by "
                      "hand, then pick the recording.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#667085;font-size:11px")
        v.addWidget(note)
        b = QPushButton("Calibrate from recording…")
        b.clicked.connect(self.calibrate)
        v.addWidget(b)
        right.addWidget(g)

        # 4. run
        g = QGroupBox("4. Run")
        v = QVBoxLayout(g)
        form = QFormLayout()
        self.target = QLineEdit(DEFAULT_TARGET)
        form.addRow("bioarena", self.target)
        v.addLayout(form)
        self.practice = QCheckBox("Practice: send to a test receiver here")
        self.realtime = QCheckBox("Play recordings at real speed")
        self.realtime.setChecked(True)
        self.csv = QCheckBox("Save a CSV log of every count")
        self.csv.setChecked(True)
        for w in (self.practice, self.realtime, self.csv):
            v.addWidget(w)
        row = QHBoxLayout()
        self.start_btn = QPushButton("START")
        # A style sheet hides Qt's own disabled look, so say it explicitly:
        # START looked pressable while the feed ran.
        self.start_btn.setStyleSheet(
            "QPushButton{background:#43a047;color:white;font:bold 18px;"
            "padding:8px 20px}QPushButton:disabled{background:#c8e6c9;"
            "color:#f1f8e9}")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#c62828;color:white;font:bold 18px;"
            "padding:8px 20px}QPushButton:disabled{background:#ffcdd2;"
            "color:#fff5f5}")
        self.stop_btn.clicked.connect(self.ctl.stop)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        v.addLayout(row)
        right.addWidget(g)
        right.addStretch(1)

    # -- helpers ---------------------------------------------------------------
    def cam(self) -> Optional[Dict]:
        for c in (self.last_state.get("cfg") or {}).get("cameras", []):
            if c["name"] == self.current:
                return c
        return None

    def _call(self, fn):
        try:
            return fn()
        except (ValueError, KeyError, OSError) as e:
            QMessageBox.warning(self, "Hub counter", str(e))
            return None

    # -- polling ---------------------------------------------------------------
    def refresh(self) -> None:
        s = self.ctl.state(self.log_id)
        self.last_state = s
        for i, t, m in s["log"]:
            self.log_id = i
            self.log.appendPlainText(f"{t}  {m}")
        self.path_lbl.setText(s["path"] or "(unsaved setup)")
        names = [c["name"] for c in s["cfg"]["cameras"]]
        if self.current not in names:
            self.current = names[0] if names else None
        self.filling = True
        if [self.cams.item(i).text() for i in range(self.cams.count())] != names:
            self.cams.clear()
            self.cams.addItems(names)
        if self.current:
            items = self.cams.findItems(self.current, Qt.MatchExactly)
            if items and self.cams.currentItem() is not items[0]:
                self.cams.setCurrentItem(items[0])
        for hub, cb in self.combine.items():
            cb.setCurrentText(s["cfg"]["combine"].get(hub, "sum"))
        self.filling = False
        self._fill_form()
        self._fill_zones()
        self._live(s)
        self._job(s["job"])
        pic = s["pictures"].get(self.current) if self.current else None
        key = f"{self.current}:{pic}:{s['job']['id']}"
        if pic and (s["running"] or key != self.pic_key):
            self.pic_key = key
            frame = self.ctl.picture(self.current)
            if frame is not None:
                self.view.set_frame(frame)
        elif not pic:
            self.view.clear()
        self.view.update()
        self.start_btn.setEnabled(not s["running"])
        self.stop_btn.setEnabled(s["running"])
        if not names:
            self.hint.setText("Add a camera (Find cameras) or a recording to begin.")

    def _fill_form(self) -> None:
        c = self.cam()
        if not c:
            return
        self.filling = True
        for k, e in self.f.items():
            if not e.hasFocus():
                val = c.get(k, "")
                e.setText("" if val in (None, 0, 0.0) else str(val))
        if not self.blur.isSliderDown():
            self.blur.setValue(int(round(float(c.get("blur") or 0) * 10)))
        self.static.setChecked(bool(c.get("remove_static")))
        self.filling = False

    def _fill_zones(self) -> None:
        c = self.cam()
        live = self.last_state["live"]["zones"]
        rows = [f"{z['hub'].upper():5} {z['name']}  "
                + (f"{live[z['name']]} in" if z["name"] in live
                   else f"({len(z['outline'])} corners)")
                for z in (c or {}).get("zones", [])]
        if [self.zones.item(i).text() for i in range(self.zones.count())] != rows:
            sel = self.zones.currentRow()
            self.zones.clear()
            self.zones.addItems(rows)
            self.zones.setCurrentRow(min(sel, len(rows) - 1))

    def _live(self, s) -> None:
        v = s["live"]
        self.red.setText(f"RED {v['counts']['red']}")
        self.blue.setText(f"BLUE {v['counts']['blue']}")
        if not s["running"]:
            text, col = "stopped", "#667085"
        elif v["stale"]:
            text, col = (f"NO PICTURE from {', '.join(v['stale'])} -- bioarena "
                         f"shows OFFLINE"), "#c62828"
        elif v["linked"]:
            r = v["reply"] or {}
            text, col = (f"bioarena OK  {r.get('match_state', '')} "
                         f"{r.get('shift', '')}  {v['rtt_ms']} ms"), "#2e7d32"
        else:
            text, col = f"no reply from bioarena ({v['target']})", "#c62828"
        cams = "   ".join(f"{n} {c['fps']} fps" for n, c in v["cameras"].items())
        self.link.setText(f"<b style='color:{col}'>{text}</b><br>{cams}")

    def _job(self, j) -> None:
        if j["running"]:
            pct = f" {j['progress']:.0%}" if j.get("progress") is not None else ""
            self.hint.setText(f"{j['name']}…{pct}")
            return
        if j["id"] == self.job_seen:
            return
        self.job_seen = j["id"]
        if j["error"]:
            QMessageBox.warning(self, "Hub counter", j["error"])
            self.hint.setText("")
            return
        r = j["result"]
        if j["name"] == "find cameras":
            if not r:
                QMessageBox.warning(self, "Hub counter",
                                    "No camera answered. Check it is plugged in, "
                                    "and on macOS that this app may use the camera "
                                    "(System Settings > Privacy > Camera).")
                return
            items = [f"Camera {c['index']}  ({c['size']})" for c in r]
            pick, ok = QInputDialog.getItem(
                self, "Add a camera", "Numbers can change when a camera is "
                "re-plugged -- check the picture after adding:", items, 0, False)
            if ok:
                idx = str(r[items.index(pick)]["index"])
                cam = self._call(lambda: self.ctl.add_camera(idx))
                if cam:
                    self.current = cam["name"]
                    self.grab()
        elif j["name"] == "measure":
            self.hint.setText(f"One ball = {r:.0f} px. Now START, or calibrate "
                              f"first." if r else "No isolated ball found near "
                              "the outline -- put a few balls apart near the hub.")
        elif j["name"] == "calibrate" and r:
            lines = "\n".join(f"{h}: {r['counts'].get(h, 0)} counted, {n} by hand "
                              f"(uncorrected {r['uncorrected'].get(h, 0)})"
                              for h, n in r["hand"].items())
            ask = (f"Best: blur {r['blur']}, ignore still yellow "
                   f"{'on' if r['remove_static'] else 'off'}\n{lines}\n\n"
                   f"Fitted to one recording -- check it on a second one before "
                   f"trusting it.\n\nUse these settings?")
            if QMessageBox.question(self, "Calibration", ask) == QMessageBox.Yes:
                self._call(lambda: self.ctl.update_camera(
                    r["camera"], {"blur": r["blur"],
                                  "remove_static": r["remove_static"]}))
        elif j["name"] == "picture":
            c = self.cam()
            if c and not c["zones"]:
                self.hint.setText("Now draw each hub's outline: Draw RED or Draw "
                                  "BLUE, then click the corners.")

    # -- actions -------------------------------------------------------------------
    def _pick_camera(self, name: str) -> None:
        if self.filling or not name or name == self.current:
            return
        self.current = name
        self.pic_key = ""
        if name not in self.last_state.get("pictures", {}):
            self.grab()

    def store_camera(self) -> None:
        if self.filling or not self.current:
            return
        fields = {k: e.text() for k, e in self.f.items()}
        fields["blur"] = self.blur.value() / 10
        fields["remove_static"] = self.static.isChecked()
        c = self._call(lambda: self.ctl.update_camera(self.current, fields))
        if c:
            self.current = c["name"]

    def find_cameras(self) -> None:
        self.hint.setText("Looking for cameras…")
        self.ctl.job("find cameras", probe_cameras)

    def add_recording(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "A recording to set up or "
                                           "practise on", "", VIDEO_FILTER)
        if p:
            cam = self._call(lambda: self.ctl.add_camera(p))
            if cam:
                self.current = cam["name"]
                self.grab()

    def remove_camera(self) -> None:
        if self.current and QMessageBox.question(
                self, "Remove", f"Remove {self.current}?") == QMessageBox.Yes:
            self.ctl.remove_camera(self.current)
            self.current = None

    def grab(self) -> None:
        if not self.current:
            return
        name = self.current
        try:
            at = float(self.seek.text()) if self.seek.text().strip() else None
        except ValueError:
            at = None
        self.ctl.job("picture", lambda: (self.ctl.grab(name, at), None)[1])

    def measure(self) -> None:
        if self.current:
            self.hint.setText("Measuring for 5 s -- balls should be sitting "
                              "apart near the hub…")
            name = self.current
            self.ctl.job("measure", lambda: self.ctl.measure(name))

    def start_outline(self, hub: str) -> None:
        if not self.cam() or self.view.pix is None:
            QMessageBox.warning(self, "Hub counter",
                                "Pick a camera and show its picture first.")
            return
        self.view.drawing = {"hub": hub, "pts": []}
        self.view.setFocus()
        self.hint.setText(f"Click the corners of the {hub.upper()} hub's opening. "
                          f"Double-click, right-click or Enter to finish; Esc to "
                          f"cancel.")
        self.view.update()

    def finish_outline(self) -> None:
        d = self.view.drawing
        if not d:
            return
        z = self._call(lambda: self.ctl.add_zone(self.current, d["hub"], d["pts"]))
        if z:
            self.view.drawing = None
            self.hint.setText("Outline saved. Put a few balls near the hub and "
                              "press Measure ball.")
        self.view.update()

    def delete_zone(self) -> None:
        c = self.cam()
        row = self.zones.currentRow()
        if c and 0 <= row < len(c["zones"]):
            self.ctl.delete_zone(c["name"], c["zones"][row]["name"])

    def calibrate(self) -> None:
        c = self.cam()
        if not c:
            return
        video, _ = QFileDialog.getOpenFileName(
            self, f"A recording from {c['name']}, same position and settings",
            "", VIDEO_FILTER)
        if not video:
            return
        hand = {}
        for hub in sorted({z["hub"] for z in c["zones"]}):
            n, ok = QInputDialog.getInt(self, "Hand count", f"Balls that went "
                                        f"into the {hub.upper()} hub in this "
                                        f"recording:", 0, 0, 100000)
            if not ok:
                return
            hand[hub] = n
        name = c["name"]
        self.ctl.job("calibrate", lambda: self.ctl.calibrate(name, video, hand))

    def start(self) -> None:
        self.store_camera()
        self._call(lambda: self.ctl.start(self.target.text(),
                                          self.practice.isChecked(),
                                          self.realtime.isChecked(),
                                          self.csv.isChecked()))

    def open_setup(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Open setup", "", "Setup (*.json)")
        if p:
            self._call(lambda: self.ctl.load(p))
            self.current = None
            self.pic_key = ""

    def save(self) -> None:
        if not self.ctl.path:
            return self.save_as()
        self._call(self.ctl.save)

    def save_as(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save setup", "cams.json",
                                           "Setup (*.json)")
        if p:
            self._call(lambda: self.ctl.save(p))

    def closeEvent(self, e):
        self.ctl.stop()
        super().closeEvent(e)


def main(setup_path: Optional[str] = None) -> int:
    app = QApplication.instance() or QApplication([])
    win = HubWindow(HubController(setup_path))
    win.resize(1400, 900)
    win.show()
    return app.exec()
