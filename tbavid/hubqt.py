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
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget,
                               QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QVBoxLayout, QWidget)

from .hubapp import (DEFAULT_TARGET, HubController, encode, fit_scale,
                     probe_cameras)

COL = {"red": QColor("#ff4d4f"), "blue": QColor("#3b8cff")}

# One dark theme, the same palette as the web page, so the two front ends look
# like one product. Qt draws its own widgets from this; no platform theme.
QSS = """
* { font-family: -apple-system, "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial; font-size: 13px; }
QMainWindow, QWidget#root { background: #0b0e14; color: #e7ebf3; }
QWidget { color: #e7ebf3; }
QLabel#muted { color: #6f7a8f; font-size: 12px; }
QLabel#hint { color: #a9b2c3; font-size: 14px; }
QLabel#file { color: #a9b2c3; }
QLabel#brand { font-size: 15px; font-weight: 700; }
QFrame#topbar { background: #0f131b; border-bottom: 1px solid #252c3b; }
QFrame#card, QGroupBox { background: #141923; border: 1px solid #252c3b; border-radius: 14px; }
QGroupBox { margin-top: 14px; padding: 30px 12px 12px 12px; font-weight: 700; font-size: 14px; }
QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 14px; top: 10px; color: #e7ebf3; }
QGroupBox[done="true"]::title { color: #22c55e; }
QFrame#red { border-radius: 14px; background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ff4d4f, stop:1 #c9302c); }
QFrame#blue { border-radius: 14px; background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #3b8cff, stop:1 #1f5fd1); }
QLabel#tileLbl { color: rgba(255,255,255,.85); font-size: 11px; font-weight: 800; letter-spacing: 2px; background: transparent; }
QLabel#tileNum { color: white; font-size: 52px; font-weight: 800; background: transparent; }
QLabel#state { font-size: 15px; font-weight: 700; background: transparent; }
QLabel#sub { color: #a9b2c3; font-size: 12px; background: transparent; }
QPushButton { background: #1a2030; border: 1px solid #252c3b; border-radius: 9px; padding: 7px 13px; font-weight: 600; }
QPushButton:hover { border-color: #5b7cfa; background: #1d2540; }
QPushButton:pressed { background: #16203a; }
QPushButton:disabled { color: #4a5263; border-color: #1f2533; }
QPushButton#primary { background: #5b7cfa; border: none; color: white; }
QPushButton#primary:hover { background: #6d8bff; }
QPushButton#red { color: #ff4d4f; border-color: #5a2a33; }
QPushButton#red:hover { background: #2a1519; }
QPushButton#blue { color: #3b8cff; border-color: #22385e; }
QPushButton#blue:hover { background: #121f36; }
QPushButton#start { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2fd06a, stop:1 #16a34a); border: none; color: white; font-size: 17px; font-weight: 800; padding: 14px; border-radius: 12px; letter-spacing: 2px; }
QPushButton#start:disabled { background: #1d3326; color: #4d7a5d; }
QPushButton#stop { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ff5a5a, stop:1 #dc2626); border: none; color: white; font-size: 17px; font-weight: 800; padding: 14px; border-radius: 12px; letter-spacing: 2px; }
QPushButton#stop:disabled { background: #3a1c1f; color: #7a4a4d; }
QLineEdit, QComboBox { background: #0f131b; border: 1px solid #252c3b; border-radius: 8px; padding: 6px 9px; selection-background-color: #5b7cfa; }
QLineEdit:focus, QComboBox:focus { border-color: #5b7cfa; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #141923; border: 1px solid #252c3b; selection-background-color: #5b7cfa; }
QListWidget { background: #0f131b; border: 1px solid #252c3b; border-radius: 10px; padding: 4px; outline: none; }
QListWidget::item { padding: 7px 8px; border-radius: 7px; }
QListWidget::item:selected { background: #22305a; color: white; }
QListWidget::item:hover { background: #1a2238; }
QPlainTextEdit { background: #0f131b; border: 1px solid #252c3b; border-radius: 10px; padding: 6px; color: #a9b2c3; font-family: Menlo, Consolas, monospace; font-size: 11px; }
QCheckBox { spacing: 9px; padding: 3px 0; }
QCheckBox::indicator { width: 34px; height: 18px; border-radius: 9px; background: #252c3b; }
QCheckBox::indicator:checked { background: #5b7cfa; }
QSlider::groove:horizontal { height: 4px; background: #252c3b; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #5b7cfa; border-radius: 2px; }
QSlider::handle:horizontal { background: white; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }
QScrollArea, QScrollArea > QWidget > QWidget, QWidget#side { border: none; background: #0b0e14; }
QScrollBar:vertical { background: transparent; width: 10px; }
QScrollBar::handle:vertical { background: #252c3b; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QMessageBox, QInputDialog, QFileDialog { background: #141923; }
"""
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
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#252c3b"), 1))
        p.setBrush(QColor("#05070b"))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 14, 14)
        if self.pix is None:
            p.setPen(QColor("#8a93a6"))
            f = QFont()
            f.setPointSize(14)
            f.setBold(True)
            p.setFont(f)
            p.drawText(self.rect().adjusted(0, -20, 0, -20), Qt.AlignCenter,
                       "No picture yet")
            f.setPointSize(11)
            f.setBold(False)
            p.setFont(f)
            p.drawText(self.rect().adjusted(0, 20, 0, 20), Qt.AlignCenter,
                       "Add a camera or a recording in step 1.")
            return
        p.drawPixmap(0, 0, self.pix)
        if self.win.last_state.get("running"):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(8, 10, 16, 190))
            p.drawRoundedRect(12, 12, 70, 26, 8, 8)
            p.setBrush(QColor("#ff3b3b"))
            p.drawEllipse(QPointF(26, 25), 5, 5)
            p.setPen(QColor("white"))
            p.setFont(QFont("Helvetica", 11, QFont.Bold))
            p.drawText(QPointF(37, 30), "LIVE")
        cam = self.win.cam()
        k = self.scale
        live = self.win.last_state.get("live", {}).get("zones", {}) \
            if self.win.last_state else {}
        f = QFont("Helvetica", 11, QFont.Bold)
        p.setFont(f)
        for z in (cam or {}).get("zones", []):
            col = COL[z["hub"]]
            poly = QPolygonF([QPointF(x * k, y * k) for x, y in z["outline"]])
            fill = QColor(col)
            fill.setAlpha(40)
            p.setBrush(fill)
            p.setPen(QPen(col, 2.5))
            p.drawPolygon(poly)
            label = (f"{live[z['name']]}  " if z["name"] in live else "") + z["hub"].upper()
            x = min(q[0] for q in z["outline"]) * k
            y = min(q[1] for q in z["outline"]) * k - 28
            w = p.fontMetrics().horizontalAdvance(label) + 14
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(int(x), int(y), w, 22, 6, 6)
            p.setPen(QColor("white"))
            p.drawText(QPointF(x + 7, y + 16), label)
        p.setBrush(Qt.NoBrush)
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
        root.setObjectName("root")
        self.setCentralWidget(root)
        page = QVBoxLayout(root)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)

        top = QFrame()
        top.setObjectName("topbar")
        bar = QHBoxLayout(top)
        bar.setContentsMargins(18, 10, 18, 10)
        brand = QLabel("▲  Hub Counter")
        brand.setObjectName("brand")
        bar.addWidget(brand)
        bar.addSpacing(14)
        self.path_lbl = QLabel("New setup")
        self.path_lbl.setObjectName("file")
        bar.addWidget(self.path_lbl, 1)
        for text, fn in (("Open…", self.open_setup), ("Save", self.save),
                         ("Save as…", self.save_as)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        page.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(18, 16, 18, 18)
        body.setSpacing(16)
        page.addLayout(body, 1)
        left = QVBoxLayout()
        left.setSpacing(12)
        body.addLayout(left, 1)
        side = QWidget()
        side.setObjectName("side")
        right = QVBoxLayout(side)
        right.setContentsMargins(0, 0, 6, 0)
        right.setSpacing(12)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(410)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body.addWidget(scroll)

        score = QHBoxLayout()
        score.setSpacing(12)
        self.red, self.blue = QLabel("0"), QLabel("0")
        for num, key, title in ((self.red, "red", "RED HUB"),
                                (self.blue, "blue", "BLUE HUB")):
            tile = QFrame()
            tile.setObjectName(key)
            tile.setMinimumSize(200, 104)
            v = QVBoxLayout(tile)
            v.setContentsMargins(18, 12, 18, 8)
            lbl = QLabel(title)
            lbl.setObjectName("tileLbl")
            num.setObjectName("tileNum")
            v.addWidget(lbl)
            v.addWidget(num)
            score.addWidget(tile, 1)
        card = QFrame()
        card.setObjectName("card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 12, 16, 12)
        self.state_lbl = QLabel("●  Stopped")
        self.state_lbl.setObjectName("state")
        self.link = QLabel("Counts are sent to bioarena once you press Start.")
        self.link.setObjectName("sub")
        self.link.setWordWrap(True)
        v.addWidget(self.state_lbl)
        v.addWidget(self.link)
        score.addWidget(card, 1)
        left.addLayout(score)

        self.view = PictureView(self)
        left.addWidget(self.view, 1)
        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        left.addWidget(self.hint)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        left.addWidget(self.log)

        # 1. cameras
        g = QGroupBox("1   Cameras")
        self.steps = [g]
        v = QVBoxLayout(g)
        self.cams = QListWidget()
        self.cams.setMaximumHeight(90)
        self.cams.currentTextChanged.connect(self._pick_camera)
        v.addWidget(self.cams)
        row = QHBoxLayout()
        for text, fn in (("Find cameras", self.find_cameras),
                         ("Add video…", self.add_recording),
                         ("Add stream…", self.add_stream)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("Remove")
        b.clicked.connect(self.remove_camera)
        row.addWidget(b)
        row.addStretch(1)
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
        b.setObjectName("primary")
        b.clicked.connect(self.measure)
        row.addWidget(b)
        v.addLayout(row)
        row = QHBoxLayout()
        seek_lbl = QLabel("Recording: picture at second")
        seek_lbl.setObjectName("muted")
        row.addWidget(seek_lbl)
        self.seek = QLineEdit()
        self.seek.setPlaceholderText("1/3 in")
        self.seek.setMaximumWidth(70)
        row.addWidget(self.seek)
        row.addStretch(1)
        v.addLayout(row)
        right.addWidget(g)

        # 2. outlines
        g = QGroupBox("2   Hub outlines")
        self.steps.append(g)
        v = QVBoxLayout(g)
        self.zones = QListWidget()
        self.zones.setMaximumHeight(80)
        v.addWidget(self.zones)
        row = QHBoxLayout()
        for hub in ("red", "blue"):
            b = QPushButton(f"Draw {hub.upper()}")
            b.setObjectName(hub)
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
        note.setObjectName("muted")
        v.addWidget(note)
        right.addWidget(g)

        # 3. calibrate
        g = QGroupBox("3   Calibrate (optional)")
        v = QVBoxLayout(g)
        note = QLabel("Record this camera while balls go in, count them by "
                      "hand, then pick the recording.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        v.addWidget(note)
        b = QPushButton("Calibrate from recording…")
        b.clicked.connect(self.calibrate)
        v.addWidget(b)
        right.addWidget(g)

        # 4. run
        g = QGroupBox("4   Run")
        self.steps.append(g)
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
        # The theme spells out the disabled look: a style sheet hides Qt's
        # own, and START looked pressable while the feed ran.
        self.start_btn.setObjectName("start")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("stop")
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
        self.path_lbl.setText(os.path.basename(s["path"]) if s["path"] else "New setup")
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
        self.red.setText(str(v["counts"]["red"]))
        self.blue.setText(str(v["counts"]["blue"]))
        if not s["running"]:
            state, col = "Stopped", "#6f7a8f"
            sub = (s["problems"][0] if s["problems"] else
                   "Ready. Press Start to send counts to bioarena.")
        elif v["stale"]:
            state, col = "No picture", "#ef4444"
            sub = f"{', '.join(v['stale'])} stopped sending frames -- bioarena shows OFFLINE."
        elif v["linked"]:
            r = v["reply"] or {}
            delayed = "stream" in s["kinds"].values()
            state, col = "Connected to bioarena", "#f59e0b" if delayed else "#22c55e"
            sub = (f"{str(r.get('match_state', '')).replace('_', ' ').lower()} · "
                   f"{r.get('shift', '')} · {v['rtt_ms']} ms"
                   + (" · counting from a delayed stream" if delayed else ""))
        else:
            state, col = "Sending, no reply yet", "#f59e0b"
            sub = f"No answer from {v['target']}. Check the cable and address."
        cams = "   ".join(f"{n}  {c['fps']:.0f} fps" for n, c in v["cameras"].items())
        self.state_lbl.setText(f"<span style='color:{col}'>●</span>  {state}")
        self.link.setText(sub + (f"<br><span style='color:#6f7a8f'>{cams}</span>"
                                 if cams else ""))
        cams_ = s["cfg"]["cameras"]
        done = [bool(cams_), bool(cams_) and all(c["zones"] for c in cams_),
                bool(s["running"])]
        titles = ("Cameras", "Hub outlines", "Run")
        nums = ("1", "2", "4")
        for g, d, t, n in zip(self.steps, done, titles, nums):
            text = f"{'✓' if d else n}   {t}"
            if g.title() != text:
                g.setTitle(text)
                g.setProperty("done", d)
                g.style().unpolish(g)
                g.style().polish(g)

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

    def add_stream(self) -> None:
        from .hubapp import STREAM_WARNING
        url, ok = QInputDialog.getText(
            self, "Add a live stream",
            "Twitch or YouTube live address:\n\n" + STREAM_WARNING)
        url = url.strip()
        if ok and url:
            cam = self._call(lambda: self.ctl.add_camera(url))
            if cam:
                self.current = cam["name"]
                self.hint.setText("Connecting to the stream…")
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
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = HubWindow(HubController(setup_path))
    win.resize(1400, 900)
    win.show()
    return app.exec()
