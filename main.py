import sys
import tempfile
import os
import math
import numpy as np
from PIL import Image as PILImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QComboBox,
    QProgressBar, QScrollArea,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QEvent, QSettings

# ---------------------------------------------------------------------------
# Model / weights metadata
# ---------------------------------------------------------------------------

MODELS = {
    "CSRNet": {
        "description": (
            "Dilated convolutional network. Fast, reliable, well-tested baseline "
            "for dense crowd counting."
        ),
        "weights": ["SHA", "SHB"],
        "default_weight": "SHA",
    },
    "SFANet": {
        "description": (
            "Scale-free attention network. Handles images where crowd density "
            "varies significantly across the frame."
        ),
        "weights": ["SHB"],
        "default_weight": "SHB",
    },
    "Bay": {
        "description": (
            "Bayesian loss model. Achieves high accuracy by optimising the full "
            "density-map distribution rather than just the count."
        ),
        "weights": ["SHA", "SHB", "QNRF"],
        "default_weight": "SHA",
    },
    "DM-Count": {
        "description": (
            "Distribution-matching count model. State-of-the-art accuracy; best "
            "choice when count precision matters most."
        ),
        "weights": ["SHA", "SHB", "QNRF"],
        "default_weight": "SHA",
    },
}

WEIGHTS_INFO = {
    "SHA": (
        "ShanghaiTech Part A — extremely dense crowds (~300 people / image). "
        "Recommended for packed concert or stadium photos."
    ),
    "SHB": (
        "ShanghaiTech Part B — sparser crowds (~120 people / image). "
        "Good for street scenes and moderate gatherings."
    ),
    "QNRF": (
        "UCF-QNRF — large dataset with extreme density variation. "
        "Most versatile; recommended when scene type is unknown."
    ),
}

DEFAULT_MODEL = "CSRNet"
ZOOM_STEP = 1.07
ZOOM_MIN = 0.05
ZOOM_MAX = 16.0
ARROW_SCROLL_PX = 60
OVERLAY_ALPHA = 0.35

# ---------------------------------------------------------------------------
# Background worker — tile-based counting
# ---------------------------------------------------------------------------

class CountWorker(QThread):
    # (tile_idx, total_tiles, tile_count, density_array, x, y, w, h)
    tile_done = pyqtSignal(int, int, float, object, int, int, int, int)
    finished = pyqtSignal(float)   # total count
    failed = pyqtSignal(str)

    def __init__(self, img_path: str, model_name: str, model_weights: str,
                 max_tile_dim: int | None):
        super().__init__()
        self.img_path = img_path
        self.model_name = model_name
        self.model_weights = model_weights
        self.max_tile_dim = max_tile_dim   # None = no tiling (single block)
        self.tmp_path: str | None = None

    def run(self):
        try:
            self._patch_lwcc()
            from lwcc import LWCC

            img = PILImage.open(self.img_path).convert("RGB")
            tiles = self._compute_tiles(*img.size)
            total = len(tiles)
            total_count = 0.0

            for idx, (tx, ty, tw, th) in enumerate(tiles):
                tile_img = img.crop((tx, ty, tx + tw, ty + th))

                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                self.tmp_path = tmp.name
                tmp.close()
                tile_img.save(self.tmp_path)

                count, density = LWCC.get_count(
                    self.tmp_path,
                    model_name=self.model_name,
                    model_weights=self.model_weights,
                    return_density=True,
                    resize_img=False,
                )
                total_count += float(count)
                self.tile_done.emit(idx, total, float(count), density,
                                    tx, ty, tw, th)
                self._cleanup_tmp()

            self.finished.emit(total_count)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._cleanup_tmp()

    # -- helpers --

    def _patch_lwcc(self):
        import lwcc.util.functions as _lwcc_fn
        from pathlib import Path
        import gdown as _gdown

        def _weights_check_fixed(model_name, model_weights):
            weights_dir = Path.home() / ".lwcc" / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{model_name}_{model_weights}.pth"
            output = str(weights_dir / file_name)
            if not os.path.isfile(output):
                url = _lwcc_fn.build_url(file_name)
                _gdown.download(url, output, quiet=False)
            return output

        _lwcc_fn.weights_check = _weights_check_fixed

    def _compute_tiles(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        md = self.max_tile_dim
        if md is None or max(width, height) <= md:
            return [(0, 0, width, height)]

        cols = math.ceil(width / md)
        rows = math.ceil(height / md)
        tile_w = math.ceil(width / cols)
        tile_h = math.ceil(height / rows)

        tiles = []
        for r in range(rows):
            for c in range(cols):
                x = c * tile_w
                y = r * tile_h
                w = min(tile_w, width - x)
                h = min(tile_h, height - y)
                tiles.append((x, y, w, h))
        return tiles

    def _cleanup_tmp(self):
        if self.tmp_path and os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)
            self.tmp_path = None

# ---------------------------------------------------------------------------
# Event filter: wheel-zoom, drag-pan, click detection
# ---------------------------------------------------------------------------

class ImageEventFilter(QObject):
    zoomed = pyqtSignal(int)
    image_clicked = pyqtSignal()

    def __init__(self, scroll_area: QScrollArea):
        super().__init__()
        self._scroll = scroll_area
        self._drag_start = None
        self._dragged = False
        self._hbar_start = 0
        self._vbar_start = 0

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        try:
            etype = event.type()

            # Reset drag state if widget loses focus mid-drag
            if etype == QEvent.Type.FocusOut:
                if self._drag_start is not None:
                    self._drag_start = None
                    self._dragged = False
                    try:
                        obj.setCursor(Qt.CursorShape.OpenHandCursor)
                    except RuntimeError:
                        pass
                return False

            if etype == QEvent.Type.Wheel:
                self.zoomed.emit(1 if event.angleDelta().y() > 0 else -1)
                return True

            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_start = event.globalPosition().toPoint()
                self._hbar_start = self._scroll.horizontalScrollBar().value()
                self._vbar_start = self._scroll.verticalScrollBar().value()
                self._dragged = False
                obj.setCursor(Qt.CursorShape.ClosedHandCursor)
                return True

            if etype == QEvent.Type.MouseMove and self._drag_start is not None:
                delta = event.globalPosition().toPoint() - self._drag_start
                if not self._dragged and (abs(delta.x()) > 4 or abs(delta.y()) > 4):
                    self._dragged = True
                if self._dragged:
                    self._scroll.horizontalScrollBar().setValue(self._hbar_start - delta.x())
                    self._scroll.verticalScrollBar().setValue(self._vbar_start - delta.y())
                return True

            if etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                was_drag = self._dragged
                self._drag_start = None
                self._dragged = False
                obj.setCursor(Qt.CursorShape.OpenHandCursor)
                if not was_drag:
                    self.image_clicked.emit()
                return True

        except Exception:
            # Never let a Python exception escape into SIP/Qt — it calls
            # qFatal() → abort(), crashing the entire application.
            self._drag_start = None
            self._dragged = False
            return False

        return False

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def jet_colormap(data: np.ndarray) -> np.ndarray:
    r = np.interp(data, [0.0, 0.35, 0.66, 0.89, 1.0], [0.0, 0.0, 1.0, 1.0, 0.5])
    g = np.interp(data, [0.0, 0.125, 0.375, 0.64, 0.91, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    b = np.interp(data, [0.0, 0.11, 0.34, 0.65, 1.0], [0.5, 1.0, 1.0, 0.0, 0.0])
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def blend_tile_into(wip: PILImage.Image, orig: PILImage.Image,
                    density: np.ndarray, x: int, y: int, w: int, h: int):
    """Blend one tile's density heatmap into the working overlay image in-place."""
    d = density.astype(float)
    if d.max() > 0:
        d /= d.max()
    coloured = PILImage.fromarray(jet_colormap(d), mode="RGB")
    coloured = coloured.resize((w, h), PILImage.LANCZOS)
    orig_tile = orig.crop((x, y, x + w, y + h)).convert("RGB")
    blended = PILImage.blend(orig_tile, coloured, alpha=OVERLAY_ALPHA)
    wip.paste(blended, (x, y))


def pil_to_qpixmap(img: PILImage.Image) -> QPixmap:
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    raw = img_rgb.tobytes("raw", "RGB")
    # .copy() is critical: QImage does not own `raw`, so without copy the
    # QPixmap would reference freed memory once `raw` is garbage-collected.
    qimg = QImage(raw, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

LABEL_STYLE = "color: #aaaaaa; font-size: 11px; padding-top: 4px;"
DESC_STYLE = "color: #777777; font-size: 10px; font-style: italic; padding: 2px 0 6px 0;"
COMBO_STYLE = """
    QComboBox {
        background-color: #3a3a3a; color: #dddddd;
        border: 1px solid #555555; border-radius: 3px;
        padding: 4px 6px; font-size: 11px;
    }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background-color: #3a3a3a; color: #dddddd;
        selection-background-color: #3c8dbc;
    }
"""
ICON_BTN_STYLE = """
    QPushButton {
        background-color: #3a3a3a; color: #cccccc;
        border: 1px solid #555555; border-radius: 3px;
        font-size: 14px; padding: 2px 8px;
   }
    QPushButton:hover { background-color: #4a4a4a; }
    QPushButton:pressed { background-color: #2a2a2a; }
    QPushButton:disabled { color: #555555; }
"""

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer — Crowd Counter")
        self.resize(1100, 720)

        self._settings = QSettings("CrowdCounter", "ImageViewer")

        self._img_path: str | None = None
        self._pil_image: PILImage.Image | None = None
        self._current_pixmap: QPixmap | None = None
        self._overlay_active: bool = False
        self._worker: CountWorker | None = None

        # Progressive overlay state
        self._wip_image: PILImage.Image | None = None
        self._running_count: float = 0.0

        # Zoom state
        self._zoom_fit: bool = True
        self._zoom_factor: float = 1.0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._make_sidebar())
        root.addWidget(self._make_image_area(), stretch=1)

    # ---- Sidebar ----

    def _make_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background-color: #2d2d2d;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Tools")
        title.setStyleSheet("color: #cccccc; font-size: 14px; font-weight: bold; padding-bottom: 6px;")
        layout.addWidget(title)
        layout.addWidget(self._hline())

        open_btn = QPushButton("Open Image")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setStyleSheet(self._btn_style("#3c8dbc", "#4fa3d1", "#2e6f99"))
        open_btn.clicked.connect(self._open_image)
        layout.addWidget(open_btn)

        layout.addSpacing(14)

        section = QLabel("Crowd Counting")
        section.setStyleSheet("color: #cccccc; font-size: 12px; font-weight: bold; padding-top: 4px;")
        layout.addWidget(section)
        layout.addWidget(self._hline())

        layout.addWidget(self._small_label("Model"))
        self._model_combo = QComboBox()
        self._model_combo.setStyleSheet(COMBO_STYLE)
        self._model_combo.addItems(list(MODELS.keys()))
        self._model_combo.setCurrentText(DEFAULT_MODEL)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self._model_combo)
        self._model_desc = QLabel()
        self._model_desc.setWordWrap(True)
        self._model_desc.setStyleSheet(DESC_STYLE)
        layout.addWidget(self._model_desc)

        layout.addWidget(self._small_label("Weights"))
        self._weights_combo = QComboBox()
        self._weights_combo.setStyleSheet(COMBO_STYLE)
        self._weights_combo.currentTextChanged.connect(self._on_weights_changed)
        layout.addWidget(self._weights_combo)
        self._weights_desc = QLabel()
        self._weights_desc.setWordWrap(True)
        self._weights_desc.setStyleSheet(DESC_STYLE)
        layout.addWidget(self._weights_desc)

        self._on_model_changed(DEFAULT_MODEL)

        layout.addSpacing(10)

        self._count_btn = QPushButton("Count People")
        self._count_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._count_btn.setEnabled(False)
        self._count_btn.setStyleSheet(self._btn_style("#27a745", "#34c95a", "#1e8035"))
        self._count_btn.clicked.connect(self._run_count)
        layout.addWidget(self._count_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setStyleSheet(self._btn_style("#c0392b", "#e74c3c", "#922b21"))
        self._stop_btn.clicked.connect(self._stop_count)
        self._stop_btn.hide()
        layout.addWidget(self._stop_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setStyleSheet("""
            QProgressBar { background-color: #1a1a1a; border: none; border-radius: 3px; }
            QProgressBar::chunk { background-color: #3c8dbc; border-radius: 3px; }
        """)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_label.setStyleSheet(
            "color: #f0f0f0; font-size: 13px; font-weight: bold; "
            "background-color: #1a1a1a; border-radius: 4px; padding: 8px; margin-top: 6px;"
        )
        self._result_label.hide()
        layout.addWidget(self._result_label)

        self._hint_label = QLabel("Click image to remove overlay")
        self._hint_label.setWordWrap(True)
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setStyleSheet("color: #888888; font-size: 10px; font-style: italic; padding-top: 4px;")
        self._hint_label.hide()
        layout.addWidget(self._hint_label)

        layout.addStretch()
        return sidebar

    # ---- Image area ----

    def _make_image_area(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background-color: #121212;")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet("background-color: #121212; border: none;")
        self._scroll.setWidgetResizable(False)
        self._scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #121212; color: #555555; font-size: 16px;")
        self._image_label.setText("Open an image to get started")
        self._scroll.setWidget(self._image_label)

        self._evt_filter = ImageEventFilter(self._scroll)
        self._evt_filter.zoomed.connect(self._on_wheel_zoom)
        self._evt_filter.image_clicked.connect(self._on_image_clicked)
        self._image_label.installEventFilter(self._evt_filter)

        vbox.addWidget(self._scroll, stretch=1)
        vbox.addWidget(self._make_zoom_toolbar())
        return container

    def _make_zoom_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background-color: #1e1e1e; border-top: 1px solid #333333;")
        bar.setFixedHeight(36)
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(10, 0, 10, 0)
        hbox.setSpacing(4)
        hbox.addStretch()

        zoom_out = QPushButton("\u2212")
        zoom_out.setFixedSize(28, 24)
        zoom_out.setStyleSheet(ICON_BTN_STYLE)
        zoom_out.setCursor(Qt.CursorShape.PointingHandCursor)
        zoom_out.clicked.connect(lambda: self._zoom(1 / ZOOM_STEP))

        self._zoom_label = QLabel("Fit")
        self._zoom_label.setStyleSheet("color: #888888; font-size: 11px;")
        self._zoom_label.setFixedWidth(44)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        zoom_in = QPushButton("+")
        zoom_in.setFixedSize(28, 24)
        zoom_in.setStyleSheet(ICON_BTN_STYLE)
        zoom_in.setCursor(Qt.CursorShape.PointingHandCursor)
        zoom_in.clicked.connect(lambda: self._zoom(ZOOM_STEP))

        fit_btn = QPushButton("Fit")
        fit_btn.setFixedHeight(24)
        fit_btn.setStyleSheet(ICON_BTN_STYLE)
        fit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        fit_btn.clicked.connect(self._zoom_reset)

        hbox.addWidget(zoom_out)
        hbox.addWidget(self._zoom_label)
        hbox.addWidget(zoom_in)
        hbox.addSpacing(6)
        hbox.addWidget(fit_btn)
        hbox.addStretch()
        return bar

    # ------------------------------------------------------------------
    # Dropdown callbacks
    # ------------------------------------------------------------------

    def _on_model_changed(self, model_name: str):
        info = MODELS[model_name]
        self._model_desc.setText(info["description"])
        self._weights_combo.blockSignals(True)
        self._weights_combo.clear()
        self._weights_combo.addItems(info["weights"])
        self._weights_combo.setCurrentText(info["default_weight"])
        self._weights_combo.blockSignals(False)
        self._on_weights_changed(info["default_weight"])

    def _on_weights_changed(self, weight: str):
        self._weights_desc.setText(WEIGHTS_INFO.get(weight, ""))

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _open_image(self):
        if self._worker and self._worker.isRunning():
            self._stop_count()

        last_dir = self._settings.value("last_open_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", last_dir,
            "Image files (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.tiff);;All files (*)"
        )
        if not path:
            return

        self._settings.setValue("last_open_dir", os.path.dirname(path))

        self._img_path = path
        self._pil_image = PILImage.open(path)
        self._current_pixmap = pil_to_qpixmap(self._pil_image)
        self._overlay_active = False
        self._wip_image = None
        self._image_label.setText("")
        self._image_label.setCursor(Qt.CursorShape.OpenHandCursor)

        self._result_label.hide()
        self._hint_label.hide()
        self._count_btn.setEnabled(True)
        self._count_btn.setText("Count People")

        self._zoom_reset()
        self._scroll.setFocus()

    # ------------------------------------------------------------------
    # Counting (tile-based)
    # ------------------------------------------------------------------

    def _run_count(self):
        if not self._img_path:
            return
        if self._overlay_active:
            self._reset_to_original()
            return

        self._count_btn.setEnabled(False)
        self._count_btn.hide()
        self._stop_btn.show()
        self._result_label.hide()
        self._hint_label.hide()

        # Prepare progressive overlay: start with a copy of the original
        self._wip_image = self._pil_image.convert("RGB").copy()
        self._running_count = 0.0

        # Start indeterminate until first tile reports back with total
        self._progress_bar.setRange(0, 0)
        self._progress_bar.show()

        model_name = self._model_combo.currentText()
        weights = self._weights_combo.currentText()

        self._worker = CountWorker(self._img_path, model_name, weights,
                                   max_tile_dim=1000)
        self._worker.tile_done.connect(self._on_tile_done)
        self._worker.finished.connect(self._on_count_finished)
        self._worker.failed.connect(self._on_count_error)
        self._worker.start()

    def _stop_count(self):
        if self._worker and self._worker.isRunning():
            self._worker.tile_done.disconnect()
            self._worker.finished.disconnect()
            self._worker.failed.disconnect()
            self._worker.terminate()
            self._worker.wait()
            self._worker._cleanup_tmp()
            self._worker = None
        self._wip_image = None
        self._reset_counting_ui()

    def _on_tile_done(self, idx: int, total: int, count: float,
                      density: np.ndarray, x: int, y: int, w: int, h: int):
        # Switch to determinate progress on first tile
        if idx == 0:
            self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(idx + 1)

        self._running_count += count

        # Blend this tile into the working image
        blend_tile_into(self._wip_image, self._pil_image, density, x, y, w, h)

        # Update live display
        self._current_pixmap = pil_to_qpixmap(self._wip_image)
        self._update_display()

        # Show running count
        self._result_label.setText(
            f"Counting... {self._running_count:.1f}\n"
            f"({idx + 1}/{total} tiles)"
        )
        self._result_label.show()

    def _on_count_finished(self, total_count: float):
        self._worker = None
        self._reset_counting_ui()
        self._result_label.setText(f"Estimated count:\n{total_count:.1f}")
        self._result_label.show()
        self._overlay_active = True
        self._hint_label.show()

    def _on_count_error(self, msg: str):
        self._worker = None
        self._wip_image = None
        self._reset_counting_ui()
        self._result_label.setText(f"Error:\n{msg}")
        self._result_label.show()

    def _reset_counting_ui(self):
        self._progress_bar.hide()
        self._progress_bar.setRange(0, 0)   # reset to indeterminate for next run
        self._stop_btn.hide()
        self._count_btn.setText("Count People")
        self._count_btn.setEnabled(self._img_path is not None)
        self._count_btn.show()

    # ------------------------------------------------------------------
    # Overlay reset
    # ------------------------------------------------------------------

    def _on_image_clicked(self):
        if self._overlay_active:
            self._reset_to_original()

    def _reset_to_original(self):
        self._overlay_active = False
        self._wip_image = None
        self._current_pixmap = pil_to_qpixmap(self._pil_image)
        self._hint_label.hide()
        self._update_display()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _zoom(self, factor: float):
        if self._current_pixmap is None:
            return
        if self._zoom_fit:
            vp = self._scroll.viewport().size()
            px = self._current_pixmap
            if px.width() == 0 or px.height() == 0 or vp.width() == 0 or vp.height() == 0:
                return
            self._zoom_factor = min(vp.width() / px.width(), vp.height() / px.height())
            self._zoom_fit = False
        self._zoom_factor = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom_factor * factor))
        self._update_display()

    def _zoom_reset(self):
        self._zoom_fit = True
        self._update_display()

    def _on_wheel_zoom(self, direction: int):
        self._zoom(ZOOM_STEP if direction > 0 else 1 / ZOOM_STEP)

    # ------------------------------------------------------------------
    # Arrow-key panning
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        if key == Qt.Key.Key_Left:
            hbar.setValue(hbar.value() - ARROW_SCROLL_PX)
        elif key == Qt.Key.Key_Right:
            hbar.setValue(hbar.value() + ARROW_SCROLL_PX)
        elif key == Qt.Key.Key_Up:
            vbar.setValue(vbar.value() - ARROW_SCROLL_PX)
        elif key == Qt.Key.Key_Down:
            vbar.setValue(vbar.value() + ARROW_SCROLL_PX)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _update_display(self):
        if self._current_pixmap is None:
            return

        vp = self._scroll.viewport().size()
        px = self._current_pixmap

        if self._zoom_fit:
            scaled = px.scaled(vp, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            self._image_label.setPixmap(scaled)
            self._image_label.resize(vp)
            self._zoom_label.setText("Fit")
        else:
            tw = int(px.width() * self._zoom_factor)
            th = int(px.height() * self._zoom_factor)
            scaled = px.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            self._image_label.setPixmap(scaled)
            self._image_label.resize(max(tw, vp.width()), max(th, vp.height()))
            self._zoom_label.setText(f"{int(self._zoom_factor * 100)}%")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom_fit and self._current_pixmap:
            self._update_display()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
            self._worker._cleanup_tmp()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("color: #444444; margin-bottom: 4px;")
        return f

    def _small_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(LABEL_STYLE)
        return lbl

    @staticmethod
    def _btn_style(normal: str, hover: str, pressed: str) -> str:
        return f"""
            QPushButton {{
                background-color: {normal}; color: white;
                border: none; padding: 8px;
                border-radius: 4px; font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:pressed {{ background-color: {pressed}; }}
            QPushButton:disabled {{ background-color: #555555; color: #888888; }}
        """


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageViewer()
    window.show()
    sys.exit(app.exec())
