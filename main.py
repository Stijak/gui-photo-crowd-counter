import sys
import tempfile
import os
import numpy as np
from PIL import Image as PILImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy, QComboBox,
    QProgressBar,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal

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

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class CountWorker(QThread):
    finished = pyqtSignal(float, object)   # count, density_map (numpy 2-D)
    failed = pyqtSignal(str)

    def __init__(self, img_path: str, model_name: str, model_weights: str, max_dim: int | None):
        super().__init__()
        self.img_path = img_path
        self.model_name = model_name
        self.model_weights = model_weights
        self.max_dim = max_dim  # None = full resolution

    def run(self):
        tmp_path = None
        try:
            # Patch lwcc's broken weights path (uses /.lwcc instead of ~/.lwcc)
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

            # Resize image to max_dim before inference to control memory usage.
            # We write a temp PNG so lwcc always receives a pre-scaled image
            # and we always pass resize_img=False to prevent double-scaling.
            infer_path = self.img_path
            img = PILImage.open(self.img_path).convert("RGB")
            w, h = img.size
            if self.max_dim is not None and max(w, h) > self.max_dim:
                scale = self.max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_path = tmp.name
                tmp.close()
                img.save(tmp_path)
                infer_path = tmp_path

            from lwcc import LWCC
            count, density = LWCC.get_count(
                infer_path,
                model_name=self.model_name,
                model_weights=self.model_weights,
                return_density=True,
                resize_img=False,
            )
            self.finished.emit(float(count), density)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Clickable image label
# ---------------------------------------------------------------------------

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jet_colormap(data: np.ndarray) -> np.ndarray:
    """Apply a jet colormap to a 2-D float array normalised to [0, 1].
    Returns an (H, W, 3) uint8 array."""
    r = np.interp(data, [0.0, 0.35, 0.66, 0.89, 1.0], [0.0, 0.0, 1.0, 1.0, 0.5])
    g = np.interp(data, [0.0, 0.125, 0.375, 0.64, 0.91, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    b = np.interp(data, [0.0, 0.11, 0.34, 0.65, 1.0], [0.5, 1.0, 1.0, 0.0, 0.0])
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def blend_density(original: PILImage.Image, density: np.ndarray, alpha: float = 0.5) -> QPixmap:
    """Overlay a density map on top of the original PIL image."""
    orig_w, orig_h = original.size

    # Normalise density
    d = density.astype(float)
    if d.max() > 0:
        d /= d.max()

    # Colourmap → PIL
    coloured = PILImage.fromarray(jet_colormap(d), mode="RGB")
    coloured = coloured.resize((orig_w, orig_h), PILImage.LANCZOS)

    # Composite
    base = original.convert("RGB")
    blended = PILImage.blend(base, coloured, alpha=alpha)

    # PIL → QPixmap
    data = blended.tobytes("raw", "RGB")
    qimg = QImage(data, orig_w, orig_h, orig_w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def pil_to_qpixmap(img: PILImage.Image) -> QPixmap:
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    data = img_rgb.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ---------------------------------------------------------------------------
# Sidebar styling helpers
# ---------------------------------------------------------------------------

LABEL_STYLE = "color: #aaaaaa; font-size: 11px; padding-top: 4px;"
DESC_STYLE = "color: #777777; font-size: 10px; font-style: italic; padding: 2px 0 6px 0;"
COMBO_STYLE = """
    QComboBox {
        background-color: #3a3a3a;
        color: #dddddd;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 4px 6px;
        font-size: 11px;
    }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background-color: #3a3a3a;
        color: #dddddd;
        selection-background-color: #3c8dbc;
    }
"""

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer — Crowd Counter")
        self.resize(1100, 720)

        self._img_path: str | None = None
        self._pil_image: PILImage.Image | None = None
        self._current_pixmap: QPixmap | None = None   # what is shown (original or overlay)
        self._overlay_active: bool = False
        self._worker: CountWorker | None = None

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

    def _make_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background-color: #2d2d2d;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Title
        title = QLabel("Tools")
        title.setStyleSheet("color: #cccccc; font-size: 14px; font-weight: bold; padding-bottom: 6px;")
        layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444444; margin-bottom: 8px;")
        layout.addWidget(sep)

        # Open button
        open_btn = QPushButton("Open Image")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setStyleSheet(self._btn_style("#3c8dbc", "#4fa3d1", "#2e6f99"))
        open_btn.clicked.connect(self._open_image)
        layout.addWidget(open_btn)

        layout.addSpacing(14)

        # ----- Crowd counting section -----
        section = QLabel("Crowd Counting")
        section.setStyleSheet("color: #cccccc; font-size: 12px; font-weight: bold; padding-top: 4px;")
        layout.addWidget(section)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #444444; margin-bottom: 4px;")
        layout.addWidget(sep2)

        # Model dropdown
        model_label = QLabel("Model")
        model_label.setStyleSheet(LABEL_STYLE)
        layout.addWidget(model_label)

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

        # Weights dropdown
        weights_label = QLabel("Weights")
        weights_label.setStyleSheet(LABEL_STYLE)
        layout.addWidget(weights_label)

        self._weights_combo = QComboBox()
        self._weights_combo.setStyleSheet(COMBO_STYLE)
        self._weights_combo.currentTextChanged.connect(self._on_weights_changed)
        layout.addWidget(self._weights_combo)

        self._weights_desc = QLabel()
        self._weights_desc.setWordWrap(True)
        self._weights_desc.setStyleSheet(DESC_STYLE)
        layout.addWidget(self._weights_desc)

        # Populate with defaults
        self._on_model_changed(DEFAULT_MODEL)

        layout.addSpacing(6)

        # Max resolution dropdown
        res_label = QLabel("Max Resolution")
        res_label.setStyleSheet(LABEL_STYLE)
        layout.addWidget(res_label)

        self._res_combo = QComboBox()
        self._res_combo.setStyleSheet(COMBO_STYLE)
        self._res_combo.addItems(["1000 px", "2000 px", "3000 px", "4000 px", "Full"])
        self._res_combo.setCurrentText("2000 px")
        layout.addWidget(self._res_combo)

        self._res_desc = QLabel(
            "Longest edge limit before inference. Lower = faster and less memory. "
            "Full resolution may use very large amounts of RAM."
        )
        self._res_desc.setWordWrap(True)
        self._res_desc.setStyleSheet(DESC_STYLE)
        layout.addWidget(self._res_desc)

        layout.addSpacing(10)

        # Count button
        self._count_btn = QPushButton("Count People")
        self._count_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._count_btn.setEnabled(False)
        self._count_btn.setStyleSheet(self._btn_style("#27a745", "#34c95a", "#1e8035"))
        self._count_btn.clicked.connect(self._run_count)
        layout.addWidget(self._count_btn)

        # Progress bar (indeterminate, shown only while counting)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate / pulsing
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #3c8dbc;
                border-radius: 3px;
            }
        """)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        # Result display
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_label.setStyleSheet(
            "color: #f0f0f0; font-size: 13px; font-weight: bold; "
            "background-color: #1a1a1a; border-radius: 4px; padding: 8px; margin-top: 6px;"
        )
        self._result_label.hide()
        layout.addWidget(self._result_label)

        # Hint label (shown when overlay is active)
        self._hint_label = QLabel("Click image to remove overlay")
        self._hint_label.setWordWrap(True)
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setStyleSheet("color: #888888; font-size: 10px; font-style: italic; padding-top: 4px;")
        self._hint_label.hide()
        layout.addWidget(self._hint_label)

        # Spacer for future controls
        layout.addStretch()

        return sidebar

    def _make_image_area(self) -> ClickableLabel:
        self._image_label = ClickableLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_label.setStyleSheet("background-color: #121212; color: #555555; font-size: 16px;")
        self._image_label.setText("Open an image to get started")
        self._image_label.clicked.connect(self._on_image_clicked)
        return self._image_label

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
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image files (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.tiff);;All files (*)"
        )
        if not path:
            return

        self._img_path = path
        self._pil_image = PILImage.open(path)
        self._current_pixmap = pil_to_qpixmap(self._pil_image)
        self._overlay_active = False
        self._image_label.setText("")

        self._result_label.hide()
        self._hint_label.hide()
        self._count_btn.setEnabled(True)
        self._count_btn.setText("Count People")

        self._fit_pixmap(self._current_pixmap)

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    def _run_count(self):
        if not self._img_path:
            return

        # If overlay is active, clicking Count resets first
        if self._overlay_active:
            self._reset_to_original()
            return

        self._count_btn.setEnabled(False)
        self._count_btn.setText("Counting…")
        self._result_label.hide()
        self._hint_label.hide()
        self._progress_bar.show()

        model_name = self._model_combo.currentText()
        weights = self._weights_combo.currentText()
        res_text = self._res_combo.currentText()
        max_dim = None if res_text == "Full" else int(res_text.split()[0])

        self._worker = CountWorker(self._img_path, model_name, weights, max_dim)
        self._worker.finished.connect(self._on_count_done)
        self._worker.failed.connect(self._on_count_error)
        self._worker.start()

    def _on_count_done(self, count: float, density: np.ndarray):
        self._progress_bar.hide()
        self._count_btn.setText("Count People")
        self._count_btn.setEnabled(True)

        self._result_label.setText(f"Estimated count:\n{count:.1f}")
        self._result_label.show()

        # Build overlay pixmap and show it
        overlay_px = blend_density(self._pil_image, density, alpha=0.55)
        self._current_pixmap = overlay_px
        self._overlay_active = True
        self._hint_label.show()
        self._fit_pixmap(overlay_px)

    def _on_count_error(self, msg: str):
        self._progress_bar.hide()
        self._count_btn.setText("Count People")
        self._count_btn.setEnabled(True)
        self._result_label.setText(f"Error:\n{msg}")
        self._result_label.show()

    # ------------------------------------------------------------------
    # Reset overlay
    # ------------------------------------------------------------------

    def _on_image_clicked(self):
        if self._overlay_active:
            self._reset_to_original()

    def _reset_to_original(self):
        self._overlay_active = False
        self._current_pixmap = pil_to_qpixmap(self._pil_image)
        self._hint_label.hide()
        self._count_btn.setText("Count People")
        self._fit_pixmap(self._current_pixmap)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _fit_pixmap(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_pixmap:
            self._fit_pixmap(self._current_pixmap)

    # ------------------------------------------------------------------
    # Style helper
    # ------------------------------------------------------------------

    @staticmethod
    def _btn_style(normal: str, hover: str, pressed: str) -> str:
        return f"""
            QPushButton {{
                background-color: {normal};
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
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
