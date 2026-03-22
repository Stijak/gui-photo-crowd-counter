# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Code

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main script
python main.py
```

## Environment

- Python 3.14 (virtual environment at `.venv/`)
- Black formatter configured (via PyCharm IDE settings)
- macOS (Homebrew Python — tkinter is NOT available, use PyQt6 for GUI)

## Dependencies

- **PyQt6** — GUI framework (replaced tkinter due to missing Tk support in Homebrew Python)
- **Pillow** — image loading and manipulation
- **numpy** — array operations for density maps
- **lwcc** — crowd counting models (pulls in PyTorch, torchvision, gdown)

All dependencies listed in `requirements.txt`.

## Architecture

Single-file PyQt6 app (`main.py`) with these key components:

- **`ImageViewer`** (QMainWindow) — main window, owns all UI and state
- **`CountWorker`** (QThread) — runs tile-based lwcc inference off the main thread (hardcoded 1000px max tile dimension)
- **`ImageEventFilter`** (QObject) — event filter handling wheel-zoom, drag-to-pan, and click detection
- **Helper functions** — `jet_colormap`, `blend_tile_into`, `pil_to_qpixmap` for image processing

## Known Issues / Gotchas

- **lwcc has a broken cache path** — it tries to write to `/.lwcc` (filesystem root). The `CountWorker` monkey-patches `lwcc.util.functions.weights_check` to use `~/.lwcc` instead.
- **QImage does not own pixel data** — when converting PIL → QPixmap, always call `.copy()` on the QImage before creating the QPixmap, or the buffer will be freed by Python's GC.
- **High-res images can exhaust memory** — lwcc with `resize_img=False` on a 30+ MP image can use 50+ GB. The app automatically splits images into tiles (max 1000px per dimension) before passing to lwcc.
- **Python exceptions in eventFilter crash the app** — PyQt6/SIP calls `qFatal()` → `abort()` if a Python exception escapes `eventFilter`. The filter body is wrapped in `try/except` to prevent this.

## Project State

Desktop image viewer with crowd counting functionality. The sidebar has space reserved for adding future tools/features.
