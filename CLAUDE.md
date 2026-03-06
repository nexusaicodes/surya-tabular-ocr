# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trimmed fork of Surya focused on **English-only OCR, layout analysis, and table recognition** from document screenshots. Math/LaTeX, multilingual support, Streamlit UIs, and OCR error detection have been removed.

## Build & Development Commands

```bash
# Install (requires Python 3.10+, PyTorch 2.7+)
poetry install              # main + dev deps
poetry install --with dev   # dev dependencies
poetry shell                # activate venv

# Run tests
pytest                      # all tests
pytest tests/test_recognition.py  # single test file
pytest tests/test_recognition.py::test_name  # single test

# CLI tools (after install)
surya_ocr DATA_PATH         # OCR: detect text lines + read text
surya_layout DATA_PATH      # Layout: detect tables, headers, sections, etc.
surya_table DATA_PATH       # Table recognition: rows, columns, cells

# Benchmarks
python benchmark/detection.py --max_rows 256
python benchmark/recognition.py
python benchmark/layout.py
python benchmark/table_recognition.py --max_rows 1024
```

### CLI common flags
All CLI tools accept: `INPUT_PATH` (required), `--output_dir PATH`, `--page_range TEXT` (e.g. `0,5-10,20`), `--images` (save debug images), `-d/--debug`.

`surya_ocr` additionally accepts `--task_name` (default: `ocr_with_boxes`, alternatives: `ocr_without_boxes`, `block_without_boxes`).

`surya_table` additionally accepts `--detect_boxes` and `--skip_table_detection`.

## Architecture

### Predictor Pattern
All models follow a predictor pattern inheriting from `surya/common/predictor.py:BasePredictor`:
- `BasePredictor` handles model/processor loading, batch size selection, device management
- Each predictor's `__call__` is the main entry point

### Model Hierarchy
- **FoundationPredictor** (`surya/foundation/__init__.py`) — Core autoregressive model shared by OCR, layout, and table structure tasks. Uses continuous batching with a KV cache, beacon tokens, and multi-token prediction. This is the most complex component.
- **RecognitionPredictor** (`surya/recognition/__init__.py`) — Wraps FoundationPredictor. Detects text lines (via DetectionPredictor internally), slices image regions, runs OCR, and reassembles results with character-level bounding boxes. Math mode is disabled.
- **LayoutPredictor** (`surya/layout/__init__.py`) — Wraps FoundationPredictor for layout analysis + reading order. Labels: Table, SectionHeader, Text, Picture, Figure, Caption, ListItem, Form, PageHeader, PageFooter, Footnote, Equation, Code, TableOfContents.
- **DetectionPredictor** (`surya/detection/__init__.py`) — Standalone segmentation model (EfficientViT-based) for text line detection. Used internally by RecognitionPredictor; no standalone CLI.
- **TableRecPredictor** (`surya/table_rec/__init__.py`) — Standalone encoder-decoder model for table structure. Two-pass: first predicts rows/columns, then predicts cells within rows.

### Key Directories
- `surya/common/` — Shared utilities: `predictor.py` (base class), `polygon.py`, model loaders (`load.py`, `s3.py`), architecture bases (`surya/`, `donut/`, `adetr/`)
- `surya/foundation/` — Foundation model: loader, processor, cache implementations (dynamic/static), utilities
- `surya/scripts/` — CLI entry points (`ocr_text.py`, `detect_layout.py`, `table_recognition.py`) and the finetuning script (`finetune_ocr.py`)
- `surya/settings.py` — All configuration via pydantic-settings `Settings` class. Every setting is overridable via environment variables.
- `surya/input/` — Image preprocessing (slicing, polygon extraction)
- `benchmark/` — Benchmark scripts for detection, recognition, layout, table recognition

### Configuration
Settings are in `surya/settings.py` using pydantic-settings. Override any setting with env vars (e.g., `TORCH_DEVICE=cuda`, `RECOGNITION_BATCH_SIZE=512`). Key batch size env vars: `DETECTOR_BATCH_SIZE`, `RECOGNITION_BATCH_SIZE`, `LAYOUT_BATCH_SIZE`, `TABLE_REC_BATCH_SIZE`. Model compilation: `COMPILE_DETECTOR`, `COMPILE_LAYOUT`, `COMPILE_TABLE_REC`, `COMPILE_ALL`.

### Test Fixtures
Tests use session-scoped fixtures in `tests/conftest.py` that load models once. Test images are generated programmatically (no external data needed for basic tests).

### SDK Usage (TableExtractionPipeline)
`surya/pipeline.py` provides `TableExtractionPipeline` — a single-class SDK for table extraction:
```python
from surya.pipeline import TableExtractionPipeline
pipeline = TableExtractionPipeline()  # loads all models once
result = pipeline.extract_tables(pil_image, ocr=True)
# result is a plain dict, JSON-serializable
```
- Accepts `PIL.Image` or raw `bytes`
- `ocr=True` runs text recognition on each detected table
- `skip_table_detection=True` treats the whole image as one table
- Returns `{"tables": [...], "image_size": [w, h]}`

### Underlying pipeline steps
1. `LayoutPredictor` — find where tables/headers are on the page
2. `TableRecPredictor` — extract row/column/cell structure from detected tables
3. `RecognitionPredictor` — read actual text content from regions
