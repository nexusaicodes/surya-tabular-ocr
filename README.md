# Surya Tabular OCR

A trimmed fork of [Surya](https://github.com/VikParuchuri/surya) focused on **English-only OCR, layout analysis, and table recognition** from document images. Programmatic use only — no CLI.

## Installation

Requires Python 3.11+ and PyTorch. You may need to install the CPU version of torch first if you're not using a Mac or a GPU machine. See [here](https://pytorch.org/get-started/locally/) for more details.

```shell
pip install surya-tabular-ocr
# or
uv add surya-tabular-ocr
```

Model weights download automatically on first use.

## Usage

### Table extraction pipeline (recommended)

Single-call interface that runs layout detection, table recognition, and OCR together:

```python
from surya.pipeline import TableExtractionPipeline

pipeline = TableExtractionPipeline()  # loads all models once
result = pipeline.extract_tables(image, ocr=True)
# result is a plain dict: {"tables": [...], "image_size": [w, h]}
```

- Accepts `PIL.Image` or raw `bytes`
- `ocr=True` runs text recognition on each detected table
- `skip_table_detection=True` treats the whole image as one table

### Individual predictors

**OCR:**

```python
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

image = Image.open("doc.png")
recognition = RecognitionPredictor(FoundationPredictor())
detection = DetectionPredictor()

predictions = recognition([image], det_predictor=detection)
```

**Layout analysis:**

```python
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor

image = Image.open("doc.png")
layout = LayoutPredictor(FoundationPredictor())

predictions = layout([image])
```

**Table recognition:**

```python
from PIL import Image
from surya.table_rec import TableRecPredictor

image = Image.open("table.png")
table_rec = TableRecPredictor()

predictions = table_rec([image])
```

### Configuration

All settings are in `surya/settings.py` and overridable via environment variables:

- `TORCH_DEVICE` — override auto-detected device (e.g. `cuda`)
- `RECOGNITION_BATCH_SIZE`, `DETECTOR_BATCH_SIZE`, `LAYOUT_BATCH_SIZE`, `TABLE_REC_BATCH_SIZE`
- `COMPILE_DETECTOR`, `COMPILE_LAYOUT`, `COMPILE_TABLE_REC`, `COMPILE_ALL` — enable torch compilation

## Development

```bash
git clone https://github.com/nexusaicodes/surya-tabular-ocr.git
cd surya-tabular-ocr
uv sync --group dev
pre-commit install          # enable ruff linting/formatting on commit
uv run pytest
```

## License

Code is GPL-3.0-or-later (inherited from upstream). Model weights use a modified AI Pubs Open Rail-M license. See [LICENSE](LICENSE) and [MODEL_LICENSE](MODEL_LICENSE).

## Acknowledgments

Trimmed fork of [Surya](https://github.com/VikParuchuri/surya) by Vik Paruchuri and the Datalab team.
