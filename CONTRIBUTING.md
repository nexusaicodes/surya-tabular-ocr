# Contributing to surya-tabular-ocr

## Scope

This is a trimmed fork of [Surya](https://github.com/VikParuchuri/surya) focused on **English-only OCR, layout analysis, and table recognition** from document images. The following are intentionally excluded and PRs re-adding them will not be accepted:

- Multilingual / non-English support
- Math / LaTeX recognition
- CLI interfaces or Streamlit UIs
- OCR error detection

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nexusaicodes/surya-tabular-ocr.git
cd surya-tabular-ocr
uv sync --group dev
pre-commit install
```

## Running tests

```bash
uv run pytest                                        # all tests
uv run pytest tests/test_recognition.py              # single file
uv run pytest tests/test_recognition.py::test_name   # single test
```

Tests use session-scoped fixtures that load models once. Test images are generated programmatically — no external data needed.

## Code style

Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/) via pre-commit hooks. To run manually:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Pull requests

- Keep PRs focused — one logical change per PR.
- Include tests for new functionality.
- Make sure `uv run pytest` passes before opening a PR.
- Describe _what_ and _why_ in the PR description.

## Architecture overview

All models follow a predictor pattern (`surya/common/predictor.py:BasePredictor`). The main entry points are:

| Predictor | Purpose |
|---|---|
| `RecognitionPredictor` | OCR (text line detection + recognition) |
| `LayoutPredictor` | Page layout analysis + reading order |
| `TableRecPredictor` | Table structure (rows, columns, cells) |
| `DetectionPredictor` | Text line segmentation (used internally) |
| `FoundationPredictor` | Shared autoregressive model (used by recognition + layout) |

`surya/pipeline.py:TableExtractionPipeline` composes these into a single-call table extraction API.

Configuration lives in `surya/settings.py` (pydantic-settings). Override any setting via environment variables.

## License

By contributing you agree that your contributions will be licensed under GPL-3.0-or-later, consistent with the project license.
