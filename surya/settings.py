import os
from typing import Callable, Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
from pathlib import Path
from platformdirs import user_cache_dir


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI: int = 96  # Used for detection, layout, reading order
    IMAGE_DPI_HIGHRES: int = 192  # Used for OCR, table rec
    FLATTEN_PDF: bool = True  # Flatten PDFs by merging form fields before processing
    DISABLE_TQDM: bool = False  # Disable tqdm progress bars
    S3_BASE_URL: str = "https://models.datalab.to"
    PARALLEL_DOWNLOAD_WORKERS: int = (
        10  # Number of workers for parallel model downloads
    )
    MODEL_CACHE_DIR: str = str(Path(user_cache_dir("datalab")) / "models")
    LOGLEVEL: str = "INFO"  # Logging level

    # Paths
    DATA_DIR: str = "data"
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    # Text detection
    DETECTOR_BATCH_SIZE: Optional[int] = None  # Defaults to 2 for CPU/MPS, 32 otherwise
    DETECTOR_MODEL_CHECKPOINT: str = "s3://text_detection/2025_05_07"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = (
        1400  # Height at which to slice images vertically
    )
    DETECTOR_TEXT_THRESHOLD: float = (
        0.6  # Threshold for text detection (above this is considered text)
    )
    DETECTOR_BLANK_THRESHOLD: float = (
        0.35  # Threshold for blank space (below this is considered blank)
    )
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(
        8, os.cpu_count()
    )  # Number of workers for postprocessing
    DETECTOR_MIN_PARALLEL_THRESH: int = (
        3  # Minimum number of images before we parallelize
    )
    DETECTOR_BOX_Y_EXPAND_MARGIN: float = (
        0.05  # Margin by which to expand detected boxes vertically
    )
    COMPILE_DETECTOR: bool = False

    # Text recognition
    FOUNDATION_MODEL_CHECKPOINT: str = "s3://text_recognition/2025_09_23"
    FOUNDATION_MODEL_QUANTIZE: bool = False
    FOUNDATION_MAX_TOKENS: Optional[int] = None
    FOUNDATION_CHUNK_SIZE: Optional[int] = None
    FOUNDATION_PAD_TO_NEAREST: int = 256
    COMPILE_FOUNDATION: bool = False
    FOUNDATION_MULTI_TOKEN_MIN_CONFIDENCE: float = 0.9

    RECOGNITION_MODEL_CHECKPOINT: str = "s3://text_recognition/2025_09_23"
    RECOGNITION_BATCH_SIZE: Optional[int] = (
        None  # Defaults to 8 for CPU/MPS, 256 otherwise
    )
    RECOGNITION_PAD_VALUE: int = 255  # Should be 0 or 255

    # Layout
    LAYOUT_MODEL_CHECKPOINT: str = "s3://layout/2025_09_23"
    LAYOUT_IMAGE_SIZE: Dict = {"height": 768, "width": 768}
    LAYOUT_SLICE_MIN: Dict = {
        "height": 1500,
        "width": 1500,
    }  # When to start slicing images
    LAYOUT_SLICE_SIZE: Dict = {"height": 1200, "width": 1200}  # Size of slices
    LAYOUT_BATCH_SIZE: Optional[int] = None
    LAYOUT_MAX_BOXES: int = 100
    COMPILE_LAYOUT: bool = False

    # Table Rec
    TABLE_REC_MODEL_CHECKPOINT: str = "s3://table_recognition/2025_02_18"
    TABLE_REC_IMAGE_SIZE: Dict = {"height": 768, "width": 768}
    TABLE_REC_MAX_BOXES: int = 150
    TABLE_REC_BATCH_SIZE: Optional[int] = None
    COMPILE_TABLE_REC: bool = False

    COMPILE_ALL: bool = False

    @computed_field
    def DETECTOR_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_DETECTOR

    @computed_field
    def LAYOUT_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_LAYOUT

    @computed_field
    def FOUNDATION_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_FOUNDATION

    @computed_field
    def TABLE_REC_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_TABLE_REC

    @computed_field
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        return torch.float16

    @computed_field
    def MODEL_DTYPE_BFLOAT(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        return torch.bfloat16

    @computed_field
    def INFERENCE_MODE(self) -> Callable:
        return torch.inference_mode

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
