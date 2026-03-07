import math
from typing import List
import torch

import torch.nn.functional as F

from surya.common.polygon import PolygonBox
from surya.settings import settings


def get_nearest_pad(
    length: int, pad_multiple: int = settings.FOUNDATION_PAD_TO_NEAREST
):
    return math.ceil(length / pad_multiple) * pad_multiple


def clean_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
        xs = [point[0] for point in box_obj.polygon]
        ys = [point[1] for point in box_obj.polygon]
        if max(xs) == min(xs) or max(ys) == min(ys):
            continue

        box = box_obj.bbox
        contained = False
        for other_box_obj in boxes:
            if other_box_obj.polygon == box_obj.polygon:
                continue

            other_box = other_box_obj.bbox
            if box == other_box:
                continue
            if (
                box[0] >= other_box[0]
                and box[1] >= other_box[1]
                and box[2] <= other_box[2]
                and box[3] <= other_box[3]
            ):
                contained = True
                break
        if not contained:
            new_boxes.append(box_obj)
    return new_boxes


def expand_bbox(bbox, expansion_factor=0.01):
    expansion_low = 1 - expansion_factor
    expansion_high = 1 + expansion_factor
    return [
        bbox[0] * expansion_low,
        bbox[1] * expansion_low,
        bbox[2] * expansion_high,
        bbox[3] * expansion_high,
    ]


def is_flash_attn_2_supported(device: str | torch.device) -> bool:
    if not torch.cuda.is_available():
        return False

    if "cuda" not in str(device):
        return False

    # Check CUDA version >= 12.0
    cuda_version_str = torch.version.cuda
    if cuda_version_str is None:
        return False
    cuda_version = tuple(map(int, cuda_version_str.split(".")))
    if cuda_version < (12, 0):
        return False

    # Check GPU compute capability (Ampere, Ada, Hopper GPUs)
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major + minor / 10
    if compute_capability < 8.0:
        return False

    return True


def pad_to_batch_size_repeat(tensor: torch.Tensor, batch_size: int):
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    if pad_size < 0:
        return tensor

    # Repeat the last row pad_size times
    last_row = tensor[-1:].repeat(pad_size, 1, 1)

    # Concatenate original tensor with repeated last rows
    return torch.cat([tensor, last_row], dim=0)


def pad_to_batch_size(tensor: torch.Tensor, batch_size: int):
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)

    return F.pad(tensor, padding, mode="constant", value=0)
