from __future__ import annotations
from typing import Sequence
import numpy as np

def decode_rle_c_order(mask_rle: Sequence[int], shape: tuple[int, int]) -> np.ndarray:
    """按照 BrushNet 官方默认 C-order 解码 1-based start/length RLE。"""
    values = np.asarray(mask_rle, dtype=np.int64)
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError('mask_rle 必须是一维 start/length 偶数序列')
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f'非法 mask shape：{shape}')
    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths
    if np.any(starts < 0) or np.any(lengths < 0) or np.any(ends > shape[0] * shape[1]):
        raise ValueError('RLE 范围越界')
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        flat[int(start):int(end)] = 1
    return flat.reshape(shape)
