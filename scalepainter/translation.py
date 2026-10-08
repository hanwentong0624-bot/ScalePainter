"""ScalePainter translation. Derived from the verified research implementation."""
from __future__ import annotations
import numpy as np

def _validate_shift(dx: int, dy: int, height: int, width: int) -> tuple[int, int]:
    """验证整数平移不超过画布尺寸，并返回规范化后的整数。"""
    shift_x = int(dx)
    shift_y = int(dy)
    if abs(shift_x) >= int(width) or abs(shift_y) >= int(height):
        raise ValueError(f'平移不能跨越完整画布：dx={shift_x}, dy={shift_y}, height={height}, width={width}')
    return (shift_x, shift_y)

def shift_image_reflect(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """以反射填充执行整数平移，保持 dtype、通道数和空间尺寸不变。"""
    value = np.asarray(image)
    if value.ndim not in (2, 3):
        raise ValueError(f'图像必须为 [H,W] 或 [H,W,C]，实际 {value.shape}')
    height, width = value.shape[:2]
    shift_x, shift_y = _validate_shift(dx, dy, height, width)
    if shift_x == 0 and shift_y == 0:
        return np.ascontiguousarray(value.copy())
    pad_x = abs(shift_x)
    pad_y = abs(shift_y)
    if value.ndim == 2:
        pad_width: tuple[tuple[int, int], ...] = ((pad_y, pad_y), (pad_x, pad_x))
    else:
        pad_width = ((pad_y, pad_y), (pad_x, pad_x), (0, 0))
    padded = np.pad(value, pad_width, mode='reflect')
    start_y = pad_y - shift_y
    start_x = pad_x - shift_x
    shifted = padded[start_y:start_y + height, start_x:start_x + width]
    return np.ascontiguousarray(shifted)

def exact_inverse_valid_mask(height: int, width: int, dx: int, dy: int) -> np.ndarray:
    """返回先平移再逆平移后严格来自原像素、未经过反射填充的区域。"""
    shift_x, shift_y = _validate_shift(dx, dy, int(height), int(width))
    valid = np.ones((int(height), int(width)), dtype=bool)
    if shift_x > 0:
        valid[:, int(width) - shift_x:] = False
    elif shift_x < 0:
        valid[:, :-shift_x] = False
    if shift_y > 0:
        valid[int(height) - shift_y:, :] = False
    elif shift_y < 0:
        valid[:-shift_y, :] = False
    return valid

def inverse_shift_reflect(shifted_image: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """把平移坐标系中的图像移回原坐标，并返回精确可逆区域。"""
    value = np.asarray(shifted_image)
    restored = shift_image_reflect(value, -int(dx), -int(dy))
    valid = exact_inverse_valid_mask(value.shape[0], value.shape[1], int(dx), int(dy))
    return (restored, valid)

def inverse_shift_with_source_fill(shifted_image: np.ndarray, source_image: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """逆平移后仅用输入源图补齐不可逆边条，作为边缘填充敏感性结果。"""
    restored, valid = inverse_shift_reflect(shifted_image, dx, dy)
    source = np.asarray(source_image)
    if source.shape != restored.shape:
        raise ValueError(f'源图与生成图 shape 不一致：{source.shape} != {restored.shape}')
    filled = np.asarray(restored).copy()
    filled[~valid] = source[~valid]
    return (np.ascontiguousarray(filled), valid)

def build_strict_composite(source_image: np.ndarray, edited_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """只在原 placement mask 内提交编辑结果，mask 外逐元素复用输入源图。"""
    source = np.asarray(source_image)
    edited = np.asarray(edited_image)
    binary = np.asarray(mask).astype(bool, copy=False)
    if source.shape != edited.shape:
        raise ValueError(f'source/edited shape 不一致：{source.shape} != {edited.shape}')
    if binary.shape != source.shape[:2]:
        raise ValueError(f'mask shape 不一致：{binary.shape} != {source.shape[:2]}')
    result = source.copy()
    result[binary] = edited[binary]
    return np.ascontiguousarray(result)
