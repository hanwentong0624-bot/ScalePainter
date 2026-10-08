"""ScalePainter proxy. Derived from the verified research implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy import ndimage

@dataclass(frozen=True)
class ContextProxyResult:
    """一次代理构造结果及其无泄漏审计。"""
    image: np.ndarray
    audit: dict[str, Any]

def _validate(source: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把输入规范为连续 float32 RGB 与 bool mask，并拒绝空/全图 mask。"""
    image = np.asarray(source)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'source 必须为 HxWx3 RGB，实际 {image.shape}')
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32, copy=False)
    if not bool(np.isfinite(image).all()) or float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError('source 必须是 [0,1] 内有限 RGB')
    binary = np.asarray(mask).astype(bool, copy=False)
    if binary.shape != image.shape[:2]:
        raise ValueError(f'mask/source shape 不一致：{binary.shape} != {image.shape[:2]}')
    if not bool(binary.any()) or bool(binary.all()):
        raise ValueError('context proxy 要求 mask 非空且不能覆盖全图')
    return (np.ascontiguousarray(image), np.ascontiguousarray(binary))

def _base_audit(method: str, source: np.ndarray, mask: np.ndarray, proxy: np.ndarray) -> dict[str, Any]:
    """生成统一的像素级恒等与数值审计。"""
    outside_error = np.abs(proxy[~mask] - source[~mask])
    inside = proxy[mask]
    return {'method': str(method), 'height': int(mask.shape[0]), 'width': int(mask.shape[1]), 'mask_pixels': int(mask.sum()), 'mask_ratio': float(mask.mean()), 'outside_max_abs_error': float(outside_error.max(initial=0.0)), 'outside_exact': bool(np.array_equal(proxy[~mask], source[~mask])), 'inside_mean_rgb': [float(v) for v in inside.mean(axis=0)], 'inside_std_rgb': [float(v) for v in inside.std(axis=0)], 'finite': bool(np.isfinite(proxy).all()), 'range_min': float(proxy.min()), 'range_max': float(proxy.max()), 'reads_masked_source_pixels': False, 'external_model_used': False}

def build_nearest_proxy(source: np.ndarray, mask: np.ndarray) -> ContextProxyResult:
    """仅用最近的 mask 外像素填充未知区域。

    ``distance_transform_edt`` 对 True 区域返回最近 False 像素的坐标；因此索引
    只能落在 ``~mask``。原始 mask 内值在索引前先被清零，额外防止未来代码改动
    意外使用这些值。
    """
    image, binary = _validate(source, mask)
    known_only = image.copy()
    known_only[binary] = 0.0
    _, indices = ndimage.distance_transform_edt(binary, return_indices=True)
    nearest_y = indices[0][binary]
    nearest_x = indices[1][binary]
    if bool(binary[nearest_y, nearest_x].any()):
        raise AssertionError('nearest proxy 的采样坐标落入了 mask')
    proxy = image.copy()
    proxy[binary] = known_only[nearest_y, nearest_x]
    audit = _base_audit('nearest', image, binary, proxy)
    distances = ndimage.distance_transform_edt(binary)
    audit.update({'maximum_context_distance_px': float(distances[binary].max(initial=0.0)), 'mean_context_distance_px': float(distances[binary].mean()), 'sampling_coordinates_outside_mask': True})
    return ContextProxyResult(np.ascontiguousarray(proxy), audit)
