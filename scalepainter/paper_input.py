"""ScalePainter paper input. Derived from the verified research implementation."""
from __future__ import annotations
import numpy as np
from PIL import Image
from scipy import ndimage

def native_unknown_mask(mask: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = native_shape
    mh, mw = mask.shape
    if mh % h or mw % w or mh < h or (mw < w):
        raise ValueError('E146a only accepts exact integer upscaling')
    unknown = mask.reshape(h, mh // h, w, mw // w).any(axis=(1, 3))
    if not unknown.any() or unknown.all():
        raise ValueError('Native visibility must contain both known and unknown pixels')
    return unknown

def build_safe_condition(native_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Nearest-visible native fill, then the original LANCZOS resize.

    The generator subsequently applies its unchanged final-grid nearest proxy.
    No original hidden RGB reaches interpolation or an encoder.
    """
    rgb = np.asarray(native_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 HxWx3 RGB')
    unknown = native_unknown_mask(mask, rgb.shape[:2])
    known = rgb.copy()
    known[unknown] = 0
    _, donors = ndimage.distance_transform_edt(unknown, return_indices=True)
    yy, xx = (donors[0][unknown], donors[1][unknown])
    assert not unknown[yy, xx].any()
    native_proxy = known.copy()
    native_proxy[unknown] = known[yy, xx]
    height, width = mask.shape
    condition = np.asarray(Image.fromarray(native_proxy).resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8).copy()
    return (condition, {'method': 'native_nearest_before_resize_then_unchanged_final_nearest', 'native_shape': list(rgb.shape[:2]), 'output_shape': [height, width], 'native_unknown_pixels': int(unknown.sum()), 'native_mask_rule': 'any_overlap_of_native_pixel_footprint', 'hidden_rgb_zeroed_before_first_resampling': True, 'donors_outside_native_unknown': True, 'placement_mask_modified': False})
