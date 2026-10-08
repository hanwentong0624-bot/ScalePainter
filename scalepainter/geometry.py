"""Safe native-image adaptation, separate from the frozen paper geometry."""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .paper_input import build_safe_condition


@dataclass(frozen=True)
class Geometry:
    native_height: int
    native_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int
    canvas_size: int
    mode: str


def resize_mask_conservative(mask, height, width):
    """Mark a destination pixel unknown if its source footprint overlaps a mask pixel."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    y0 = np.arange(height, dtype=np.int64) * h // height
    y1 = ((np.arange(height, dtype=np.int64) + 1) * h + height - 1) // height
    x0 = np.arange(width, dtype=np.int64) * w // width
    x1 = ((np.arange(width, dtype=np.int64) + 1) * w + width - 1) // width
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    counts = integral[y1[:, None], x1] - integral[y0[:, None], x1] - integral[y1[:, None], x0] + integral[y0[:, None], x0]
    return counts > 0


def prepare_input(rgb, mask, *, mode='generic', canvas_size=1024):
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, dtype=bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected native uint8 RGB image')
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError('The mask must contain both known and unknown pixels')
    h, w = rgb.shape[:2]
    if mode == 'paper':
        if mask.shape != (1024, 1024) or canvas_size != 1024:
            raise ValueError('Paper mode requires the original 1024 x 1024 prepared placement mask')
        safe, audit = build_safe_condition(rgb, mask)
        geometry = Geometry(h, w, 1024, 1024, 0, 0, 1024, mode)
        return safe, mask.copy(), geometry, audit
    if mode != 'generic':
        raise ValueError(f'Unknown geometry mode: {mode}')
    if mask.shape != (h, w):
        raise ValueError('Generic mode requires an image and mask with equal native dimensions')
    if canvas_size <= 0:
        raise ValueError('Canvas size must be positive')
    safe = rgb.copy()
    safe[mask] = 0
    _, donors = ndimage.distance_transform_edt(mask, return_indices=True)
    safe[mask] = safe[donors[0][mask], donors[1][mask]]
    factor = canvas_size / max(h, w)
    rh, rw = max(1, round(h * factor)), max(1, round(w * factor))
    # No original hidden RGB can reach this first resampling operation.
    resized = np.asarray(Image.fromarray(safe).resize((rw, rh), Image.Resampling.LANCZOS))
    resized_mask = resize_mask_conservative(mask, rh, rw)
    top, left = (canvas_size - rh) // 2, (canvas_size - rw) // 2
    canvas = np.pad(resized, ((top, canvas_size-rh-top), (left, canvas_size-rw-left), (0, 0)), mode='edge')
    canvas_mask = np.pad(resized_mask, ((top, canvas_size-rh-top), (left, canvas_size-rw-left)))
    if canvas_mask.all():
        raise ValueError('No known context remains at model resolution; use a larger visible region')
    geometry = Geometry(h, w, rh, rw, top, left, canvas_size, mode)
    return canvas, canvas_mask, geometry, {'hidden_rgb_zeroed_before_first_resampling': True, 'mask_rule': 'any_source_footprint_overlap', 'geometry': 'aspect_preserving_letterbox'}


def restore_native(raw, geometry):
    if geometry.mode == 'paper':
        return raw
    g = geometry
    crop = np.asarray(raw, dtype=np.float32)[g.top:g.top+g.resized_height, g.left:g.left+g.resized_width]
    channels = [np.asarray(Image.fromarray(crop[..., c]).resize((g.native_width, g.native_height), Image.Resampling.LANCZOS)) for c in range(3)]
    return np.stack(channels, axis=-1).clip(0, 1).astype(np.float32)
