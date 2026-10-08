from __future__ import annotations
from typing import Any
import numpy as np
from PIL import Image
from scipy import ndimage
EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)

def _as_rgb_array(value: Image.Image | np.ndarray) -> np.ndarray:
    """把 PIL/NumPy 输入规范成连续的 ``uint8 HxWx3`` RGB 数组。"""
    if isinstance(value, Image.Image):
        array = np.asarray(value.convert('RGB'), dtype=np.uint8)
    else:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f'EditBench RGB 输入形状非法：{array.shape}')
        array = np.asarray(array[..., :3], dtype=np.uint8)
    return np.ascontiguousarray(array)

def _resize_reference(reference: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """仅为分量判别把参考图对齐到原始黑洞输入分辨率。"""
    height, width = target_shape
    if reference.shape[:2] == (height, width):
        return reference
    resized = Image.fromarray(reference, mode='RGB').resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)

def _mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    """计算 mask 面积、包围框、触边状态和等效宽度等审计量。"""
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError('EditBench 重建 mask 为空')
    y0, y1 = (int(ys.min()), int(ys.max()) + 1)
    x0, x1 = (int(xs.min()), int(xs.max()) + 1)
    height, width = mask.shape
    bbox_height = y1 - y0
    bbox_width = x1 - x0
    area_pixels = int(mask.sum())
    touches_border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return {'height': int(height), 'width': int(width), 'area_pixels': area_pixels, 'area_ratio': float(area_pixels / (height * width)), 'bbox_xyxy': [x0, y0, x1, y1], 'bbox_width': int(bbox_width), 'bbox_height': int(bbox_height), 'bbox_area_ratio': float(bbox_width * bbox_height / (height * width)), 'bbox_aspect_ratio': float(bbox_width / max(bbox_height, 1)), 'equivalent_width_pixels': float(np.sqrt(area_pixels)), 'touches_image_border': touches_border}

def extract_editbench_hole_mask(masked_input: Image.Image | np.ndarray, reference_image: Image.Image | np.ndarray, *, difference_threshold: float=8.0) -> tuple[np.ndarray, dict[str, Any]]:
    """从 EditBench 的黑洞输入中重建唯一 placement mask。

    Args:
        masked_input: EditBench ``masks_*.png`` 中的 RGB 黑洞输入。
        reference_image: 同一案例的 ``references_*.png`` 参考图。
        difference_threshold: 选择连通分量时使用的平均 RGB 差异阈值。该值只影响
            “选择哪个纯黑分量”，不会改变最终分量的像素边界。

    Returns:
        ``(mask, audit)``。``mask`` 是原始分辨率的布尔数组；``audit`` 保存可复核
        的分量、差异证据和几何统计。
    """
    masked = _as_rgb_array(masked_input)
    reference = _resize_reference(_as_rgb_array(reference_image), masked.shape[:2])
    exact_black = np.max(masked, axis=2) == 0
    if not bool(exact_black.any()):
        raise ValueError('EditBench 黑洞输入中不存在严格纯黑像素')
    difference = np.abs(reference.astype(np.int16) - masked.astype(np.int16)).mean(axis=2)
    difference_seed = exact_black & (difference >= float(difference_threshold))
    labels, component_count = ndimage.label(exact_black, structure=EIGHT_CONNECTED)
    component_sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    component_scores = np.bincount(labels.ravel(), weights=difference_seed.ravel().astype(np.float64), minlength=component_count + 1)
    component_sizes[0] = 0
    component_scores[0] = 0.0
    candidates = np.arange(1, component_count + 1, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError('EditBench 黑洞输入未形成有效连通分量')
    order = sorted(candidates.tolist(), key=lambda index: (float(component_scores[index]), int(component_sizes[index]), -int(index)), reverse=True)
    selected_label = int(order[0])
    selected_raw = labels == selected_label
    selected = np.asarray(ndimage.binary_fill_holes(selected_raw), dtype=bool)
    geometry = _mask_geometry(selected)
    selected_seed_count = int((selected_raw & difference_seed).sum())
    selected_raw_count = int(selected_raw.sum())
    exact_black_count = int(exact_black.sum())
    audit: dict[str, Any] = {'method': 'exact_black_component_ranked_by_reference_difference_v2', 'difference_threshold_mean_rgb': float(difference_threshold), 'black_definition': 'max_rgb_equals_0', 'connectivity': 8, 'component_count': int(component_count), 'selected_component_label': selected_label, 'selected_component_raw_pixels': selected_raw_count, 'selected_component_filled_pixels': int(selected.sum()), 'selected_component_difference_seed_pixels': selected_seed_count, 'selected_component_seed_fraction': float(selected_seed_count / max(selected_raw_count, 1)), 'selected_fraction_of_all_exact_black': float(selected_raw_count / max(exact_black_count, 1)), 'all_exact_black_pixels': exact_black_count, 'all_exact_black_ratio': float(exact_black.mean()), 'reference_resized_for_component_selection': bool(_as_rgb_array(reference_image).shape[:2] != masked.shape[:2]), **geometry}
    return (selected, audit)

def resize_binary_mask(mask: np.ndarray, size: int) -> Image.Image:
    """用最近邻把布尔 mask 对齐到冻结的正方形生成分辨率。"""
    if mask.ndim != 2:
        raise ValueError(f'二值 mask 必须为二维数组：{mask.shape}')
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
    if image.size != (int(size), int(size)):
        image = image.resize((int(size), int(size)), Image.Resampling.NEAREST)
    return image
