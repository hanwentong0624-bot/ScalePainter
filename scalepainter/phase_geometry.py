"""ScalePainter phase geometry. Derived from the verified research implementation."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class PhaseMetrics:
    """保存一个候选平移在多尺度网格上的完整几何指标。"""
    dx: int
    dy: int
    weighted_core_mass: float
    weighted_top1_coverage: float
    earliest_core_edge: int | None
    per_scale: tuple[dict[str, float | int], ...]

def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """把输入规范为非空、非全图的二维布尔掩码。"""
    value = np.asarray(mask)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f'mask 必须为二维数组，实际 shape={value.shape}')
    binary = value.astype(bool, copy=False)
    if not bool(binary.any()):
        raise ValueError('mask 不能为空')
    if bool(binary.all()):
        raise ValueError('mask 不能覆盖整张图')
    return np.ascontiguousarray(binary)

def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """返回掩码紧致框 ``(y0, y1, x0, x1)``，右下边界为开区间。"""
    binary = normalize_binary_mask(mask)
    ys, xs = np.nonzero(binary)
    return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)

def legal_shift_bounds(mask: np.ndarray, max_shift_px: int) -> tuple[int, int, int, int]:
    """计算既不裁掉掩码、又不超过最大位移的 ``dx/dy`` 合法范围。"""
    if int(max_shift_px) < 0:
        raise ValueError('max_shift_px 不能为负')
    binary = normalize_binary_mask(mask)
    height, width = binary.shape
    y0, y1, x0, x1 = mask_bbox(binary)
    dx_min = max(-int(max_shift_px), -x0)
    dx_max = min(int(max_shift_px), width - x1)
    dy_min = max(-int(max_shift_px), -y0)
    dy_max = min(int(max_shift_px), height - y1)
    return (dx_min, dx_max, dy_min, dy_max)

def enumerate_legal_shifts(mask: np.ndarray, *, max_shift_px: int, shift_step_px: int) -> tuple[tuple[int, int], ...]:
    """枚举规则网格上的合法平移，并保证原位置始终包含在候选中。"""
    if int(shift_step_px) <= 0:
        raise ValueError('shift_step_px 必须为正数')
    dx_min, dx_max, dy_min, dy_max = legal_shift_bounds(mask, max_shift_px)
    axis = list(range(-int(max_shift_px), int(max_shift_px) + 1, int(shift_step_px)))
    if 0 not in axis:
        axis.append(0)
    shifts = {(dx, dy) for dy in axis for dx in axis if dx_min <= dx <= dx_max and dy_min <= dy <= dy_max}
    shifts.add((0, 0))
    return tuple(sorted(shifts, key=lambda item: (item[1], item[0])))

def shift_mask_zero_fill(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """以零填充方式平移掩码；主要用于审计和可视化。"""
    binary = normalize_binary_mask(mask)
    height, width = binary.shape
    result = np.zeros_like(binary)
    source_x0 = max(0, -int(dx))
    source_x1 = min(width, width - int(dx))
    source_y0 = max(0, -int(dy))
    source_y1 = min(height, height - int(dy))
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return result
    target_x0 = source_x0 + int(dx)
    target_x1 = source_x1 + int(dx)
    target_y0 = source_y0 + int(dy)
    target_y1 = source_y1 + int(dy)
    result[target_y0:target_y1, target_x0:target_x1] = binary[source_y0:source_y1, source_x0:source_x1]
    return result

def _adaptive_bins(length: int, output_size: int) -> tuple[tuple[int, int], ...]:
    """复现 PyTorch adaptive average pooling 的离散区间定义。"""
    if int(length) <= 0 or int(output_size) <= 0:
        raise ValueError('输入尺寸与输出尺寸必须为正数')
    return tuple(((index * int(length) // int(output_size), ceil((index + 1) * int(length) / int(output_size))) for index in range(int(output_size))))

def _integral_image(mask: np.ndarray) -> np.ndarray:
    """构造带零边框的二维积分图。"""
    value = normalize_binary_mask(mask).astype(np.int64, copy=False)
    integral = np.zeros((value.shape[0] + 1, value.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = value.cumsum(axis=0).cumsum(axis=1)
    return integral

def _rect_sum(integral: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> int:
    """读取允许越界的半开矩形和；画布外区域按零处理。"""
    height = integral.shape[0] - 1
    width = integral.shape[1] - 1
    clipped_y0 = min(max(int(y0), 0), height)
    clipped_y1 = min(max(int(y1), 0), height)
    clipped_x0 = min(max(int(x0), 0), width)
    clipped_x1 = min(max(int(x1), 0), width)
    if clipped_y1 <= clipped_y0 or clipped_x1 <= clipped_x0:
        return 0
    return int(integral[clipped_y1, clipped_x1] - integral[clipped_y0, clipped_x1] - integral[clipped_y1, clipped_x0] + integral[clipped_y0, clipped_x0])

def coverage_map_from_integral(integral: np.ndarray, *, output_edge: int, dx: int=0, dy: int=0) -> np.ndarray:
    """精确计算平移掩码在一个方形 token 网格上的面积覆盖率。"""
    height = integral.shape[0] - 1
    width = integral.shape[1] - 1
    y_bins = _adaptive_bins(height, int(output_edge))
    x_bins = _adaptive_bins(width, int(output_edge))
    coverage = np.zeros((int(output_edge), int(output_edge)), dtype=np.float64)
    for row, (target_y0, target_y1) in enumerate(y_bins):
        for column, (target_x0, target_x1) in enumerate(x_bins):
            pixel_sum = _rect_sum(integral, target_y0 - int(dy), target_y1 - int(dy), target_x0 - int(dx), target_x1 - int(dx))
            token_area = (target_y1 - target_y0) * (target_x1 - target_x0)
            coverage[row, column] = float(pixel_sum) / float(token_area)
    return coverage

def _evaluate_phase_from_integral(integral: np.ndarray, *, dx: int, dy: int, formation_edges: Sequence[int], formation_weights: Sequence[float], core_threshold: float) -> PhaseMetrics:
    """复用同一积分图计算候选相位，避免搜索阶段反复扫描百万像素。"""
    if len(formation_edges) != len(formation_weights) or not formation_edges:
        raise ValueError('formation_edges 与 formation_weights 必须等长且非空')
    if not 0.0 < float(core_threshold) <= 1.0:
        raise ValueError('core_threshold 必须位于 (0,1]')
    weights = np.asarray(formation_weights, dtype=np.float64)
    if bool((weights < 0).any()) or float(weights.sum()) <= 0.0:
        raise ValueError('formation_weights 必须非负且总和为正')
    weights = weights / weights.sum()
    scale_rows: list[dict[str, float | int]] = []
    core_mass_values: list[float] = []
    top1_values: list[float] = []
    earliest_core_edge: int | None = None
    for edge in formation_edges:
        coverage = coverage_map_from_integral(integral, output_edge=int(edge), dx=int(dx), dy=int(dy))
        support = coverage > 0.0
        core = coverage >= float(core_threshold)
        total_mass = float(coverage.sum())
        core_mass = float(coverage[core].sum() / total_mass) if total_mass > 0.0 else 0.0
        top1 = float(coverage.max(initial=0.0))
        if earliest_core_edge is None and bool(core.any()):
            earliest_core_edge = int(edge)
        core_mass_values.append(core_mass)
        top1_values.append(top1)
        scale_rows.append({'edge': int(edge), 'support_tokens': int(support.sum()), 'core_tokens': int(core.sum()), 'core_mass_fraction': core_mass, 'top1_coverage': top1, 'mean_support_coverage': float(coverage[support].mean()) if bool(support.any()) else 0.0})
    return PhaseMetrics(dx=int(dx), dy=int(dy), weighted_core_mass=float(np.dot(weights, np.asarray(core_mass_values))), weighted_top1_coverage=float(np.dot(weights, np.asarray(top1_values))), earliest_core_edge=earliest_core_edge, per_scale=tuple(scale_rows))
